"""Persistent local corpus embeddings: SQLite text/index plus memory-mapped vectors.

A completed cache is self-contained and never depends on the original uploads.
Only completed directories are listed. Model configuration is immutable per cache.
"""
from __future__ import annotations
import hashlib
import json
import shutil
import sqlite3
import uuid
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from .data_loading import CorpusSentence, corpus_sentences, select_usages
from .embeddings import LoadedModel, EmbeddedUsages, iter_token_embeddings, model_metadata, pool_span

SCHEMA_VERSION = 1


@dataclass
class StorageEstimate:
    sentences: int
    tokens: int
    dimensions: int
    dtype: str
    vector_bytes: int
    text_bytes: int
    index_bytes: int

    @property
    def estimated_bytes(self):
        return self.vector_bytes + self.text_bytes + self.index_bytes


def estimate_storage(records: list[CorpusSentence], tokenizer, dimensions: int, *, dtype="float32", storage="tokens", raw_text_bytes=0):
    if dtype not in {"float16", "float32"} or storage not in {"tokens", "sentences"}:
        raise ValueError("Choose float16/float32 and tokens/sentences storage.")
    if dimensions < 1:
        raise ValueError("The model must specify its embedding dimension.")
    count = 0
    for start in range(0, len(records), 64):
        encoded = tokenizer([r.sentence for r in records[start:start+64]], add_special_tokens=False,
                            truncation=False, return_attention_mask=False)
        count += sum(len(ids) for ids in encoded['input_ids'])
    vectors = count if storage == "tokens" else len(records)
    text_bytes = raw_text_bytes + sum(len(r.sentence.encode()) for r in records)
    # SQLite page/index overhead is approximate; numeric payload sizes are exact estimates.
    return StorageEstimate(len(records), count, dimensions, dtype, vectors*dimensions*np.dtype(dtype).itemsize,
                           text_bytes, (count*16 if storage == "tokens" else 0) + len(records)*512 + 65536)


def format_bytes(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1000 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1000


def create_corpus_cache(root: str | Path, name: str, texts: dict[str,str], loaded_model: LoadedModel, *,
                         storage="tokens", dtype="float32", layer=-1, max_length=256, batch_size=16,
                         lines=False, progress_callback=None, estimate: StorageEstimate | None = None) -> Path:
    if not name.strip():
        raise ValueError("Give the saved corpora a name.")
    records = corpus_sentences(texts, lines=lines)
    metadata = model_metadata(loaded_model, layer=layer, max_length=max_length)
    if estimate is None:
        estimate = estimate_storage(records, loaded_model.tokenizer, metadata['dimension'], dtype=dtype, storage=storage,
                                    raw_text_bytes=sum(len(t.encode()) for t in texts.values()))
    if (estimate.sentences != len(records) or estimate.dimensions != metadata['dimension'] or estimate.dtype != dtype):
        raise ValueError("The storage estimate is stale. Estimate again with the current inputs.")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    required = estimate.estimated_bytes
    if shutil.disk_usage(root).free < required * 1.1:
        raise ValueError(f"Not enough free disk space. Estimated storage: {format_bytes(required)}.")
    key = uuid.uuid4().hex
    temp, final = root / f'.{key}.incomplete', root / key
    temp.mkdir()
    vectors = offsets_file = None
    connection = None
    try:
        capacity = estimate.tokens if storage == "tokens" else len(records)
        vectors = np.lib.format.open_memmap(temp/'vectors.npy', mode='w+', dtype=dtype,
                                            shape=(capacity, metadata['dimension']))
        if storage == "tokens":
            offsets_file = np.lib.format.open_memmap(temp/'offsets.npy', mode='w+', dtype='int64', shape=(capacity,2))
        connection = sqlite3.connect(temp/'index.sqlite')
        connection.executescript('''
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE corpora (name TEXT PRIMARY KEY, position INTEGER, source_text TEXT, sha256 TEXT);
            CREATE TABLE sentences (id TEXT PRIMARY KEY, position INTEGER, corpus TEXT, sentence TEXT,
                                    vector_start INTEGER, vector_count INTEGER);
            CREATE INDEX sentence_corpus ON sentences(corpus);
        ''')
        for i,(corpus,text) in enumerate(texts.items()):
            connection.execute('INSERT INTO corpora VALUES (?,?,?,?)',
                               (corpus,i,text,hashlib.sha256(text.encode()).hexdigest()))
        position = 0
        for i, offsets, token_vectors in iter_token_embeddings(loaded_model,[r.sentence for r in records],
                        batch_size=batch_size, layer=layer, max_length=max_length, progress_callback=progress_callback):
            array = token_vectors.numpy() if storage == "tokens" else token_vectors.mean(0,keepdim=True).numpy()
            count = len(array)
            if position + count > capacity:
                raise ValueError("Tokenizer produced more vectors than estimated. No incomplete cache was saved.")
            cast = array.astype(dtype)
            if not np.isfinite(cast).all():
                raise ValueError("Vectors overflow the selected precision. Retry with float32 storage.")
            vectors[position:position+count] = cast
            if offsets_file is not None:
                offsets_file[position:position+count] = offsets
            record = records[i]
            connection.execute('INSERT INTO sentences VALUES (?,?,?,?,?,?)',
                               (record.sentence_id,i,record.corpus,record.sentence,position,count))
            position += count
        info = {"schema_version": SCHEMA_VERSION, "name": name.strip(), "storage": storage, "dtype": dtype,
                "model": metadata, "sentences": len(records), "vectors": position, "capacity": capacity,
                "corpus_order": list(texts), "splitter": "lines_v1" if lines else "punctuation_v1"}
        connection.execute('INSERT INTO metadata VALUES (?,?)',('info',json.dumps(info)))
        connection.commit()
        connection.close()
        connection = None
        vectors.flush()
        if offsets_file is not None:
            offsets_file.flush()
        # Release mappings before renaming, including on Windows.
        del vectors, offsets_file
        vectors = offsets_file = None
        temp.rename(final)
        return final
    except Exception:
        if connection is not None:
            connection.close()
        del vectors, offsets_file
        shutil.rmtree(temp,ignore_errors=True)
        raise


class CorpusCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not (self.path/'index.sqlite').is_file():
            raise ValueError("Saved corpus index is missing.")
        with closing(self._connect()) as con:
            row = con.execute("SELECT value FROM metadata WHERE key='info'").fetchone()
            if row is None:
                raise ValueError("Saved corpus metadata is missing.")
            self.info = json.loads(row[0])
            if self.info.get('schema_version') != SCHEMA_VERSION:
                raise ValueError("Unsupported saved corpus version.")
            self.rows = con.execute('SELECT id,corpus,sentence,vector_start,vector_count FROM sentences ORDER BY position').fetchall()
        self.vectors = np.load(self.path/'vectors.npy', mmap_mode='r', allow_pickle=False)
        self.offsets = np.load(self.path/'offsets.npy', mmap_mode='r', allow_pickle=False) if self.info['storage']=='tokens' else None
        if self.vectors.shape != (self.info['capacity'],self.info['model']['dimension']):
            raise ValueError("Saved vector dimensions do not match the index.")
        if self.offsets is not None and self.offsets.shape != (self.info['capacity'], 2):
            raise ValueError("Saved token offsets do not match the index.")
        if len(self.rows) != self.info['sentences']:
            raise ValueError("Saved corpus is incomplete.")
        for _,_,_,start,count in self.rows:
            if count < 1 or start < 0 or start+count > self.info['vectors']:
                raise ValueError("Invalid sentence offsets in saved corpus.")
        self._by_id = {row[0]:row for row in self.rows}

    def _connect(self):
        return sqlite3.connect(self.path.joinpath('index.sqlite').resolve().as_uri()+'?mode=ro',uri=True)

    @property
    def sentences(self):
        return [CorpusSentence(c,s,sid) for sid,c,s,_,_ in self.rows]

    def source_texts(self):
        with closing(self._connect()) as con:
            return dict(con.execute('SELECT name,source_text FROM corpora ORDER BY position'))

    def select(self, word: str, forms=None, *, representation="word", pooling="mean", max_per_corpus=0, seed=42):
        if representation == 'word' and self.offsets is None:
            raise ValueError("This cache contains sentence vectors only. Create a token cache for word representations.")
        if representation not in {'word','sentence'}:
            raise ValueError("Choose word or sentence representations.")
        data = select_usages(self.sentences,word,forms)
        if max_per_corpus:
            rng=np.random.default_rng(seed)
            kept=[]
            for c in data.corpus_labels:
                ids=data.indices_for_corpus(c)
                kept.extend(sorted(rng.choice(ids,min(max_per_corpus,len(ids)),replace=False).tolist()))
            data=data.subset(sorted(kept))
        out=[]
        for u in data.usages:
            _,_,_,start,count=self._by_id[u.sentence_id]
            vectors=torch.from_numpy(np.array(self.vectors[start:start+count],dtype=np.float32))
            if representation=='sentence':
                out.append(vectors.mean(0))
            else:
                out.append(pool_span(vectors,np.asarray(self.offsets[start:start+count]),u.char_span,u.sentence,pooling))
        metadata={**self.info['model'], 'representation':representation, 'pooling':pooling,
                  'cache_id':self.path.name,'storage_dtype':self.info['dtype']}
        return data,EmbeddedUsages(word,data.usages,torch.stack(out),data.corpus_labels,metadata=metadata)


def list_corpus_caches(root: str | Path) -> list[dict]:
    root=Path(root)
    if not root.exists():
        return []
    result=[]
    for path in sorted(root.iterdir()):
        if path.name.startswith('.') or not (path/'index.sqlite').exists():
            continue
        try:
            with closing(sqlite3.connect((path/'index.sqlite').resolve().as_uri()+'?mode=ro',uri=True)) as con:
                info=json.loads(con.execute("SELECT value FROM metadata WHERE key='info'").fetchone()[0])
            if info.get('schema_version')==SCHEMA_VERSION:
                result.append({**info,'path':str(path)})
        except (sqlite3.Error, OSError, ValueError, TypeError):
            continue
    return result
