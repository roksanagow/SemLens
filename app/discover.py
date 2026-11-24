"""Discover related words: words used like a set of seed words or seed usages, in each corpus.

A section of its own, in five steps. The corpora are read and every word is
counted (1); seeds define the concept (2); an optional quick first pass with
static vectors narrows the vocabulary (3); an even sample of every word's
occurrences is embedded and kept on disk (4); every sampled occurrence is then
compared with its nearest seeds, words are ranked in each corpus, and what the
user marks can join the seeds (5). Chosen words open in "Analyse a word".
"""
from __future__ import annotations
import hashlib
import io
import json
import shutil
import time
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from semlens.corpus_store import CorpusCache,list_corpus_caches
from semlens.data_loading import decode_text,validate_corpus_names
from semlens.resources import Cost,assess,count_tokens,format_bytes,system_resources
from semlens.utils import keyword_in_context
from semlens.words import MAX_WORDS,entry_of
from semlens.discover.vocabulary import cap_estimates,contexts_from_texts,count_occurrences,default_stop_words,sample_occurrences
from semlens.discover.contextual import (DEFAULT_MODEL,VectorStore,contexts_from_cache,default_layer,embed_seed_usage_dict,
                                         encode_from_cache,encode_sample)
from semlens.discover.knn import (ALL_CORPORA,StoreCandidates,background_rows,knn_cost,nearest_seeds,score_words,update_scoring)
from semlens.discover.seeds import (distinct_seed_words,parse_entries,seed_keys,seed_matrix,seed_rows,seed_usages_from_annotations,
                                    seed_usages_from_table,spelling_overlaps)
from semlens.discover.static import build_spaces,recommended_method,static_cost,static_scores
from semlens.discover.project import (DISCOVERY_FILE,Discovery,delete_discovery,folder_name,handoff_sentences,handoff_with_vectors,
                                      list_discoveries,load_seed_vectors,save_seed_vectors,source_texts)
from app.common import STORAGE,get_model
from app.costs import cost_check,device_kind,model_profile,remember_speed
from app.presentation import concordance_html,page_tabs,plot_chart,readable_table

DISCOVER_ROOT=STORAGE/'discover'
STEPS=['Corpora','Seeds','Quick first pass','Sample & embed','Results']
MODELS={
    'XLM-RoBERTa base · many languages, quick (recommended)':'FacebookAI/xlm-roberta-base',
    'XLM-RoBERTa large · slower, sometimes finer distinctions':'FacebookAI/xlm-roberta-large',
    'XL-LEXEME · trained to compare usages of one word':'pierluigic/xl-lexeme',
    'BERT base · English only, small':'google-bert/bert-base-uncased',
    'Another model…':None,
}
CAPS=[25,50,100,200,500,0]
# Settings kept while other steps are shown: Streamlit forgets the value of a
# widget that is not drawn, so these are re-saved at the start of every run and
# their widgets take their starting values from here rather than from value=.
DEFAULTS={
    'dv_restrict':'','dv_seed_words_input':'','dv_seed_table_forms':'',
    'dv_static_method':'word2vec','dv_static_dims':100,'dv_static_window':5,'dv_static_min':5,'dv_static_epochs':10,'dv_static_workers':1,'dv_static_k':10,
    'dv_static_top':50,'dv_static_keep':200,
    'dv_min_count':5,'dv_min_length':2,'dv_stop':True,'dv_extra_stop':'','dv_cap':'100','dv_model_other':DEFAULT_MODEL,
    'dv_max_length':256,'dv_batch':16,
    'dv_k':10,'dv_background':False,'dv_min_sampled':5,'dv_hide_seeds':True,'dv_hide_rejected':True,'dv_hide_spelling':False,'dv_top':50,
}
PERSIST=set(DEFAULTS)|{'dv_step','dv_source_choice','dv_usage_source','dv_static_method','dv_static_corpus','dv_vocabulary','dv_model_preset',
                       'dv_representation','dv_name','dv_vectors_from','dv_rank_by','dv_view','dv_inspect_order'}
PERSIST_PREFIXES=('dv_split_','dv_layer_')

GUIDE=(
    'Discover finds words used like a set of **seeds**: seed words you choose, or particular usages, such as sentences you annotated in SemLens. '
    'It ranks the words of each corpus by how close their usages are to the seeds.\n\n'
    '1. **Corpora**: one text per corpus. Optionally keep only the sentences (or paragraphs) that contain certain words or phrases.\n'
    '2. **Seeds**: the words or usages that define what you are looking for. Several different seed words work best.\n'
    '3. **Quick first pass** (optional): a fast shortlist of words, without a language model.\n'
    '4. **Sample & embed**: a language model embeds a random sample of each word’s usages in each corpus. You see the time and disk space it needs first.\n'
    '5. **Results**: the ranked words. Read their usages, mark the ones that fit, and add them to the seeds to refine the search. '
    'Open the words you choose in **Analyse a word**.\n\n'
    'Each usage is compared with its *k* nearest seeds rather than with their average, so a word can rank high when it resembles only some of the seeds.')
GLOSSARY=(
    '**Usage** · one occurrence of a word together with its context: the sentence, line or paragraph it appears in.\n\n'
    '**Seed** · a usage that defines what you are looking for: a usage of one of your seed words, or a sentence you supply.\n\n'
    '**Embedding** · the list of numbers a language model gives a usage. Similar usages get similar embeddings.\n\n'
    '**Distance** · how different two embeddings are (cosine distance): 0 for the same, larger for less alike.\n\n'
    '**kNN-AMD** · how close a word is to the seeds: for each of its usages, the average distance to the *k* nearest seeds, '
    'then the average over its usages. Lower is closer.\n\n'
    '**Near the concept** · the share of a word’s usages that are as close to the seeds as a typical seed is to the other seed words.\n\n'
    '**Background correction** · subtracts each usage’s distance to usages in general, so that words close to everything (*lot*, *kind*) do not rank high.\n\n'
    '**Cap** · the most usages of a word sampled from each corpus, picked at random.\n\n'
    '**Target word / masked target** · a usage represented by the embedding of the word itself, or of the model’s mask token put in its place, '
    'which reflects only the context.\n\n'
    '**Subword tokens** · the pieces a model splits a word into (*bubble*: bu + bble). Words that share pieces with a seed word can look alike '
    'because of their spelling.\n\n'
    '**Static embeddings** · one embedding per word and corpus (word2vec, or PPMI with SVD), used by the quick first pass.')
CAP_HELP=('The most usages of each word taken from each corpus. A more frequent word keeps a random sample of this size, spread over the whole corpus.\n\n'
          'A cap keeps time and disk space down: without one, the most frequent words take most of the work. '
          'But rarer senses of frequent words may be missed. Raise the cap if they matter, or keep only the sentences with the words you care about (step 1). '
          '“all” keeps every usage.')


# --- State -------------------------------------------------------------------------------------------

def _state(key,default=None):
    return st.session_state.get(key,default)


def _discovery():
    return st.session_state.get('dv_discovery')


def _store():
    """The active discovery's vectors, loaded once per session."""
    disc=_discovery()
    if disc is None:
        return None
    cached=st.session_state.get('_dv_store')
    if cached is None or cached.folder!=disc.folder:
        cached=VectorStore.load(disc.folder)
        st.session_state['_dv_store']=cached
        st.session_state['_dv_word_index']={w:i for i,w in enumerate(cached.sample.words)}
    return cached


def _activate(disc):
    st.session_state.dv_discovery=disc
    st.session_state.pop('_dv_store',None)
    st.session_state['_dv_seed_vectors']=load_seed_vectors(disc.folder)
    st.session_state['dv_seed_words_input']=', '.join(disc.seed_words)
    st.session_state['dv_seed_usages']=list(disc.seed_usages)
    st.session_state['dv_seed_corpora']=disc.seed_corpora
    for key in ('_dv_scoring','_dv_background','_dv_scores'):
        st.session_state.pop(key,None)


def _seed_words():
    try:
        return parse_entries(st.session_state.get('dv_seed_words_input',''))
    except ValueError:
        return []


def _seed_usages():
    return list(st.session_state.get('dv_seed_usages',[]))


def _save_seeds():
    """Keep the active discovery's seeds equal to the ones on the Seeds step."""
    disc=_discovery()
    if disc is None:
        return
    words,usages,corpora=_seed_words(),_seed_usages(),st.session_state.get('dv_seed_corpora')
    if (words,usages,corpora)!=(disc.seed_words,disc.seed_usages,disc.seed_corpora):
        disc.seed_words,disc.seed_usages,disc.seed_corpora=words,usages,corpora
        disc.save()


def _corpora():
    """Corpus names: the active discovery's, else the corpora read in step 1."""
    store=_store()
    if store is not None:
        return list(store.sample.corpora)
    return list(_state('dv_corpora',[]))


def _styled(frame):
    """A table with the theme's colours, as readable_table gives the rest of the app."""
    dark=st.session_state.get('theme_mode','Light')=='Dark'
    return frame.style.set_properties(**{'background-color':'#1b2230' if dark else '#ffffff','color':'#f1f5f9' if dark else '#172033'})


def _go(step):
    st.session_state['_dv_next_step']=step


def _tokenizer(model_id):
    return _load_tokenizer(model_id)


@st.cache_resource(show_spinner=False,max_entries=4)
def _load_tokenizer(model_id):
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(model_id,use_fast=True,local_files_only=True)
    except OSError:
        return AutoTokenizer.from_pretrained(model_id,use_fast=True)


# --- Page ----------------------------------------------------------------------------------------------

def render_discover_sidebar():
    disc=_discovery()
    if disc is not None:
        try:
            store=_store()
        except (OSError,ValueError) as exc:
            # The folder was moved or deleted outside the app.
            st.session_state.pop('dv_discovery',None)
            st.session_state.pop('_dv_store',None)
            _notice('warning',f'The discovery “{disc.name}” could not be opened ({exc}). Open another from Corpora → A saved discovery.')
            disc=None
    if disc is not None:
        st.write(f'**{disc.name}**')
        st.caption(f'{len(store.sample):,} occurrences · {len(store.sample.words):,} words · {len(store.sample.corpora)} corpora')
        st.caption(f'{store.metadata.get("model_id","")} · layer {store.metadata.get("layer")}')
    elif _state('dv_contexts'):
        st.caption(f'{len(st.session_state.dv_contexts):,} contexts read from {len(_corpora())} corpora')
    with st.expander('Glossary'):
        st.markdown(GLOSSARY)


def keep_settings():
    """Re-save Discover's settings at the start of every run, whichever section is shown.

    Streamlit forgets a widget's value after a run that does not draw it, so
    without this a visit to Analyse a word would clear the seed words and settings.
    """
    if st.session_state.pop('_dv_reset_name',False):
        st.session_state['dv_name']=_default_name()
    for key in list(st.session_state):
        if key in PERSIST or key.startswith(PERSIST_PREFIXES):
            st.session_state[key]=st.session_state[key]
    for key,value in DEFAULTS.items():
        st.session_state.setdefault(key,value)


def _default_name():
    """The first seed word and today's date, numbered when a discovery of that name is already saved."""
    seeds=[e for e in _seed_words() if not e.endswith('*')]
    base=(seeds[0] if seeds else 'discovery')+time.strftime(' %Y-%m-%d')
    name,number=base,2
    while (DISCOVER_ROOT/folder_name(name)).exists():
        name,number=f'{base} {number}',number+1
    return name


def _choice(key,options,default=None):
    """Keep a persisted choice valid when its options change."""
    if st.session_state.get(key) not in options:
        st.session_state[key]=options[0] if default is None else default


def render_discover():
    keep_settings()
    pending=st.session_state.pop('_dv_next_step',None)
    if pending:
        st.session_state.dv_step=pending
    # The steps on the first line, as the workspaces are in Analyse a word.
    step=page_tabs('Step',STEPS,'dv_step',format_func=lambda s:f'{STEPS.index(s)+1} · {s}')
    st.header('Discover related words')
    st.caption('Find the words used like a set of seed words or seed usages, in each of your corpora.')
    for kind,message in st.session_state.pop('_dv_notices',[]):
        getattr(st,kind)(message)
    {'Corpora':_render_corpora,'Seeds':_render_seeds,'Quick first pass':_render_quick_pass,
     'Sample & embed':_render_embed,'Results':_render_results}[step]()
    following=STEPS[(STEPS.index(step)+1)%len(STEPS)]
    with st.container(horizontal=True,horizontal_alignment='right',key='page_continue'):
        st.button(f'Continue to {following}' if step!=STEPS[-1] else 'Return to Corpora',icon=':material/arrow_forward:',
                  on_click=_go,args=(following,),key='dv_continue')


def _notice(kind,message):
    st.session_state.setdefault('_dv_notices',[]).append((kind,message))


# --- 1 · Corpora ---------------------------------------------------------------------------------------

def _render_corpora():
    with st.expander('How Discover works',expanded=not _state('dv_contexts') and _discovery() is None):
        st.markdown(GUIDE)
    sources=['Upload text files (.txt)','Upload a table (CSV / TSV)']
    if st.session_state.get('source_texts'):
        sources.append('Texts loaded in Analyse a word')
    sources+=['Saved corpora','A saved discovery']
    _choice('dv_source_choice',sources)
    source=st.selectbox('Corpora from',sources,key='dv_source_choice',
                        help='**Text files**: one plain-text file per corpus, up to ten.\n\n'
                             '**A table**: one row per text, with a column of text and, optionally, a column naming each row’s corpus.\n\n'
                             '**Texts loaded in Analyse a word**: the corpora already uploaded there in this session.\n\n'
                             '**Saved corpora**: a collection saved in Analyse a word. When it holds word embeddings, Discover can reuse them '
                             'instead of running a model again.\n\n'
                             '**A saved discovery**: continue an earlier discovery where you left off.')
    if source=='A saved discovery':
        _render_saved_discoveries()
        return
    texts,cache,description=None,None,''
    split_default='auto'
    if source=='Upload text files (.txt)':
        uploads=st.file_uploader('One text file per corpus (up to 10)',type=['txt'],accept_multiple_files=True,key='dv_uploads')
        names,values=[],[]
        for i,f in enumerate(uploads or []):
            names.append(st.text_input(f'Corpus name for {f.name}',value=Path(f.name).stem,key=f'dv_name_{i}_{f.name}').strip())
            values.append(decode_text(f.getvalue())[0])
        if uploads:
            texts=dict(zip(names,values))
            if len(texts)<len(names):
                st.error('Two files have the same corpus name. Give each corpus its own name.')
                return
            description='uploaded text files'
    elif source=='Upload a table (CSV / TSV)':
        upload=st.file_uploader('Table of texts',type=['csv','tsv'],key='dv_table')
        if upload is not None:
            try:
                frame=pd.read_csv(io.StringIO(decode_text(upload.getvalue())[0]),sep='\t' if upload.name.endswith('.tsv') else ',',
                                  dtype=str,keep_default_na=False)
            except Exception as exc:
                st.error(f'The table could not be read: {exc}')
                return
            columns=list(frame.columns)
            if not columns or frame.empty:
                st.error('The table has no rows.')
                return
            guess=next((c for c in columns if c.casefold() in {'text','body','content','sentence','document'}),columns[0])
            c1,c2=st.columns(2)
            text_col=c1.selectbox('Column of text',columns,index=columns.index(guess),key=f'dv_table_text_{upload.name}')
            options=['(all one corpus)']+[c for c in columns if c!=text_col]
            corpus_guess=next((c for c in options if c.casefold() in {'corpus','period','decade','year','genre','source'}),options[0])
            corpus_col=c2.selectbox('Column naming the corpus',options,index=options.index(corpus_guess),key=f'dv_table_corpus_{upload.name}',
                                    help='Rows with the same value form one corpus. Up to ten corpora.')
            groups=frame.groupby(corpus_col,sort=False)[text_col] if corpus_col!='(all one corpus)' else [(Path(upload.name).stem,frame[text_col])]
            texts={str(name):'\n\n'.join(t for t in values if t.strip()) for name,values in groups}
            description=f'table {upload.name}'
            split_default='paragraphs'
    elif source=='Texts loaded in Analyse a word':
        texts=dict(st.session_state.source_texts)
        st.caption(f'{len(texts)} corpora: {", ".join(texts)}.')
        description='texts loaded in Analyse a word'
    else:
        caches=list_corpus_caches(STORAGE/'corpora')
        if not caches:
            st.info('No saved corpora yet. Save a collection in Analyse a word (Data & Model → Prepare reusable corpus embeddings).')
            return
        chosen=st.selectbox('Saved collection',caches,key='dv_cache',
                            format_func=lambda c:f"{c['name']} · {c['sentences']:,} sentences · {c['model']['model_id']} · layer {c['model']['layer']}")
        cache=chosen['path']
        description=f'saved corpora {chosen["name"]}'
        if chosen['storage']=='tokens':
            st.caption('This collection holds word embeddings, so step 4 can reuse them instead of running a model again.')
        else:
            st.caption('This collection holds sentence embeddings only, so step 4 embeds its sentences again.')
    split=split_default
    if cache is None:
        divisions={'auto':'At sentence punctuation','lines':'One context per line','paragraphs':'One context per paragraph (blank lines between)'}
        st.session_state.setdefault(f'dv_split_{source}',split_default)
        split=st.selectbox('Divide the text into contexts',list(divisions),format_func=divisions.get,
                           key=f'dv_split_{source}',
                           help='Each occurrence is read inside its context. Sentences are the usual choice; paragraphs give the model more to go on but take longer.')
    restrict=st.text_input('Only contexts containing (optional)',key='dv_restrict',placeholder='For example: lockdown, social distanc*, echo chamber',
                           help='Keep only the contexts that contain at least one of these words or phrases; they need not be seed words. '
                                'This makes everything after it quicker and keeps the search to the part of the corpora you care about. '
                                'Commas separate alternatives; * matches any ending; capitals are ignored.').strip()
    if texts is None and cache is None:
        return
    if cache is None:
        size=sum(len(t.encode()) for t in texts.values())
    else:
        size=Path(cache,'index.sqlite').stat().st_size
    cost=Cost('Reading and counting the corpora',memory_bytes=size*8,seconds=size/6/1.5e6,rough_time=True,
              details=[f'{format_bytes(size)} of text'])
    allowed=cost_check(cost,'dv_read',remedy='Keep only the contexts containing given words, or use fewer or smaller corpora.')
    if st.button('Read the corpora',type='primary',disabled=not allowed):
        try:
            with st.spinner('Dividing the text into contexts and counting every word…'):
                if cache is None:
                    validate_corpus_names(list(texts))
                    contexts=contexts_from_texts(texts,split=split,restrict=restrict or None)
                    order=list(texts)
                else:
                    opened=CorpusCache(cache)
                    contexts=contexts_from_cache(opened,restrict or None)
                    order=list(opened.info.get('corpus_order') or dict.fromkeys(c.corpus for c in contexts))
                if not contexts:
                    raise ValueError('No context contains the words given under “Only contexts containing”.' if restrict else 'The corpora contain no text.')
                counts=count_occurrences(contexts)
            st.session_state['_dv_cache']={}
            st.session_state.dv_contexts=contexts
            st.session_state.dv_counts=counts
            st.session_state.dv_corpora=[c for c in order if any(x.corpus==c for x in contexts)]
            seed_corpora=[c for c in (st.session_state.get('dv_seed_corpora') or []) if c in st.session_state.dv_corpora]
            st.session_state.dv_seed_corpora=seed_corpora or None
            st.session_state.dv_source={'description':description,'cache':cache,'split':split,'restrict':restrict}
            for key in ('dv_sample','dv_static','dv_only_words'):
                st.session_state.pop(key,None)
            if _discovery() is not None:
                st.session_state.pop('dv_discovery',None)
                st.session_state.pop('_dv_store',None)
                _notice('info','The earlier discovery is saved: reopen it from Corpora → A saved discovery.')
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    if _state('dv_contexts'):
        _corpus_summary()


def _cached(key,compute):
    """A result computed once for the corpora that were read: the word counts can run to millions of entries."""
    store=st.session_state.setdefault('_dv_cache',{})
    if key not in store:
        store[key]=compute()
    return store[key]


def _corpus_counts():
    """Per corpus: the occurrences of each word, sorted, for quick totals and vocabulary sizes."""
    def compute():
        per={c:[] for c in st.session_state.dv_corpora}
        for (corpus,_),n in st.session_state.dv_counts.items():
            per.setdefault(corpus,[]).append(n)
        return {c:np.sort(np.asarray(v,dtype=np.int64)) for c,v in per.items()}
    return _cached('corpus_counts',compute)


def _vocabulary(corpus,min_count):
    counts=_corpus_counts()[corpus]
    return int(len(counts)-np.searchsorted(counts,min_count))


def _corpus_summary():
    contexts=st.session_state.dv_contexts
    info=st.session_state.get('dv_source',{})
    def compute():
        per_corpus=pd.Series([x.corpus for x in contexts]).value_counts()
        counts=_corpus_counts()
        return pd.DataFrame([{'Corpus':c,'Contexts':int(per_corpus.get(c,0)),'Running words':int(counts[c].sum()),'Different words':len(counts[c])}
                             for c in st.session_state.dv_corpora])
    st.write('**Read**'+(f' (only contexts containing: {info["restrict"]})' if info.get('restrict') else ''))
    readable_table(_cached('summary',compute))
    st.caption('Next: give the seeds.')


def _render_saved_discoveries():
    found=list_discoveries(DISCOVER_ROOT)
    if not found:
        st.info('No saved discoveries yet. Each discovery is saved as soon as its sample is embedded.')
        return
    chosen=st.selectbox('Saved discovery',found,key='dv_saved_choice',
                        format_func=lambda d:f"{d['name']} · {d['occurrences']:,} occurrences · {', '.join(d['corpora'])} · {d['model']}")
    c1,c2=st.columns([1,1])
    if c1.button('Open this discovery',type='primary'):
        try:
            _activate(Discovery.load(chosen['folder']))
            _go('Results')
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    if st.session_state.get('_dv_delete')==chosen['folder']:
        st.warning(f'Permanently delete “{chosen["name"]}”, with its embeddings and marks?')
        d1,d2=st.columns(2)
        if d1.button('Delete'):
            try:
                if _discovery() is not None and str(_discovery().folder)==chosen['folder']:
                    st.session_state.pop('dv_discovery',None)
                    st.session_state.pop('_dv_store',None)
                delete_discovery(chosen['folder'])
                st.session_state.pop('_dv_delete',None)
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
        if d2.button('Keep it'):
            st.session_state.pop('_dv_delete',None)
            st.rerun()
    elif c2.button('Delete…'):
        st.session_state['_dv_delete']=chosen['folder']
        st.rerun()
    size=sum(f.stat().st_size for f in Path(chosen['folder']).iterdir() if f.is_file())
    st.caption(f'Folder: {chosen["folder"]} · {format_bytes(size)}')


# --- 2 · Seeds -----------------------------------------------------------------------------------------

def _entry_counts(entries):
    """Occurrences of each seed entry in each corpus that was read, before sampling."""
    counts=st.session_state.get('dv_counts')
    if not counts or not entries:
        return None
    return _cached(('entries',tuple(entries)),lambda:_count_entries(counts,entries))


def _count_entries(counts,entries):
    corpora=st.session_state.dv_corpora
    table={e:dict.fromkeys(corpora,0) for e in entries}
    folded=[e.casefold() for e in entries]
    for (corpus,word),n in counts.items():
        entry=entry_of(word.casefold(),folded)
        if entry is not None and corpus in table[entries[folded.index(entry)]]:
            table[entries[folded.index(entry)]][corpus]+=n
    return pd.DataFrame([{'Seed word':e,**table[e],'Total':sum(table[e].values())} for e in entries])


def _render_seeds():
    st.caption('Seeds define what you are looking for: words, or particular usages of a word.')
    st.text_input('Seed words',key='dv_seed_words_input',on_change=_save_seeds,
                  placeholder='For example: bubble, echo chamber, silo, enclave, cocoon',
                  help='Every sampled usage of these words becomes a seed. Separate words with commas; end a word with * for any ending (cocoon*). Press Enter to count them. '
                      'Several different words work best: what their usages share defines the concept. '
                      'Phrases are not supported yet: give the main word.')
    try:
        entries=parse_entries(st.session_state.dv_seed_words_input)
    except ValueError as exc:
        st.error(str(exc))
        entries=[]
    corpora=_corpora()
    store=_store()
    if entries and store is not None:
        rows=[]
        for entry in entries:
            found=seed_rows(store.sample,[entry],valid=store.valid)
            counts=np.bincount(store.sample.corpus[found],minlength=len(corpora)) if len(found) else np.zeros(len(corpora),int)
            rows.append({'Seed word':entry,**{c:int(n) for c,n in zip(corpora,counts)},'Total':int(counts.sum())})
        st.write('Sampled occurrences of each seed word')
        readable_table(pd.DataFrame(rows))
        missing=[r['Seed word'] for r in rows if r['Total']==0]
        if missing:
            st.warning('Not in the sample: '+', '.join(missing)+'. Words added after the sample was made are found only when they were sampled: '
                       'prepare a new sample in step 4 to include them.')
    elif entries:
        with st.spinner('Counting the seed words in the corpora…'):
            table=_entry_counts(entries)
        if table is not None:
            st.write('Occurrences of each seed word in the corpora')
            readable_table(table)
            if (table['Total']==0).any():
                st.warning('Not found in the corpora: '+', '.join(table.loc[table['Total']==0,'Seed word'])+'.')
    if corpora:
        current=st.session_state.get('dv_seed_corpora')
        chosen=st.multiselect('Take seed occurrences from',corpora,default=[c for c in (current or corpora) if c in corpora],key='dv_seed_corpora_input',
                              help='All corpora by default. Choose one corpus to ask which words in the others are used like the seed words are used in it, '
                                   'for example the words that did the job of a modern term in older texts.')
        value=None if not chosen or len(chosen)==len(corpora) else chosen
        if not chosen:
            st.caption('No corpus chosen: seed occurrences are taken from every corpus.')
        if value!=current:
            st.session_state.dv_seed_corpora=value
            _save_seeds()
    _render_seed_usages()
    _seed_advice(entries,_seed_usages())


def _seed_advice(entries,usages):
    words={e.casefold() for e in entries}|{u.word.casefold() for u in usages}
    if not words:
        st.info('Give at least one seed word or seed usage.')
    elif len(words)<3:
        st.warning('The seeds come from '+('one word' if len(words)==1 else 'two words')+'. Words that share subword tokens with them '
                   '(*bust* and *bubble*: bu + st, bu + bble) can then rank high because of their spelling. Add seed words if you can; '
                   'Results marks such words, and its background correction helps.')


def _render_seed_usages():
    st.markdown('**Seed usages** (optional)')
    st.caption('Particular usages, such as those you annotated with one sense in SemLens. They are embedded with the same model as the sample.')
    kind=st.radio('Add seed usages from',['A table (CSV / TSV)','A saved SemLens analysis','The analysis open in Analyse a word'],
                  horizontal=True,key='dv_usage_source')
    found,label=None,''
    if kind=='A table (CSV / TSV)':
        upload=st.file_uploader('Table of seed sentences',type=['csv','tsv'],key='dv_seed_table',
                                help='A column **sentence**; optional **start** and **end** give the character position of the occurrence '
                                     '(from 0, end not included); an optional **corpus** column is kept.')
        forms=st.text_input('Seed word in these sentences',key='dv_seed_table_forms',placeholder='bubble, bubbles',
                            help='Needed when the table has no start and end columns: every occurrence of these forms becomes a seed.')
        if upload is not None and st.button('Add these usages'):
            try:
                found=seed_usages_from_table(decode_text(upload.getvalue())[0],forms=parse_entries(forms),
                                             delimiter='\t' if upload.name.endswith('.tsv') else ',',source=f'table {upload.name}')
            except ValueError as exc:
                st.error(str(exc))
    else:
        snapshot=None
        if kind=='A saved SemLens analysis':
            paths=sorted((STORAGE/'projects').glob('*.semlens'),key=lambda p:p.name.casefold())
            if not paths:
                st.info('No analyses saved in Analyse a word yet.')
            else:
                path=st.selectbox('Saved analysis',paths,format_func=lambda p:p.stem,key='dv_seed_project')
                try:
                    snapshot=_project(str(path),path.stat().st_mtime)
                    label=path.stem
                except Exception as exc:
                    st.error(str(exc))
        else:
            from app.annotation_store import get_annotations,get_options
            data=st.session_state.get('target_word_data')
            if data is None or not get_annotations():
                st.info('No annotated usages are open in Analyse a word.')
            else:
                snapshot={'data':data,'annotations':get_annotations(),'annotation_options':get_options()}
                label=f'annotations of {data.word}'
        if snapshot is not None:
            senses=sorted({s for labels in snapshot['annotations'].values() for s in (labels or {})})
            if not senses:
                st.info('This analysis has no annotations.')
            else:
                chosen=st.multiselect('Usages annotated with',senses,key=f'dv_seed_senses_{label}',
                                      help='Every usage given any of these sense classes becomes a seed.')
                options=snapshot.get('annotation_options') or {}
                minimum=None
                if options.get('graded'):
                    minimum=int(st.number_input('Lowest rating to include',min_value=1,max_value=int(options.get('scale',4)),key='dv_seed_rating',
                                                help='With graded annotation, keep only usages rated at least this high for one of the chosen senses.'))
                if chosen and st.button('Add these usages'):
                    found=seed_usages_from_annotations(snapshot['data'],snapshot['annotations'],chosen,min_rating=minimum,
                                                       source=f'{label}: {", ".join(chosen)}')
                    if not found:
                        st.warning('No usage has these senses.')
    if found:
        usages=_seed_usages()
        known={u.key for u in usages}
        added=[u for u in found if u.key not in known]
        st.session_state.dv_seed_usages=usages+added
        _save_seeds()
        _notice('success',f'Added {len(added):,} seed usages'+(f' ({len(found)-len(added):,} were already seeds)' if len(added)<len(found) else '')+'.')
        st.rerun()
    usages=_seed_usages()
    if usages:
        by_source={}
        for u in usages:
            by_source.setdefault(u.source or 'unnamed',[]).append(u)
        for source,items in by_source.items():
            c1,c2=st.columns([5,1])
            c1.write(f'{len(items):,} from {source}')
            if c2.button('Remove',key=f'dv_remove_{hashlib.sha256(source.encode()).hexdigest()[:8]}'):
                st.session_state.dv_seed_usages=[u for u in usages if (u.source or 'unnamed')!=source]
                _save_seeds()
                st.rerun()
        with st.expander(f'Read the {len(usages):,} seed usages'):
            st.html(concordance_html([(u.corpus or u.source,)+keyword_in_context(u.sentence,(u.start,u.end)) for u in usages[:300]]))


@st.cache_resource(show_spinner=False,max_entries=4)
def _project(path,mtime):
    from semlens.projects import load_project
    return load_project(path)


# --- 3 · Quick first pass ------------------------------------------------------------------------------

def _static_seed_words():
    """Seed words for static vectors: the seed words, and the words of the seed usages."""
    return list(dict.fromkeys(_seed_words()+[u.word.casefold() for u in _seed_usages()]))


def _render_quick_pass():
    st.caption('Static embeddings give each word one embedding per corpus, learned from the words around it. They take seconds to minutes and need '
               'no language model, so they can shortlist words before the slower step 4. A static embedding mixes all of a word’s senses, and on small '
               'corpora it finds words that appear in the same texts rather than words used alike: treat the ranking as a shortlist.')
    contexts=_state('dv_contexts')
    seeds=_static_seed_words()
    if not contexts:
        st.info('Read the corpora in step 1 first.')
        return
    if not seeds:
        st.info('Give seed words in step 2 first.')
        return
    corpora=st.session_state.dv_corpora
    tokens={c:int(_corpus_counts()[c].sum()) for c in corpora}
    recommended=recommended_method(tokens)
    methods={'word2vec':'word2vec','ppmi':'PPMI + SVD · steadier on small corpora'}
    method=st.radio('Method',list(methods),format_func=methods.get,horizontal=True,key='dv_static_method',
                    help='**PPMI + SVD**: counts which words occur near each other, and compresses the counts.\n\n'
                         '**word2vec**: learns embeddings by predicting nearby words; it needs more text.')
    if recommended=='ppmi':
        st.caption(f'The smallest corpus has {min(tokens.values()):,} running words. Below about a million, word2vec embeddings are noisy, '
                   'and PPMI + SVD is often steadier.')
    with st.expander('Settings'):
        c1,c2,c3=st.columns(3)
        dimensions=int(c1.number_input('Dimensions',min_value=10,max_value=500,step=10,key='dv_static_dims'))
        window=int(c2.number_input('Window (words each side)',min_value=1,max_value=20,key='dv_static_window'))
        min_count=int(c3.number_input('Minimum occurrences',min_value=1,key='dv_static_min',
                                      help='Words rarer than this in a corpus get no embedding there.'))
        epochs,workers=10,1
        if method=='word2vec':
            c4,c5=st.columns(2)
            epochs=int(c4.number_input('Passes over the text',min_value=1,max_value=50,key='dv_static_epochs'))
            workers=int(c5.number_input('Processor cores',min_value=1,max_value=16,key='dv_static_workers',
                                        help='More cores are faster, but results then vary slightly between runs. One core gives the same result every time.'))
        k=int(st.number_input('Nearest seeds (k)',min_value=1,max_value=50,key='dv_static_k',
                              help='Each word is scored by its average distance to this many of the nearest seed-word embeddings (one per seed word and corpus).'))
    costs=[static_cost(tokens[c],_vocabulary(c,min_count),method,dimensions=dimensions,window=window,epochs=epochs,workers=workers)
           for c in corpora]
    cost=Cost('Building static embeddings for every corpus',memory_bytes=max(c.memory_bytes for c in costs),seconds=sum(c.seconds for c in costs),
              rough_time=True,details=[f'{sum(tokens.values()):,} running words in {len(corpora)} corpora'])
    allowed=cost_check(cost,'dv_static',remedy='Keep only the contexts containing given words (step 1), or use word2vec, which needs less memory.')
    if st.button('Run the quick first pass',type='primary',disabled=not allowed):
        try:
            progress=st.progress(0.0)
            with st.spinner('Building embeddings for each corpus and aligning them…'):
                aligned=build_spaces(contexts,corpora,method,dimensions=dimensions,window=window,min_count=min_count,epochs=epochs,
                                     workers=workers,progress_callback=lambda i,n:progress.progress(i/n))
            progress.empty()
            st.session_state.dv_static={'scores':static_scores(aligned,seeds,k),'method':method,'seeds':seeds,'k':k,'spaces':aligned}
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    result=_state('dv_static')
    if result is None:
        return
    scores=result['scores']
    if st.session_state.get('dv_stop',True):
        stop=default_stop_words()
        scores=scores[~scores.word.isin(stop)]
    if result['seeds']!=seeds:
        st.caption('The seeds have changed since this pass was run: run it again to use them.')
    _choice('dv_static_corpus',[ALL_CORPORA]+corpora)
    view=st.segmented_control('Corpus',[ALL_CORPORA]+corpora,key='dv_static_corpus') or ALL_CORPORA
    top=int(st.number_input('Words shown',min_value=10,max_value=1000,step=10,key='dv_static_top'))
    shown=scores[scores.corpus==view].sort_values('knn_amd').head(top)
    table=pd.DataFrame({'#':range(1,len(shown)+1),'Word':shown.word,'Distance to the seeds':shown.knn_amd.round(3),
                        'Occurrences':shown.occurrences})
    if view==ALL_CORPORA:
        table['Corpora']=shown.corpora.values
    st.dataframe(_styled(table),hide_index=True,width='stretch',
                 column_config={'Distance to the seeds':st.column_config.NumberColumn(format='%.3f',
                                    help='Average distance to the k nearest seed-word embeddings. Lower is closer. In All corpora: the average over the corpora the word occurs in.')})
    st.download_button('Download all scores (CSV)',scores.to_csv(index=False),file_name='quick_first_pass.csv',mime='text/csv')
    with st.expander('Map of the words',key='dv_static_map',on_change='rerun') as panel:
        if panel.open:
            _static_word_map(result,scores,corpora,view,list(shown.word))
    st.divider()
    st.write('**Use the shortlist in step 4**')
    n=int(st.number_input('Words to keep from each corpus',min_value=10,max_value=5000,step=50,key='dv_static_keep',
                          help='The top words of each corpus, and of all corpora together, become the only words sampled and embedded in step 4, '
                               'with the seed words. This can cut the time of step 4 many times over.'))
    if st.button('Embed only these words in step 4'):
        chosen=set()
        for corpus in [ALL_CORPORA]+corpora:
            chosen|=set(scores[scores.corpus==corpus].sort_values('knn_amd').head(n).word)
        st.session_state.dv_only_words=sorted(chosen)
        st.session_state.pop('dv_sample',None)
        _notice('success',f'Step 4 will sample only these {len(chosen):,} words and the seed words.')
        _go('Sample & embed')
        st.rerun()


# --- 4 · Sample & embed --------------------------------------------------------------------------------

def _parse_cap(text):
    value=str(text).strip().casefold().replace(',','')
    if value in {'all','0',''}:
        return 0
    if not value.isdigit():
        raise ValueError('The cap must be a whole number, such as 100, or “all”.')
    return int(value)


def _model_choice():
    preset=st.selectbox('Language model',list(MODELS),key='dv_model_preset',
                        help='The model that reads each context. XLM-RoBERTa base is quick and reads about a hundred languages. '
                             'Choose Another model… for any encoder on Hugging Face (organisation/model) or in a folder on this computer.')
    model_id=MODELS[preset]
    if model_id is None:
        from app.ui_data import normalise_model_reference
        model_id=normalise_model_reference(st.text_input('Model',key='dv_model_other'))
    return model_id


def _render_embed():
    contexts=_state('dv_contexts')
    if not contexts:
        if _discovery() is not None:
            st.info(f'“{_discovery().name}” is already embedded: see Results. To make a new sample, read corpora in step 1.')
        else:
            st.info('Read the corpora in step 1 first.')
        return
    st.caption('A random sample of each word’s usages in each corpus is embedded by a language model. '
               'The embeddings are saved, so the discovery can be reopened later without running the model again.')
    c1,c2,c3=st.columns(3)
    min_count=int(c1.number_input('Minimum occurrences in a corpus',min_value=1,key='dv_min_count',
                                  help='Words rarer than this in a corpus are left out of it: a handful of usages says little about a word. Seed words are always kept.'))
    min_length=int(c2.number_input('Minimum word length (letters)',min_value=1,max_value=10,key='dv_min_length'))
    stop_on=c3.checkbox('Leave out English function words',key='dv_stop',
                        help='the, of, and, which, …: about 300 words. They rarely carry a concept, and they make up about half of all running words.')
    extra=st.text_input('Other words to leave out',key='dv_extra_stop',placeholder='said, mr, would',
                        help='Separate them with commas.')
    stop=(default_stop_words() if stop_on else set())|{w.strip().casefold() for w in extra.split(',') if w.strip()}
    seeds=[e for e in _seed_words() if not e.endswith('*')]
    keep=set(w.casefold() for w in seeds)|{u.word.casefold() for u in _seed_usages()}
    only=None
    shortlist=_state('dv_only_words')
    if shortlist:
        _choice('dv_vocabulary',['shortlist','all'])
        choice=st.radio('Words to sample',['shortlist','all'],key='dv_vocabulary',horizontal=True,
                        format_func={'shortlist':f'The {len(shortlist):,} words of the quick first pass, and the seeds','all':'Every word that passes the filters'}.get)
        only=shortlist if choice=='shortlist' else None
    cap_text=st.text_input('Occurrences kept per word in each corpus',key='dv_cap',help=CAP_HELP)
    try:
        cap=_parse_cap(cap_text)
    except ValueError as exc:
        st.error(str(exc))
        return
    source=st.session_state.get('dv_source',{})
    cache_info=None
    if source.get('cache'):
        try:
            cache_info=CorpusCache(source['cache']).info
        except Exception:
            cache_info=None
    reuse=False
    if cache_info is not None and cache_info.get('storage')=='tokens':
        model=cache_info['model']
        _choice('dv_vectors_from',['cache','model'])
        reuse=st.radio('Embeddings from',['cache','model'],key='dv_vectors_from',
                       format_func={'cache':f'The saved collection ({model.get("model_id")}, layer {model.get("layer")}): no model run, much quicker',
                                    'model':'A language model, chosen below'}.get)=='cache'
        if reuse and model.get('layer')==-1:
            st.caption('These embeddings are from the model’s last layer. Discover usually works better with a middle layer '
                       '(layer 8 of 12 for XLM-RoBERTa base), so embedding again can be worth the time.')
    if reuse:
        model_id=cache_info['model'].get('model_id','')
        hidden=int(cache_info['model'].get('dimension',768))
        layer,representation,max_length,batch_size=cache_info['model'].get('layer'),'word',int(cache_info['model'].get('max_length',256)),16
    else:
        model_id=_model_choice()
        profile=model_profile(model_id)
        hidden=profile['hidden'] if profile else 768
        layers=profile['layers'] if profile else 12
        c1,c2=st.columns(2)
        st.session_state.setdefault(f'dv_layer_{model_id}',min(layers,default_layer(layers)))
        layer=int(c1.number_input('Layer',min_value=0,max_value=layers,key=f'dv_layer_{model_id}',
                                  help=f'0 is the input embeddings and {layers} the last layer. Middle-to-upper layers carry word meaning best for this; '
                                       'the last is shaped by the model’s training task. The default is two thirds of the way up.'))
        reps={'word':'Target word · the embedding of the word itself (recommended)','slot':'Masked target · the word replaced by a mask, so only its context counts (slower)'}
        representation=c2.radio('Represent each usage by',list(reps),format_func=reps.get,key='dv_representation',
                                help='**Target word**: one model pass per sentence, however many words it holds. Best at finding words used alike, '
                                     'but words that share subword tokens with the seeds can look alike because of their spelling.\n\n'
                                     '**Masked target**: the word is replaced by the model’s mask token, so its spelling plays no part. It needs one model pass '
                                     'per usage, so it is several times slower, and it tends to find words that fit the same grammatical frame.')
        with st.expander('Advanced'):
            a1,a2=st.columns(2)
            max_length=int(a1.number_input('Tokens per context window',min_value=32,max_value=1024,step=32,key='dv_max_length'))
            batch_size=int(a2.number_input('Batch size',min_value=1,max_value=128,key='dv_batch',
                                           help='Contexts processed together. It affects speed and memory only, never the results.'))
    caps=list(dict.fromkeys(CAPS+[cap]))
    estimates=_cached(('caps',tuple(caps),min_count,min_length,tuple(sorted(stop)),tuple(sorted(keep)),None if only is None else tuple(only)),
                      lambda:cap_estimates(st.session_state.dv_counts,caps,min_count=min_count,min_length=min_length,stop_words=stop,
                                           keep_words=keep,only_words=None if only is None else set(only)|keep))
    table=pd.DataFrame({'Cap':['all' if c==0 else f'{c:,}' for c in estimates.cap],'Words':estimates.words,
                        'Occurrences to embed':estimates.occurrences,'Share of their occurrences':estimates.share,
                        'Words over the cap':estimates.capped,'Disk space':[format_bytes(n*hidden*2) for n in estimates.occurrences]})
    with st.expander('What different caps would mean',expanded=False):
        st.dataframe(_styled(table),hide_index=True,width='stretch',
                     column_config={'Share of their occurrences':st.column_config.NumberColumn(format='percent')})
    _cap_warning(cap,min_count,min_length,stop,keep)
    st.session_state.setdefault('dv_name',_default_name())
    name=st.text_input('Name of this discovery',key='dv_name')
    settings={'cap':cap,'min_count':min_count,'min_length':min_length,'stop':sorted(stop),'keep':sorted(keep),
              'only':None if only is None else sorted(only),'model':model_id,'layer':layer,'representation':representation,
              'max_length':max_length,'reuse':reuse,'contexts':len(contexts),'source':source}
    signature=hashlib.sha256(json.dumps(settings,sort_keys=True,default=str).encode()).hexdigest()
    prepared=_state('dv_sample')
    if prepared is None or prepared[0]!=signature:
        if st.button('Prepare the sample',type='primary'):
            with st.spinner('Sampling occurrences…'):
                sample=sample_occurrences(contexts,cap=cap,min_count=min_count,stop_words=stop,keep_words=keep,min_length=min_length,
                                          corpora=st.session_state.dv_corpora,only_words=None if only is None else set(only)|keep)
            if not len(sample):
                st.error('No word passes the filters. Lower the minimum number of occurrences.')
                return
            st.session_state.dv_sample=(signature,sample,sample.summary())
            st.rerun()
        st.caption('Preparing the sample is quick; it shows exactly what would be embedded, and the cost, before anything starts.')
        return
    sample=prepared[1]
    readable_table(prepared[2])
    try:
        folder=DISCOVER_ROOT/folder_name(name)
    except ValueError:
        st.error('Give the discovery a name with at least one letter or digit.')
        folder=None
    disk=len(sample)*hidden*2+sum(len(c.text.encode()) for c in sample.contexts)*1.2+len(sample)*40
    if reuse:
        cost=Cost('Reusing embeddings from the saved collection',disk_bytes=disk,memory_bytes=0,seconds=len(sample)/20000,rough_time=True,
                  details=[f'{len(sample):,} occurrences'])
        allowed=cost_check(cost,'dv_embed',remedy='Lower the cap, or raise the minimum number of occurrences.')
    else:
        from app.ui_data import _model_cost,_offer_speed_trial
        texts=([sample.contexts[c].text for c in sample.contexts_to_encode()] if representation=='word'
               else [sample.text_of(r) for r in range(len(sample))])
        cost=_model_cost('Embedding the sample',model_id,texts,0,layer,max_length,batch_size,disk_bytes=disk)
        if cost is None:
            st.caption('The cost could not be estimated: the model could not be looked up. Check its name, or the internet connection for a first download.')
            allowed=True
        else:
            cost.details.append(f'{len(sample):,} occurrences'+(', one pass of the model for each' if representation=='slot' else ''))
            allowed=cost_check(cost,'dv_embed',remedy='Lower the cap, raise the minimum number of occurrences, keep only contexts containing given words (step 1), '
                               'use the shortlist of a quick first pass (step 3), or choose Target word instead of Masked target.')
            _offer_speed_trial(cost,model_id,texts[:200],layer,max_length,batch_size,'discover')
    replace=True
    if folder is not None and folder.exists():
        replace=st.checkbox(f'Replace the saved discovery “{name.strip()}”',key=f'dv_replace_{folder.name}')
    if st.button('Embed and start exploring',type='primary',disabled=not (allowed and replace and folder is not None)):
        try:
            _embed(sample,folder,name.strip(),model_id,layer,representation,max_length,batch_size,reuse,source)
            _go('Results')
            st.rerun()
        except Exception as exc:
            st.error(str(exc))


def _cap_warning(cap,min_count,min_length,stop,keep):
    if not cap:
        st.caption('Every occurrence is kept: the commonest words will take most of the time and disk space.')
        return
    counts=_cached(('over',cap,min_count,min_length,tuple(sorted(stop)),tuple(sorted(keep))),
                   lambda:[n for (c,w),n in st.session_state.dv_counts.items()
                           if n>cap and (w.casefold() in keep or (n>=min_count and len(w)>=min_length and w.casefold() not in stop))])
    if not counts:
        st.caption(f'No word occurs more than {cap:,} times in a corpus, so the cap leaves nothing out.')
        return
    typical=int(np.median(counts))
    st.caption(f'{len(counts):,} words occur more than {cap:,} times in a corpus and are sampled. A word with {typical:,} occurrences (the middle of those) '
               f'keeps {cap/typical:.0%} of them: a sense used in 1 in 100 of its occurrences appears about {cap/100:.1f} times in the sample, '
               'and may be missed. Raise the cap if rarer senses of frequent words matter.')


def _embed(sample,folder,name,model_id,layer,representation,max_length,batch_size,reuse,source):
    DISCOVER_ROOT.mkdir(parents=True,exist_ok=True)
    temp=DISCOVER_ROOT/f'.{folder.name}.incomplete'
    shutil.rmtree(temp,ignore_errors=True)
    progress=st.progress(0.0)
    try:
        if reuse:
            with st.spinner('Reusing embeddings from the saved collection…'):
                store=encode_from_cache(CorpusCache(source['cache']),sample,temp,progress_callback=lambda i,n:progress.progress(i/n))
        else:
            with st.spinner('Loading the model and embedding the sample…'):
                model=get_model(model_id)
                started=time.perf_counter()
                store=encode_sample(model,sample,temp,layer=layer,max_length=max_length,batch_size=batch_size,representation=representation,
                                    progress_callback=lambda b,n:progress.progress(b/n))
                elapsed=time.perf_counter()-started
                if elapsed>2 and representation=='word':
                    texts=[sample.contexts[c].text for c in sample.contexts_to_encode()]
                    remember_speed(model.name,count_tokens(texts,model.tokenizer)/elapsed,batch_size,max_length)
        encoded=int(store.valid.sum())
        del store
        if folder.exists():
            shutil.rmtree(folder)
        temp.rename(folder)
    except Exception:
        shutil.rmtree(temp,ignore_errors=True)
        raise
    finally:
        progress.empty()
    seed_corpora=[c for c in (st.session_state.get('dv_seed_corpora') or []) if c in sample.corpora] or None
    st.session_state.dv_seed_corpora=seed_corpora
    disc=Discovery(folder,name,source.get('description',''),_seed_words(),seed_corpora,_seed_usages(),
                   settings={'restrict':source.get('restrict',''),'split':source.get('split','')})
    # The next discovery gets a fresh default name rather than this one. The name box
    # is on screen in this run, so its value is cleared at the start of the next.
    st.session_state['_dv_reset_name']=True
    disc.save()
    _activate(disc)
    _notice('success',f'Embedded {encoded:,} of {len(sample):,} occurrences and saved “{name}”.'
            +(f' {len(sample)-encoded:,} could not be encoded and are left out.' if encoded<len(sample) else ''))


# --- 5 · Results ---------------------------------------------------------------------------------------

def _seed_keys(disc,store):
    add=bool(disc.add_marked)
    return seed_keys(store.sample,words=_seed_words(),corpora=st.session_state.get('dv_seed_corpora'),usages=_seed_usages(),
                     marked_rows=disc.marked_rows if add else (),accepted_words=disc.accepted_words if add else (),valid=store.valid)


def _embed_missing_seed_usages(store):
    vectors=st.session_state.setdefault('_dv_seed_vectors',{})
    missing=[u for u in _seed_usages() if u.key not in vectors]
    if not missing:
        return []
    model=get_model(store.metadata.get('model_id'),revision=store.metadata.get('revision'))
    found=embed_seed_usage_dict(model,missing,store)
    vectors.update(found)
    save_seed_vectors(store.folder,{k:v for k,v in vectors.items() if k in {u.key for u in _seed_usages()}})
    return [u for u in missing if u.key not in found]


def _score(store,keys,k):
    previous=st.session_state.get('_dv_scoring')
    if previous is not None and previous[0]!=str(store.folder):
        previous=None
    vectors=seed_matrix(keys,store,st.session_state.get('_dv_seed_vectors',{}))
    usages={u.key:u for u in _seed_usages()}
    groups=[store.sample.word_of(v) if kind=='row' else usages[v].word for kind,v in keys]
    progress=st.progress(0.0)
    scoring=update_scoring(StoreCandidates(store),keys,vectors,k,previous[1] if previous else None,groups=groups,
                           progress_callback=lambda i,n:progress.progress(i/n))
    progress.empty()
    st.session_state['_dv_scoring']=(str(store.folder),scoring)
    st.session_state.pop('_dv_scores',None)


def _background(store,k):
    cached=st.session_state.get('_dv_background')
    if cached is not None and cached[:2]==(str(store.folder),k):
        return cached[2]
    rows=background_rows(store.sample,store.valid,3000)
    own=np.full(len(store.sample),-1,dtype=np.int64)
    own[rows]=np.arange(len(rows))
    with st.spinner('Comparing every occurrence with a random sample of usages…'):
        neighbours=nearest_seeds(StoreCandidates(store),store.normalised(rows),k,own_seed=own)
    st.session_state['_dv_background']=(str(store.folder),k,neighbours)
    return neighbours


def _render_results():
    disc=_discovery()
    if disc is None:
        st.info('Embed a sample in step 4, or open a saved discovery in step 1.')
        return
    store=_store()
    sample=store.sample
    meta=store.metadata
    st.caption(f'**{disc.name}** · {int(store.valid.sum()):,} occurrences of {len(sample.words):,} words in {len(sample.corpora)} corpora · '
               f'{meta.get("model_id","")}, layer {meta.get("layer")}, '+('masked target' if meta.get('discover_representation')=='slot' else 'target word')
               +(f' · from {disc.source}' if disc.source else ''))
    keys=_seed_keys(disc,store)
    usages={u.key:u for u in _seed_usages()}
    if not keys:
        st.info('Give seed words or seed usages in step 2. Seed words must occur in the sample.')
        return
    words=distinct_seed_words(sample,keys,usages)
    added=len(disc.marked_rows)+len(disc.accepted_words) if disc.add_marked else 0
    st.write(f'**{len(keys):,} seeds** from {len(words)} different word{"s" if len(words)!=1 else ""}'
             +(f', including what you marked' if added else ''))
    c1,c2,c3=st.columns(3)
    k=int(c1.number_input('Nearest seeds (k)',min_value=1,max_value=100,key='dv_k',
                          help='Each occurrence is scored by its mean distance to this many of its nearest seeds. A small k lets a word rank high '
                               'when it resembles a few seeds closely; a larger k asks it to resemble more of them.'))
    corrected=c2.checkbox('Background correction',key='dv_background',
                          help='Score each occurrence by its distance to the seeds minus its distance to usages in general, so that words close to everything '
                               '(lot, kind, way) lose their advantage. Then “near the concept” means closer to the seeds than to usages in general. '
                               'Recommended when the seeds come from one or two words.')
    rank_by=c3.radio('Rank words by',['knn','near'],key='dv_rank_by',horizontal=True,
                     format_func={'knn':'kNN-AMD','near':'Share near the concept'}.get,
                     help='**kNN-AMD**: how close the word’s usages are, on average, to their nearest seeds.\n\n'
                          '**Share near the concept**: how many of its usages are as close to the seeds as a typical seed is to the other seed words; '
                          'finds words with one fitting sense among others.')
    if len(words)<3 and not corrected:
        st.caption('The seeds come from fewer than three words: words spelled like them may rank high. Try the background correction, or add seed words in step 2.')
    scoring=st.session_state.get('_dv_scoring')
    current=scoring is not None and scoring[0]==str(store.folder) and scoring[1].keys==tuple(keys) and scoring[1].k==k
    if not current:
        prefix=(scoring is not None and scoring[0]==str(store.folder) and scoring[1].k==k
                and len(scoring[1].keys)<len(keys) and tuple(keys[:len(scoring[1].keys)])==scoring[1].keys)
        cost=knn_cost(len(sample),len(keys)-(len(scoring[1].keys) if prefix else 0),store.dim,k,device_kind())
        if scoring is None or scoring[0]!=str(store.folder):
            label='Find related words'
        elif prefix:
            label=f'Update the results with {len(keys)-len(scoring[1].keys):,} new seeds'
        elif scoring[1].keys==tuple(keys):
            label=f'Score again with k = {k}'
        else:
            label='Score again with the changed seeds'+(f' and k = {k}' if scoring[1].k!=k else '')
        allowed=cost_check(cost,'dv_knn',remedy='Use fewer seeds, or a smaller sample.')
        missing=[u for u in _seed_usages() if u.key not in st.session_state.get('_dv_seed_vectors',{})]
        if missing:
            st.caption(f'{len(missing):,} seed usages are read by {meta.get("model_id")} first.')
        if st.button(label,type='primary',disabled=not allowed):
            try:
                if missing:
                    with st.spinner('Reading the seed usages…'):
                        failed=_embed_missing_seed_usages(store)
                    if failed:
                        _notice('warning',f'{len(failed):,} seed usages could not be read and are left out.')
                        st.session_state.dv_seed_usages=[u for u in _seed_usages() if u.key in st.session_state['_dv_seed_vectors']]
                        _save_seeds()
                        keys=_seed_keys(disc,store)
                _score(store,keys,k)
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
        if scoring is None or scoring[0]!=str(store.folder):
            return
        st.caption('The results below are from the earlier seeds.')
    scoring=st.session_state['_dv_scoring'][1]
    background=_background(store,scoring.k) if corrected else None
    hidden=_hidden_words(sample,disc)
    stamp=(str(store.folder),scoring.keys,scoring.k,corrected)
    cached=st.session_state.get('_dv_scores')
    if cached is None or cached[0]!=stamp:
        with st.spinner('Ranking the words…'):
            per=score_words(sample,scoring.neighbours,scoring.threshold,valid=store.valid,background=background)
            pooled=score_words(sample,scoring.neighbours,scoring.threshold,valid=store.valid,background=background,pooled=True)
        cached=(stamp,per,pooled)
        st.session_state['_dv_scores']=cached
    per,pooled=cached[1],cached[2]
    _render_rankings(disc,store,per,pooled,hidden,rank_by,corrected,scoring)


def _hidden_words(sample,disc):
    entries=[e.casefold() for e in _seed_words()]
    seeds={w for w in sample.words if entry_of(w.casefold(),entries) is not None}
    if disc.add_marked:
        seeds|=set(disc.accepted_words)
    return seeds


def _rank(frame,rank_by):
    if rank_by=='near':
        return frame.sort_values(['share_near','knn_amd'],ascending=[False,True])
    return frame.sort_values('knn_amd')


def _render_rankings(disc,store,per,pooled,seeds,rank_by,corrected,scoring):
    sample=store.sample
    c1,c2,c3,c4=st.columns(4)
    min_sampled=int(c1.number_input('Rank words with at least',min_value=1,key='dv_min_sampled',
                                    help='Sampled usages a word needs in a corpus to be ranked there. Scores from a handful of usages are unreliable.'))
    hide_seeds=c2.checkbox('Hide the seed words',key='dv_hide_seeds')
    hide_rejected=c3.checkbox('Hide words you rejected',key='dv_hide_rejected')
    hide_spelling=c4.checkbox('Hide words spelled like a seed',key='dv_hide_spelling',
                              help='Leave out words that share a subword token of two or more letters with a seed word.')
    _choice('dv_view',[ALL_CORPORA]+list(sample.corpora))
    view=st.segmented_control('Corpus',[ALL_CORPORA]+list(sample.corpora),key='dv_view') or ALL_CORPORA
    top=int(st.number_input('Words shown',min_value=10,max_value=2000,step=10,key='dv_top'))
    def filtered(frame):
        frame=frame[frame.sampled>=min_sampled]
        if hide_seeds:
            frame=frame[~frame.word.isin(seeds)]
        if hide_rejected:
            frame=frame[~frame.word.isin(disc.rejected_words)]
        return frame
    ranked=_rank(filtered(pooled if view==ALL_CORPORA else per[per.corpus==view]),rank_by)
    tokenizer=_tokenizer_or_none(store.metadata.get('model_id'))
    # The seed words as given first, then the other words among the seeds, by how many seeds they give.
    from collections import Counter
    counted=Counter(sample.word_of(v) for kind,v in scoring.keys if kind=='row')
    seed_words=[e.rstrip('*') for e in _seed_words()]+[w for w,_ in counted.most_common()]+[u.word.casefold() for u in _seed_usages()]
    notes={}
    if tokenizer is not None:
        notes=spelling_overlaps(tokenizer,list(ranked.word[:top*3 if hide_spelling else top]),list(dict.fromkeys(seed_words))[:200])
    if hide_spelling:
        ranked=ranked[~ranked.word.isin(notes)]
    shown=ranked.head(top)
    status={'accepted':'✓ fits','rejected':'✗ does not fit'}
    table=pd.DataFrame({'#':range(1,len(shown)+1),'Word':shown.word.values,'kNN-AMD':shown.knn_amd.values,
                        'Near the concept':shown.share_near.values,'Sampled':shown.sampled.values,'In the corpus':shown.occurrences.values,
                        'Spelling':[notes.get(w,'') for w in shown.word],'Your decision':[status.get(disc.word_status.get(w),'') for w in shown.word]})
    knn_help=('For each sampled usage, its average distance to the k nearest seeds minus its average distance to the k nearest usages in general; '
              'then the average over the word’s usages. Below 0: closer to the seeds than to usages in general.' if corrected else
              'For each sampled usage, its average distance to the k nearest seeds; then the average over the word’s usages. Lower is closer.')
    near_help=('Share of the word’s sampled usages closer to the seeds than to usages in general.' if corrected else
               f'Share of the word’s sampled usages at least as close to the seeds as a typical seed is to the other seed words (distance {scoring.threshold:.3f}).'
               if scoring.threshold is not None else 'Needs seeds from at least two different words, or the background correction.')
    st.dataframe(_styled(table),hide_index=True,width='stretch',
                 column_config={'kNN-AMD':st.column_config.NumberColumn(format='%.3f',help=knn_help),
                                'Near the concept':st.column_config.NumberColumn(format='percent',help=near_help),
                                'Sampled':st.column_config.NumberColumn(help='Occurrences of the word in the sample, in this corpus (all corpora: in total).'),
                                'In the corpus':st.column_config.NumberColumn(help='All occurrences of the word, before the cap.'),
                                'Spelling':st.column_config.TextColumn(width='medium',help='Shares a subword token with a seed word, so it may look alike because of its spelling.')})
    if not corrected and scoring.threshold is None:
        st.caption('“Near the concept” needs seeds from at least two different words: it asks whether a word’s usages are as close to the seeds '
                   'as the seed words are to each other. With one seed word, use the background correction instead.')
    full=pd.concat([pooled,per],ignore_index=True)
    st.download_button('Download every score (CSV)',full.to_csv(index=False),file_name=f'{folder_slug(disc)}_scores.csv',mime='text/csv')
    if len(sample.corpora)>1:
        with st.expander('Across corpora: the top words in each corpus',expanded=True):
            _overview(_rank(filtered(pooled),rank_by).head(30).word.tolist(),per,sample.corpora,rank_by,min_sampled)
    with st.expander('Map of the words and their usages',key='dv_results_map',on_change='rerun') as panel:
        if panel.open:
            _results_map(disc,store,per,list(shown.word),scoring,corrected,view)
    st.divider()
    _render_inspect(disc,store,list(shown.word),scoring,corrected)
    st.divider()
    _render_handoff(disc,store,list(shown.word))


def folder_slug(disc):
    return Path(disc.folder).name


def _tokenizer_or_none(model_id):
    try:
        return _tokenizer(model_id) if model_id else None
    except Exception:
        return None


def _overview(words,per,corpora,rank_by,min_sampled):
    if not words:
        st.caption('No words to show.')
        return
    value='share_near' if rank_by=='near' else 'knn_amd'
    frame=per[per.word.isin(words)&(per.sampled>=min_sampled)].pivot(index='word',columns='corpus',values=value)
    frame=frame.reindex(index=words,columns=list(corpora))
    sampled=per[per.word.isin(words)].pivot(index='word',columns='corpus',values='sampled').reindex(index=words,columns=list(corpora))
    z=frame.values.astype(float)
    text=[['' if np.isnan(v) else (f'{v:.0%}' if value=='share_near' else f'{v:.3f}') for v in row] for row in z]
    hover=[[f'{w} · {c}<br>'+('not ranked: too few sampled usages' if np.isnan(z[i][j]) else f'{text[i][j]} · {int(sampled.values[i][j]):,} sampled')
            for j,c in enumerate(corpora)] for i,w in enumerate(words)]
    fig=go.Figure(go.Heatmap(z=z,x=list(corpora),y=words,text=text,texttemplate='%{text}',hovertext=hover,hoverinfo='text',
                             colorscale='Blues' if value=='share_near' else 'Blues_r',
                             colorbar={'title':'Near' if value=='share_near' else 'kNN-AMD'}))
    fig.update_layout(height=max(320,22*len(words)+120),yaxis={'autorange':'reversed'},margin={'l':10,'r':10,'t':30,'b':10},
                      title='The top words of all corpora, scored in each corpus')
    plot_chart(fig,width='stretch')
    st.caption(('Darker: more of the word’s usages near the concept.' if value=='share_near' else 'Darker: closer to the seeds.')
               +' Blank: the word has too few sampled usages in that corpus, or none.')


def _kwic(text,span,width=55):
    left,target,right=keyword_in_context(text,span,width)
    return left,target,right


def _seed_text(key,sample,usages):
    kind,value=key
    if kind=='row':
        left,target,right=_kwic(sample.text_of(value),sample.span_of(value),30)
    else:
        u=usages.get(value)
        if u is None:
            return ''
        left,target,right=_kwic(u.sentence,(u.start,u.end),30)
    return f'{left.strip()} [{target}] {right.strip()}'


def _is_near(distance,threshold,corrected):
    """As in the ranking: with the background correction, strictly closer to the seeds than to usages in general."""
    if corrected:
        return bool(distance<0)
    return bool(threshold is not None and distance<=threshold)


def _set_status(disc,word,key):
    choice=st.session_state.get(key) or 'undecided'
    if choice=='undecided':
        disc.word_status.pop(word,None)
    else:
        disc.word_status[word]=choice
    disc.save()


def _set_add_marked(disc):
    disc.add_marked=st.session_state.get('dv_add_marked')=='add'
    disc.save()


def _render_inspect(disc,store,shown,scoring,corrected):
    sample=store.sample
    st.subheader('Read a word’s usages')
    index=st.session_state['_dv_word_index']
    options=list(dict.fromkeys(shown+disc.accepted_words))
    if not options:
        return
    word=st.selectbox('Word',options,key='dv_inspect_word',help='A word from the ranking above, or one you accepted.')
    if word not in index:
        return
    key=f'dv_status_{word}'
    st.session_state[key]=disc.word_status.get(word,'undecided')
    st.segmented_control('This word',['undecided','accepted','rejected'],key=key,on_change=_set_status,args=(disc,word,key),
                         format_func={'undecided':'Undecided','accepted':'Fits the concept','rejected':'Does not fit'}.get)
    st.session_state['dv_add_marked']=None if disc.add_marked is None else ('add' if disc.add_marked else 'record')
    st.radio('What marking does',['add','record'],key='dv_add_marked',index=None,on_change=_set_add_marked,args=(disc,),horizontal=True,
             format_func={'add':'Add the usages I mark as fitting, and the words I accept, to the seeds','record':'Only record my marks'}.get,
             help='Choose before marking. Added seeds sharpen the search toward your judgements: press Update the results to see the effect, '
                  'which takes seconds because only the new seeds are compared. They can also pull the search away from the original seeds, '
                  'so read the new top words with care. Marks are saved with the discovery either way, and this choice can be changed later.')
    rows=np.flatnonzero((sample.word==index[word])&store.valid)
    distance=scoring.neighbours.knn_distance[rows]
    if corrected:
        distance=distance-_background(store,scoring.k).knn_distance[rows]
    c1,c2=st.columns(2)
    corpora=c1.multiselect('Corpora',list(sample.corpora),default=list(sample.corpora),key='dv_inspect_corpora')
    order=c2.radio('Order',['nearest','farthest','corpus'],horizontal=True,key='dv_inspect_order',
                   format_func={'nearest':'Nearest the seeds first','farthest':'Farthest first','corpus':'Corpus order'}.get)
    keep=np.isin(sample.corpus[rows],[list(sample.corpora).index(c) for c in corpora])
    rows,distance=rows[keep],distance[keep]
    sort=np.argsort(distance) if order=='nearest' else np.argsort(-distance) if order=='farthest' else np.arange(len(rows))
    rows,distance=rows[sort],distance[sort]
    threshold=0.0 if corrected else scoring.threshold
    usages={u.key:u for u in _seed_usages()}
    per_page=50
    pages=max(1,-(-len(rows)//per_page))
    page=int(st.number_input('Page',min_value=1,max_value=pages,value=1,key=f'dv_page_{word}_{order}')) if pages>1 else 1
    chunk=slice((page-1)*per_page,page*per_page)
    marks={True:'Yes',False:'No'}
    unmarked='–'
    frame=pd.DataFrame([{'Fits?':marks.get(disc.marks.get(int(r)),unmarked),'Corpus':sample.corpora[int(sample.corpus[r])],
                         **dict(zip(('Before','Word','After'),_kwic(sample.text_of(r),sample.span_of(r)))),
                         'Distance':float(d),'Near':_is_near(d,threshold,corrected),
                         'Nearest seed':_seed_text(scoring.keys[int(scoring.neighbours.seeds[r,0])],sample,usages)
                                        if scoring.neighbours.seeds[r,0]>=0 else ''}
                        for r,d in zip(rows[chunk],distance[chunk])])
    if frame.empty:
        st.caption('No usages in these corpora.')
        return
    disabled=disc.add_marked is None
    edited=st.data_editor(frame,hide_index=True,width='stretch',disabled=[c for c in frame.columns if c!='Fits?'] if not disabled else True,
                          key=f'dv_marks_{folder_slug(disc)}_{word}_{order}_{page}_{"-".join(corpora)}',
                          column_config={'Before':st.column_config.TextColumn(width='medium'),'Word':st.column_config.TextColumn(width='small'),
                                         'After':st.column_config.TextColumn(width='medium'),
                                         'Distance':st.column_config.NumberColumn(format='%.3f',help='Average distance to the k nearest seeds'+(', minus the distance to usages in general.' if corrected else '.')),
                                         'Near':st.column_config.CheckboxColumn(help='Near the concept, by the threshold in the ranking.'),
                                         'Nearest seed':st.column_config.TextColumn(width='large',help='The seed closest to this usage, its word in brackets.'),
                                         'Fits?':st.column_config.SelectboxColumn(options=[unmarked,'Yes','No'],width='small',pinned=True,required=True,
                                                                                  help='Does this usage fit the concept? Double-click a cell to choose Yes or No; – clears the mark.')})
    if disabled:
        st.caption('Choose what marking does, above, to start marking usages.')
        return
    changed=False
    for r,value in zip(rows[chunk],edited['Fits?']):
        r=int(r)
        new={'Yes':True,'No':False}.get(value)
        if disc.marks.get(r)!=new:
            if new is None:
                disc.marks.pop(r,None)
            else:
                disc.marks[r]=new
            changed=True
    if changed:
        disc.save()
        st.rerun()
    fits=sum(1 for r in rows if disc.marks.get(int(r)) is True)
    no=sum(1 for r in rows if disc.marks.get(int(r)) is False)
    if fits or no:
        st.caption(f'{word}: {fits} usages marked as fitting, {no} as not fitting'+(' · fitting usages join the seeds at the next update.' if disc.add_marked else '.'))
    with st.expander('Download your marks'):
        marked=[{'corpus':sample.corpora[int(sample.corpus[r])],'word':sample.word_of(r),'sentence':sample.text_of(r),
                 'start':sample.span_of(r)[0],'end':sample.span_of(r)[1],'fits':fits_} for r,fits_ in disc.marks.items()]
        decisions=[{'word':w,'decision':s} for w,s in disc.word_status.items()]
        st.download_button('Marked usages (CSV)',pd.DataFrame(marked).to_csv(index=False),file_name=f'{folder_slug(disc)}_marks.csv',mime='text/csv',
                           disabled=not marked,help='Usable as seed usages in another discovery, or loaded in Analyse a word as a usage table.')
        st.download_button('Word decisions (CSV)',pd.DataFrame(decisions).to_csv(index=False),file_name=f'{folder_slug(disc)}_words.csv',mime='text/csv',
                           disabled=not decisions)


# --- Maps --------------------------------------------------------------------------------------------

MAP_VIEWS=['PCA','UMAP','t-SNE']


def _map_view(key,count,dimensions,what='points'):
    """The projection and its settings, as in Explore. Returns (method, settings)."""
    from app.common import PROJECTION_HELP
    from app.ui_explore import _projection_settings
    if count<5:
        st.caption(f'With fewer than five {what}, the map uses PCA.')
        return 'PCA',{'components':(1,2)}
    method=st.radio('View',MAP_VIEWS,horizontal=True,key=f'{key}_view',help=PROJECTION_HELP)
    with st.expander('View settings'):
        settings=_projection_settings(method,key,count,dimensions,'cosine')
    return method,settings


def _project(array,method,settings):
    """2D positions and axis titles. Positions are only a picture: every score uses the full embeddings."""
    from app.common import cached_projection
    from app.ui_explore import _pca_pair
    if method=='PCA':
        coords,_,shares=_pca_pair(array,tuple(settings['components']))
        return coords,tuple(f'PC{c} · {share:.0%} of the variation' for c,share in zip(settings['components'],shares))
    coords=cached_projection(array,'tsne' if method=='t-SNE' else 'umap',**settings)
    return coords,(f'{method} 1',f'{method} 2')


def _word_map(points,vectors,key,title,corpora,connect_label):
    """Words as labelled points, one colour per corpus; seed words as stars.

    ``points`` has a row per point: word, corpus, distance (to the seeds) and seed (bool).
    """
    from app.common import corpus_colours
    if points.empty:
        st.caption('No words to map.')
        return
    method,settings=_map_view(key,len(points),vectors.shape[1],'words')
    connect=False
    if points.corpus.nunique()>1:
        connect=st.checkbox(connect_label,key=f'{key}_connect',
                            help='A thin line joins the points of each word, in corpus order: a long line means the word sits in a different place in another corpus.')
    coords,axes=_project(np.asarray(vectors,dtype=np.float32),method,settings)
    points=points.reset_index(drop=True).assign(x=coords[:,0],y=coords[:,1])
    colours=corpus_colours(list(corpora),st.session_state.palette_name)
    fig=go.Figure()
    if connect:
        order={c:i for i,c in enumerate(corpora)}
        for word,group in points.groupby('word',sort=False):
            if len(group)>1:
                group=group.sort_values('corpus',key=lambda c:c.map(order))
                fig.add_trace(go.Scatter(x=group.x,y=group.y,mode='lines',line=dict(color='#94a3b8',width=1),hoverinfo='skip',showlegend=False))
    size=float(st.session_state.get('point_size',9))
    for corpus in [c for c in corpora if c in set(points.corpus)]:
        for seed in (False,True):
            part=points[(points.corpus==corpus)&(points.seed==seed)]
            if part.empty:
                continue
            hover=[f'<b>{w}</b> · {corpus}'+('' if not np.isfinite(d) else f'<br>distance to the seeds {d:.3f}')+('<br>seed word' if seed else '')
                   for w,d in zip(part.word,part.distance)]
            fig.add_trace(go.Scatter(x=part.x,y=part.y,mode='markers+text',text=part.word,textposition='top center',textfont=dict(size=11),
                                     hovertext=hover,hoverinfo='text',name=corpus,legendgroup=corpus,
                                     # One legend entry per corpus: its seed words switch on and off with it.
                                     showlegend=not seed or points[(points.corpus==corpus)&~points.seed].empty,
                                     marker=dict(size=size+6 if seed else size,color=colours[corpus],symbol='star' if seed else 'circle',
                                                 line=dict(width=1,color='#111827') if seed else None)))
    if points.seed.any():
        fig.add_trace(go.Scatter(x=[None],y=[None],mode='markers',name='Seed words',hoverinfo='skip',
                                 marker=dict(size=size+6,color='#94a3b8',symbol='star',line=dict(width=1,color='#111827'))))
    # As in Explore's plots: the title at the very top, the legend below it, with room for its rows.
    names=[t.name for t in fig.data if t.showlegend is not False and t.name]
    widths=[48+6.2*len(n) for n in names]
    rows=1 if sum(widths)<=1100 else -(-len(names)//max(1,int(900//max(widths))))
    top=52+24*rows
    fig.update_layout(title=dict(text=title,font=dict(size=16),x=0,xref='paper',xanchor='left',y=1,yref='container',yanchor='top',pad=dict(t=12)),
                      xaxis_title=axes[0],yaxis_title=axes[1],height=600+top,hovermode='closest',
                      legend=dict(orientation='h',yanchor='bottom',y=1.01,xanchor='left',x=0),margin=dict(l=60,r=30,t=top,b=60))
    plot_chart(fig,width='stretch',key=f'{key}_chart')
    st.caption('Stars are seed words. Words close together have similar embeddings; the positions are a picture, and every score uses the full embeddings.')


def _static_word_map(result,scores,corpora,view,shown):
    """The quick first pass's words and seed words, in their aligned static embeddings."""
    spaces=result.get('spaces')
    if not spaces:
        st.caption('Run the quick first pass again to draw the map.')
        return
    n=int(st.number_input('Words on the map',min_value=5,max_value=300,value=min(40,max(5,len(shown))),step=5,key='dv_static_map_n',
                          help='The top words of the ranking above, with the seed words.'))
    words=list(dict.fromkeys(shown[:n]))
    entries=[e.casefold() for e in result['seeds']]
    distance={(c,w):d for c,w,d in zip(scores.corpus,scores.word,scores.knn_amd)}
    rows,vectors=[],[]
    for corpus in (corpora if view==ALL_CORPORA else [view]):
        space=spaces.get(corpus)
        if space is None:
            continue
        index=space.index()
        seeds=[w for w in space.words if entry_of(w.casefold(),entries) is not None][:60]
        for word in dict.fromkeys(words+seeds):
            if word in index:
                rows.append({'word':word,'corpus':corpus,'distance':distance.get((corpus,word),np.nan),'seed':word in seeds})
                vectors.append(space.vectors[index[word]])
    if not rows:
        st.caption('None of these words has an embedding in the chosen corpus.')
        return
    _word_map(pd.DataFrame(rows),np.stack(vectors),'dv_static_map',
              f'Words near the seeds · {view if view!=ALL_CORPORA else "every corpus"}',corpora,'Join each word across the corpora')


def _results_map(disc,store,per,shown,scoring,corrected,view):
    """Sampled usages (or words, as the average of their usages) of the chosen words, with the seeds."""
    sample=store.sample
    index=st.session_state['_dv_word_index']
    seed_words=[w for w in dict.fromkeys(sample.word_of(v) for kind,v in scoring.keys if kind=='row')]
    options=list(dict.fromkeys(shown+disc.accepted_words+seed_words))
    words=st.multiselect('Words',options,default=[w for w in shown[:8] if w in options],key='dv_map_words',
                         help='The words to draw, with the seeds. The top words of the ranking are chosen to start with.')
    c1,c2,c3=st.columns(3)
    points=c1.radio('Points',['usages','words'],horizontal=True,key='dv_map_points',
                    format_func={'usages':'Usages','words':'Words (average of their usages)'}.get)
    corpora=list(sample.corpora) if view==ALL_CORPORA else [view]
    if points=='usages':
        colour=c2.radio('Colour by',['word','corpus'],horizontal=True,key='dv_map_colour',format_func=str.capitalize)
        per_word=int(c3.number_input('Usages per word (at most)',min_value=10,max_value=2000,value=100,step=10,key='dv_map_per_word',
                                     help='A random sample of each word’s sampled usages, so the map stays readable.'))
    distance=scoring.neighbours.knn_distance
    if corrected:
        distance=distance-_background(store,scoring.k).knn_distance
    allowed=np.isin(sample.corpus,[list(sample.corpora).index(c) for c in corpora])&store.valid
    rng=np.random.default_rng(0)
    if points=='words':
        rows,vectors=[],[]
        for word in dict.fromkeys(words+seed_words[:30]):
            if word not in index:
                continue
            for corpus in corpora:
                found=np.flatnonzero((sample.word==index[word])&allowed&(sample.corpus==list(sample.corpora).index(corpus)))
                if len(found)<1:
                    continue
                mean=store.normalised(found).mean(axis=0)
                rows.append({'word':word,'corpus':corpus,'distance':float(np.mean(distance[found])),'seed':word in seed_words})
                vectors.append(mean/max(float(np.linalg.norm(mean)),1e-12))
        if not rows:
            st.caption('None of these words has sampled usages in the chosen corpus.')
            return
        _word_map(pd.DataFrame(rows),np.stack(vectors),'dv_results_words',
                  f'Words and seed words, as the average of their usages · {view if view!=ALL_CORPORA else "every corpus"}',
                  list(sample.corpora),'Join each word across the corpora')
        return
    # Usages: the chosen words' sampled usages, then the seeds, ringed.
    chosen=[]
    for word in words:
        if word in index:
            found=np.flatnonzero((sample.word==index[word])&allowed)
            chosen.extend(sorted(rng.choice(found,min(per_word,len(found)),replace=False).tolist()) if len(found) else [])
    seed_rows=[int(v) for kind,v in scoring.keys if kind=='row' and allowed[int(v)]]
    if len(seed_rows)>1500:
        seed_rows=sorted(rng.choice(seed_rows,1500,replace=False).tolist())
        st.caption('1,500 of the seeds are drawn, chosen at random.')
    rows=list(dict.fromkeys(chosen+seed_rows))
    usages=[u for u in _seed_usages() if u.key in st.session_state.get('_dv_seed_vectors',{})]
    if not rows and not usages:
        st.caption('Choose at least one word.')
        return
    seeds=set(seed_rows)
    vectors=[store.normalised(np.asarray(rows,dtype=np.int64))] if rows else []
    if usages:
        vectors.append(np.stack([st.session_state['_dv_seed_vectors'][u.key] for u in usages]))
    vectors=np.vstack(vectors)
    sentences=[sample.text_of(r) for r in rows]+[u.sentence for u in usages]
    spans=[sample.span_of(r) for r in rows]+[(u.start,u.end) for u in usages]
    corpus_of=[sample.corpora[int(sample.corpus[r])] for r in rows]+[u.corpus or 'seed usages' for u in usages]
    word_of=[sample.word_of(r) for r in rows]+['seed usages']*len(usages)
    dist=[float(distance[r]) for r in rows]+[np.nan]*len(usages)
    groups=word_of if colour=='word' else corpus_of
    other=corpus_of if colour=='word' else word_of
    notes=[f'{"Corpus" if colour=="word" else "Word"}: {o}'+('' if not np.isfinite(d) else f' · distance to the seeds {d:.3f}')
           +(' · seed' if i<len(rows) and rows[i] in seeds or i>=len(rows) else '') for i,(o,d) in enumerate(zip(other,dist))]
    method,settings=_map_view('dv_usage_map',len(vectors),vectors.shape[1],'usages')
    coords,axes=_project(vectors.astype(np.float32),method,settings)
    order=list(dict.fromkeys(words+seed_words+['seed usages'])) if colour=='word' else list(sample.corpora)+['seed usages']
    from app.components.scatter import render_scatter
    render_scatter(coords,groups,sentences,'',char_spans=spans,corpus_order=[g for g in order if g in set(groups)],
                   palette_name=st.session_state.palette_name,title='Usages of the chosen words, with the seeds ringed',axis_labels=axes,
                   highlight_indices=[i for i,r in enumerate(rows) if r in seeds]+list(range(len(rows),len(rows)+len(usages))),
                   highlight_label='Seeds',hover_notes=notes,key='dv_usage_map')
    st.caption(f'{len(vectors):,} usages. Hover a point to read its sentence; click a name in the legend to hide or show it. '
               'The positions are a picture: every score uses the full embeddings.')


# --- Hand-off to Analyse a word ------------------------------------------------------------------------

def _render_handoff(disc,store,shown):
    st.subheader('Analyse these words in SemLens')
    st.caption('Open the chosen words as one analysis in Analyse a word, with every SemLens view: projections, scores between corpora and between words, '
               'definitions and annotation.')
    sample=store.sample
    present=set(sample.words)
    plain=[e.casefold() for e in _seed_words() if e.casefold() in present]
    options=list(dict.fromkeys(plain+disc.accepted_words+shown[:30]))
    default=list(dict.fromkeys(plain+disc.accepted_words))[:MAX_WORDS]
    words=st.multiselect('Words',options,default=default,key='dv_handoff_words',max_selections=MAX_WORDS,
                         help=f'Up to {MAX_WORDS}. They are compared as different words, each with its own colour.')
    corpora=st.multiselect('Corpora',list(sample.corpora),default=list(sample.corpora),key='dv_handoff_corpora')
    slot=store.metadata.get('discover_representation')=='slot'
    modes={'vectors':f'The sampled usages, with their embeddings ({store.metadata.get("model_id")}, layer {store.metadata.get("layer")}): opens at once',
           'sentences':'Every usage in the corpora, to embed in Data & Model (with XL-LEXEME, say)'}
    mode=st.radio('Usages',list(modes) if not slot else ['sentences'],format_func=modes.get,key='dv_handoff_mode',
                  help='**The sampled usages**: no waiting, but only up to each word’s cap, with Discover’s model and layer.\n\n'
                       '**Every usage**: all of them, embedded with the model you choose in Data & Model.'
                       +('\n\nMasked-target embeddings describe the context, not the word, so they are not used for the analysis.' if slot else ''))
    st.button('Open in Analyse a word',type='primary',disabled=not words or not corpora,on_click=_hand_off,args=(store,words,corpora,mode))
    error=st.session_state.pop('_dv_handoff_error',None)
    if error:
        st.error(error)


def _hand_off(store,words,corpora,mode):
    """Load the chosen words into Analyse a word. Runs as a button callback, before the next run draws anything."""
    from app.state import set_data
    words=sorted(words,key=lambda w:w.endswith('*'))
    try:
        if mode=='vectors':
            data,embedded=handoff_with_vectors(store,words,corpora)
            set_data(data,embedded,source='discover')
            workspace='Explore'
        else:
            data=handoff_sentences(store.sample,words,corpora)
            set_data(data,None)
            workspace='Data & Model'
            st.session_state.setdefault('_load_notices',[]).append(
                ('success',f'Loaded {len(data):,} usages of {", ".join(data.forms)} from Discover. Embed them below, choosing the model under Advanced embedding settings.'))
        # Data & Model offers the same corpora, to load other words from them.
        raw='Upload raw corpora (.txt)'
        st.session_state['_data_source']=raw
        st.session_state['_source_kind']=raw
        st.session_state['raw_split']='paragraphs'
        st.session_state[f'forms_{raw}']=', '.join(data.forms[1:])
        texts=source_texts(store.sample)
        st.session_state.source_texts={c:t for c,t in texts.items() if c in corpora}
        st.session_state.source_lines=False
        st.session_state.source_split='paragraphs'
        st.session_state.saved_cache_path=None
        st.session_state.pop('_corpus_stats',None)
        st.session_state.words_mode='words' if len(data.forms)>1 else 'forms'
        st.session_state.word_mapping={}
        st.session_state.group_by='corpus'
        st.session_state.words_shown=None
        st.session_state.corpora_shown=None
        st.session_state.workspace=workspace
        st.session_state.section='Analyse a word'
    except Exception as exc:
        st.session_state['_dv_handoff_error']=str(exc)
