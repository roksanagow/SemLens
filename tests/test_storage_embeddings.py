import io
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from semlens.data_loading import corpus_sentences,load_from_sentences,select_usages
from semlens.embeddings import load_model,embed_usages,pool_span,iter_token_embeddings
from semlens.corpus_store import estimate_storage,create_corpus_cache,CorpusCache,list_corpus_caches
from semlens.definitions import definition_inputs,embed_definitions
from semlens.projects import project_bytes,load_project,save_project,rename_project,delete_project


@pytest.fixture(scope='module')
def encoder():
    # Small real encoder already cached on the development machine; no network.
    try:
        model=load_model('google/bert_uncased_L-2_H-128_A-2',device='cpu',local_files_only=True)
    except (OSError,ValueError):
        pytest.skip('Optional cached encoder unavailable; no network download is performed')
    return model


def test_no_silent_target_substitution_or_partial_coverage():
    vecs=torch.tensor([[1.,0.],[0.,1.]])
    offsets=np.array([[0,3],[4,7]])
    with pytest.raises(ValueError,match='no encoded tokens'):
        pool_span(vecs,offsets,(10,14),'the cat   bank')
    with pytest.raises(ValueError,match='partly covered'):
        pool_span(vecs,np.array([[0,3],[4,6]]),(4,7),'the cat')


def test_suffix_template_uses_explicit_word_placeholder():
    inputs,spans=definition_inputs('bank',['the bank of a river'],template='{definition}: {word}')
    assert inputs==['the bank of a river: bank']
    assert spans==[(21,25)]
    assert inputs[0][slice(*spans[0])]=='bank'
    with pytest.raises(ValueError,match='placeholder'):
        definition_inputs('bank',['finance'],template='{definition}')
    assert definition_inputs('bank',['finance'],template='{definition}',readout='sentence')[0]==['finance']


def test_long_context_layer_and_sentence_modes(encoder):
    sentence='The river was flowing. '*90+'The bank was steep.'
    data=load_from_sentences([sentence],'bank',['old'])
    word=embed_usages(encoder,data,max_length=64,layer=-1)
    intermediate=embed_usages(encoder,data,max_length=64,layer=1)
    sentence_emb=embed_usages(encoder,data,max_length=64,representation='sentence')
    assert word.embeddings.shape==(1,128)
    assert torch.isfinite(word.embeddings).all()
    assert not torch.allclose(word.embeddings,intermediate.embeddings)
    assert not torch.allclose(word.embeddings,sentence_emb.embeddings)
    assert intermediate.metadata['layer']==1
    assert sentence_emb.metadata['representation']=='sentence'


def test_negative_layers_count_back_from_the_final_output(encoder):
    data=load_from_sentences(['The bank was steep.','A bank holiday.'],'bank',['old','new'])
    # This encoder has two transformer layers: hidden states 0 (input), 1 and 2.
    by_layer={layer:embed_usages(encoder,data,layer=layer) for layer in (-3,-2,-1,0,1,2)}
    torch.testing.assert_close(by_layer[-2].embeddings,by_layer[1].embeddings)
    torch.testing.assert_close(by_layer[-3].embeddings,by_layer[0].embeddings)
    assert not torch.allclose(by_layer[-2].embeddings,by_layer[-1].embeddings)
    assert by_layer[-2].metadata['layer']==-2
    assert by_layer[-2].metadata['hidden_state_source']=='hidden_states[-2]'
    for layer in (-4,3):
        with pytest.raises(ValueError,match='layers 0 .* to 2; negative values count back'):
            embed_usages(encoder,data,layer=layer)


def test_cache_persists_text_and_supports_new_targets_without_encoder(encoder,tmp_path,monkeypatch):
    texts={'earlier':'The bank of the river was steep. They run along the river.',
           'later':'The bank approved a loan. A runner runs past the bank.'}
    records=corpus_sentences(texts)
    estimate=estimate_storage(records,encoder.tokenizer,128,raw_text_bytes=sum(len(t.encode()) for t in texts.values()))
    assert estimate.vector_bytes==estimate.tokens*128*4
    path=create_corpus_cache(tmp_path,'Collection',texts,encoder,estimate=estimate,max_length=64)
    cache=CorpusCache(path)
    data,cached=cache.select('bank')
    direct=embed_usages(encoder,data,max_length=64)
    torch.testing.assert_close(cached.embeddings,direct.embeddings)
    sentence_data,sentence_cached=cache.select('bank',representation='sentence')
    sentence_direct=embed_usages(encoder,sentence_data,max_length=64,representation='sentence')
    torch.testing.assert_close(sentence_cached.embeddings,sentence_direct.embeddings)
    def fail(*args,**kwargs):
        raise AssertionError('Reusing cache should not call the model')
    monkeypatch.setattr(encoder.model,'forward',fail)
    del cache
    reopened=CorpusCache(path)
    different,_=reopened.select('run',['runner','runs'])
    assert [u.matched_form for u in different.usages]==['run','runner','runs']
    assert reopened.source_texts()==texts
    assert len(list_corpus_caches(tmp_path))==1
    assert reopened.info['model']['model_id']==encoder.name
    assert reopened.info['model']['revision']
    process = subprocess.run([sys.executable, '-c',
        "from semlens.corpus_store import CorpusCache; import sys; "
        "data, embedded = CorpusCache(sys.argv[1]).select('bank'); "
        "print(len(data), embedded.dim)", str(path)], check=True, capture_output=True, text=True)
    assert process.stdout.strip() == '3 128'


def test_sentence_only_cache_rejects_word_extraction(encoder,tmp_path):
    texts={'single':'The bank is closed.'}
    path=create_corpus_cache(tmp_path,'Sentence cache',texts,encoder,storage='sentences')
    cache=CorpusCache(path)
    with pytest.raises(ValueError,match='sentence vectors only'):
        cache.select('bank')
    assert cache.select('bank',representation='sentence')[1].embeddings.shape==(1,128)


def test_failed_cache_is_not_listed(encoder,tmp_path,monkeypatch):
    import semlens.corpus_store as store
    def fail(*args,**kwargs):
        raise RuntimeError('interrupted computation')
    monkeypatch.setattr(store,'iter_token_embeddings',fail)
    with pytest.raises(RuntimeError,match='interrupted'):
        create_corpus_cache(tmp_path,'Interrupted',{'one':'The bank is open.'},encoder)
    assert list_corpus_caches(tmp_path)==[]
    assert list(tmp_path.iterdir())==[]


def test_saved_analysis_roundtrip_retains_occurrences_and_annotations(encoder):
    data=load_from_sentences(['bank then bank','The bank is open.'],'bank',['old','new'])
    embedded=embed_usages(encoder,data)
    definitions=['bank: financial institution','bank: river edge']
    defs=embed_definitions(encoder,'bank',definitions)
    content=project_bytes(data,embedded,definitions=definitions,definition_embeddings=defs,annotations={1:'river'},
                          sense_classes=['river','finance'],settings={'corpus_order':['old','new']},source_texts={'original':'bank then bank'})
    restored=load_project(content)
    assert restored['data'].char_spans==data.char_spans
    assert restored['annotations']=={1:'river'}
    assert restored['source_texts']=={'original':'bank then bank'}
    assert restored['embedded'].metadata==embedded.metadata
    torch.testing.assert_close(restored['embedded'].embeddings,embedded.embeddings)
    torch.testing.assert_close(restored['definition_embeddings'],defs)


def test_saved_file_rename_delete_and_collision_safety(tmp_path):
    from semlens.embeddings import embedded_usages_from_precomputed
    data=load_from_sentences(['The bank opened.','A river bank.'],'bank',['old','new'])
    embedded=embedded_usages_from_precomputed(data,torch.eye(2),metadata={'layer':3})
    content=project_bytes(data,embedded,annotations={1:'river'},sense_classes=['river'])
    original=tmp_path/'original.semlens'
    save_project(original,content)
    other=tmp_path/'existing.semlens'
    save_project(other,b'Leave this file unchanged')
    with pytest.raises(FileExistsError):
        rename_project(original,'existing')
    for name in ['../escape','folder/name','folder\\name','', '.hidden', '..']:
        with pytest.raises(ValueError):
            rename_project(original,name)
    assert original.read_bytes()==content
    renamed=rename_project(original,'Bank study – edited.semlens')
    assert renamed.name=='Bank study – edited.semlens'
    assert not original.exists() and renamed.read_bytes()==content
    assert load_project(renamed)['annotations']=={1:'river'}
    delete_project(renamed)
    assert not renamed.exists()
    assert other.read_bytes()==b'Leave this file unchanged'


def test_saved_file_actions_do_not_follow_symlinks(tmp_path):
    target=tmp_path/'target.semlens'
    target.write_bytes(b'preserved')
    link=tmp_path/'link.semlens'
    link.symlink_to(target)
    with pytest.raises(ValueError):
        rename_project(link,'renamed')
    with pytest.raises(ValueError):
        delete_project(link)
    assert target.read_bytes()==b'preserved'


def test_token_cache_on_supplied_text_excerpts(encoder,tmp_path):
    root=Path(__file__).resolve().parents[1]/'testing_texts'
    files=list(root.glob('*.txt'))
    if not files:
        pytest.skip('User-provided testing texts are unavailable')
    all_records=corpus_sentences({p.stem:p.read_text(encoding='utf-8-sig') for p in files})
    chosen=[]
    for corpus in dict.fromkeys(r.corpus for r in all_records):
        chosen.extend([r for r in all_records if r.corpus==corpus and 'love' in r.sentence.lower()][:3])
    texts={corpus:'\n'.join(r.sentence for r in chosen if r.corpus==corpus) for corpus in dict.fromkeys(r.corpus for r in chosen)}
    path=create_corpus_cache(tmp_path,'Literary excerpts',texts,encoder,lines=True,max_length=64)
    data,cached=CorpusCache(path).select('love',['loves','loved','loving'])
    assert len(data.corpus_labels)==2
    direct=embed_usages(encoder,data,max_length=64)
    torch.testing.assert_close(cached.embeddings,direct.embeddings,atol=2e-6,rtol=2e-6)


def test_target_markers_are_applied_to_word_embeddings_when_the_model_has_them():
    from semlens.embeddings import mark_target,supports_target_markers,extract_word_embeddings
    assert mark_target('The dot-com bubble burst.',(12,18))==('The dot-com <t>bubble</t> burst.',(15,21))
    try:
        model=load_model('google/bert_uncased_L-2_H-128_A-2',device='cpu',local_files_only=True)
    except (OSError,ValueError):
        pytest.skip('Optional cached encoder unavailable; no network download is performed')
    data=load_from_sentences(['The river bank was steep and the bank was shut.','A bank holiday.'],'bank',['old','new'])
    assert not supports_target_markers(model)
    plain=embed_usages(model,data)
    assert plain.metadata['target_markers'] is False
    with pytest.raises(ValueError,match='no <t> and </t>'):
        embed_usages(model,data,target_markers=True)
    # Give the small encoder the two marker tokens XL-LEXEME has.
    model.tokenizer.add_tokens(['<t>','</t>'],special_tokens=True)
    model.model.resize_token_embeddings(len(model.tokenizer))
    assert supports_target_markers(model)
    marked=embed_usages(model,data)
    assert marked.metadata['target_markers'] is True and marked.embeddings.shape==plain.embeddings.shape
    assert not torch.allclose(marked.embeddings,plain.embeddings,atol=1e-4)
    # The marked input is what the encoder sees, and the vector is the target's own tokens.
    text,span=mark_target(data.sentences[2],data.char_spans[2])
    direct=extract_word_embeddings(model,[text],[span],target_markers=False)
    torch.testing.assert_close(marked.embeddings[2:3],direct,atol=1e-5,rtol=1e-4)
    # Two targets in one sentence are marked separately.
    assert not torch.allclose(marked.embeddings[0],marked.embeddings[1],atol=1e-4)
    # Switching markers off, or asking for sentence vectors, encodes the plain sentence.
    torch.testing.assert_close(embed_usages(model,data,target_markers=False).embeddings,plain.embeddings,atol=1e-5,rtol=1e-4)
    sentence=embed_usages(model,data,representation='sentence')
    assert sentence.metadata['target_markers'] is False
    # Definitions mark the word the template adds, never a word inside the gloss.
    with_markers=embed_definitions(model,'bank',['bank: the bank of a river'],target_markers=True)
    text,span=mark_target('bank: the bank of a river',(0,4))
    assert text=='<t>bank</t>: the bank of a river'
    torch.testing.assert_close(with_markers,extract_word_embeddings(model,[text],[span],target_markers=False),atol=1e-5,rtol=1e-4)
    without=embed_definitions(model,'bank',['bank: the bank of a river'])
    assert not torch.allclose(with_markers,without,atol=1e-4)
