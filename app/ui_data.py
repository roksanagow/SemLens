"""Data input, reusable corpus preparation, and portable analysis snapshots."""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import re
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import torch
from semlens.data_loading import (corpus_sentences,select_usages,load_from_csv,load_precomputed_embeddings,
                                  validate_corpus_names,validate_embeddings)
from semlens.embeddings import embed_usages,embedded_usages_from_precomputed
from semlens.corpus_store import (CorpusCache,list_corpus_caches,estimate_storage,create_corpus_cache,format_bytes)
from semlens.projects import project_bytes,load_project,save_project,project_filename,rename_project,delete_project
from semlens.spaces import project_to_definition_space
from app.state import set_data,reset_downstream_of
from app.annotation_store import get_annotations,get_sense_classes,restore_annotations
from app.common import ROOT,STORAGE,get_model,show_usage
from app.presentation import readable_table


def _forms(value):
    return [s.strip() for s in re.split(r'[,\n]',value) if s.strip()]


def _cap(data,cap):
    if not cap:
        return data,list(range(len(data)))
    rng=np.random.default_rng(42)
    kept=[]
    for c in data.corpus_labels:
        ids=data.indices_for_corpus(c)
        kept.extend(sorted(rng.choice(ids,min(cap,len(ids)),replace=False).tolist()))
    kept=sorted(kept)
    return data.subset(kept),kept


def _set_loaded(data,embedded=None,source='model'):
    set_data(data,embedded,source=source)
    st.success(f'Loaded {len(data):,} occurrences from {len(data.corpus_labels)} corpora.')


def _load_snapshot(snapshot, *, analysis_path=None, analysis_name=None):
    set_data(snapshot['data'],snapshot['embedded'],source='saved',preserve_annotations=False)
    st.session_state.source_texts=snapshot['source_texts']
    st.session_state.definitions_raw=snapshot['definitions']
    st.session_state.definitions_formatted=snapshot['definitions']
    st.session_state.definition_embeddings=snapshot['definition_embeddings']
    if snapshot['definition_embeddings'] is not None:
        st.session_state.def_space_all=project_to_definition_space(snapshot['embedded'].embeddings,snapshot['definition_embeddings'])
    settings=snapshot['settings']
    for key in ('definition_settings','chronological','corpus_order','distance_metric','sampling_mode','sampling_repetitions','sampling_seed','source_lines','palette_name'):
        if key in settings:
            st.session_state[key]=settings[key]
    restore_annotations(snapshot['annotations'],snapshot['sense_classes'])
    if analysis_path is not None:
        st.session_state['_active_analysis_path']=str(analysis_path)
        st.session_state['_analysis_save_name']=analysis_path.stem
    elif analysis_name:
        st.session_state['_analysis_save_name']=analysis_name


def _cancel_file_deletion():
    st.session_state.pop('_delete_saved_file',None)


def _request_file_deletion(path):
    st.session_state['_delete_saved_file']=str(path)


def _rename_saved_file(path,name_key,choice_key,scope):
    try:
        renamed=rename_project(path,st.session_state[name_key])
        st.session_state[choice_key]=renamed
        if st.session_state.get('_active_analysis_path')==str(path):
            st.session_state['_active_analysis_path']=str(renamed)
            st.session_state['_analysis_save_name']=renamed.stem
        _cancel_file_deletion()
        st.session_state['_saved_file_notice']=(scope,'success',f'Renamed “{path.stem}” to “{renamed.stem}”.')
    except (OSError,ValueError) as exc:
        st.session_state['_saved_file_notice']=(scope,'error',str(exc))


def _delete_saved_file(path,choice_key,scope):
    try:
        delete_project(path)
        st.session_state.pop(choice_key,None)
        if st.session_state.get('_active_analysis_path')==str(path):
            st.session_state.pop('_active_analysis_path',None)
            st.session_state.pop('_prepared_analysis',None)
        _cancel_file_deletion()
        st.session_state['_saved_file_notice']=(scope,'success',f'Deleted “{path.stem}”. Any analysis already open remains available in this session.')
    except (OSError,ValueError) as exc:
        st.session_state['_saved_file_notice']=(scope,'error',str(exc))


def _saved_file_controls(word_only):
    scope='words' if word_only else 'projects'
    folder=STORAGE/scope
    paths=sorted((p for p in folder.glob('*.semlens') if p.is_file() and not p.is_symlink()),key=lambda p:p.name.casefold())
    notice=st.session_state.pop('_saved_file_notice',None)
    if notice and notice[0]==scope:
        getattr(st,notice[1])(notice[2])
    if not paths:
        st.info('No word embeddings saved locally yet.' if word_only else 'No analyses saved locally yet.')
        return
    choice_key=f'saved_file_{scope}'
    if st.session_state.get(choice_key) not in paths:
        st.session_state[choice_key]=paths[0]
    choice=st.selectbox('Saved locally',paths,format_func=lambda p:p.stem,key=choice_key,on_change=_cancel_file_deletion)
    if st.button('Open saved word embeddings' if word_only else 'Open saved analysis',type='primary'):
        try:
            _load_snapshot(load_project(choice),analysis_path=None if word_only else choice)
            st.success('Word embeddings and sentences restored.' if word_only else 'Analysis restored. Edit it in the workspaces, then save your changes below.')
        except Exception as exc:
            st.error(str(exc))
    with st.expander('Manage saved word embeddings' if word_only else 'Manage saved analysis',key=f'manage_saved_{scope}',on_change='rerun'):
        st.caption(f'Selected file: {choice.name}')
        name_key=f'rename_{scope}_{choice.name}'
        with st.form(f'rename_form_{scope}_{choice.name}'):
            st.text_input('New name',value=choice.stem,key=name_key)
            st.form_submit_button('Rename',on_click=_rename_saved_file,args=(choice,name_key,choice_key,scope))
        if st.session_state.get('_delete_saved_file')==str(choice):
            st.warning(f'Permanently delete “{choice.stem}”? This removes the saved file. The currently open analysis and separately saved corpus embeddings remain available.')
            cols=st.columns(2)
            cols[0].button('Confirm deletion',on_click=_delete_saved_file,args=(choice,choice_key,scope))
            cols[1].button('Cancel',on_click=_cancel_file_deletion)
        else:
            st.button('Delete saved file…',on_click=_request_file_deletion,args=(choice,))


def render_data():
    st.header('Data & model')
    st.caption('Choose a word, add corpora, then explore their usages together.')
    mode=st.selectbox('Data source',['Upload raw corpora (.txt)','Paste sentences','Upload CSV / TSV',
                                    'Saved corpora','Saved word embeddings','Pre-computed embeddings','Saved analysis'])
    if mode in {'Saved analysis','Saved word embeddings'}:
        word_only=mode=='Saved word embeddings'
        if word_only:
            st.caption('Reopen vectors for a saved word and its forms, with the matching sentences and model/layer settings. To analyse other words, use Saved corpora.')
        _saved_file_controls(word_only)
        upload=st.file_uploader('Or open an exported embedding file' if word_only else 'Or open an exported analysis',type=['semlens'])
        if upload is not None and st.button('Open uploaded file'):
            try:
                _load_snapshot(load_project(upload),analysis_name=None if word_only else Path(upload.name).stem)
                st.success('Analysis restored.')
            except Exception as exc:
                st.error(str(exc))
    else:
        word=st.text_input('Target word',value=st.session_state.target_word,key=f'target_{mode}').strip()
        forms=_forms(st.text_input('Other forms (optional)',help='Separate forms with commas: runs, running, runner, runners. Matching is case insensitive.',key=f'forms_{mode}'))
        with st.expander('Usage selection options'):
            cap=int(st.number_input('Maximum occurrences per corpus',min_value=0,value=0,step=100,key=f'cap_{mode}',
                                    help='0 uses all occurrences. A positive cap selects a reproducible random subset before embedding; plots show the retained usages.'))
        if mode in {'Paste sentences','Upload raw corpora (.txt)'}:
            texts={}
            lines=mode=='Paste sentences'
            if lines:
                count=int(st.number_input('Number of corpora',min_value=1,max_value=10,value=2,step=1))
                names=[]
                values=[]
                for i in range(count):
                    with st.expander(f'Corpus {i+1}',expanded=count<=2):
                        names.append(st.text_input('Corpus name',value=f'corpus_{i+1}',key=f'paste_name_{i}').strip())
                        values.append(st.text_area('Sentences (one per line)',key=f'paste_text_{i}',height=140))
                texts=dict(zip(names,values))
            else:
                uploads=st.file_uploader('Corpus text files (up to 10)',type=['txt'],accept_multiple_files=True)
                names=[]
                values=[]
                for i,f in enumerate(uploads):
                    names.append(st.text_input(f'Corpus name for {f.name}',value=Path(f.name).stem,key=f'raw_name_{i}').strip())
                    try:
                        values.append(f.getvalue().decode('utf-8-sig'))
                    except UnicodeDecodeError:
                        values.append('')
                        st.error(f'{f.name} must use UTF-8 encoding.')
                texts=dict(zip(names,values))
            if st.button('Load usages',type='primary'):
                try:
                    validate_corpus_names(names)
                    records=corpus_sentences(texts,lines=lines)
                    data,_=_cap(select_usages(records,word,forms),cap)
                    _set_loaded(data)
                    st.session_state.source_texts=texts
                    st.session_state.source_lines=lines
                    st.session_state.saved_cache_path=None
                    missing=[c for c in names if c not in data.corpus_labels]
                    if missing:
                        st.warning('No selected forms found in: '+', '.join(missing))
                except Exception as exc:
                    st.error(str(exc))
            # Store all source texts for precomputation even if no target is selected yet.
            corpus_input=(names,texts,lines)
        elif mode=='Upload CSV / TSV':
            upload=st.file_uploader('Usage table',type=['csv','tsv'])
            st.caption('Columns: sentence, corpus; optional start/end (zero-based, end exclusive). Explicit offsets select exactly that occurrence.')
            if st.button('Load usages',type='primary',disabled=upload is None):
                try:
                    data=load_from_csv(io.StringIO(upload.getvalue().decode('utf-8-sig')),word,
                                       delimiter='\t' if upload.name.endswith('.tsv') else ',',forms=forms)
                    data,_=_cap(data,cap)
                    _set_loaded(data)
                    st.session_state.source_texts={}
                    st.session_state.saved_cache_path=None
                except Exception as exc:
                    st.error(str(exc))
        elif mode=='Pre-computed embeddings':
            vectors=st.file_uploader('Embeddings',type=['pt','npy'])
            metadata=st.file_uploader('Matching sentence metadata',type=['csv','tsv'])
            st.caption('One vector per metadata row. Include start/end when a sentence contains multiple matching occurrences.')
            model_id=st.text_input('Original model ID (needed for definitions)',key='import_model')
            col1,col2=st.columns(2)
            layer=int(col1.number_input('Saved layer (-1 = final output)',min_value=-1,value=-1,key='import_layer'))
            representation=col2.selectbox('Saved representation',['word','sentence'],key='import_repr')
            revision=st.text_input('Model revision (optional)',key='import_revision')
            if st.button('Load embeddings',type='primary',disabled=vectors is None or metadata is None):
                try:
                    if vectors.name.endswith('.npy'):
                        embs=torch.from_numpy(np.load(io.BytesIO(vectors.getvalue()),allow_pickle=False))
                    else:
                        embs=torch.load(io.BytesIO(vectors.getvalue()),weights_only=True,map_location='cpu')
                    data=load_from_csv(io.StringIO(metadata.getvalue().decode('utf-8-sig')),word,
                                       delimiter='\t' if metadata.name.endswith('.tsv') else ',',forms=forms,all_occurrences=False)
                    embs=validate_embeddings(embs,len(data))
                    data,indices=_cap(data,cap)
                    config={'model_id':model_id.strip(),'revision':revision.strip() or None,'layer':layer,
                            'representation':representation,'pooling':'mean','max_length':256}
                    _set_loaded(data,embedded_usages_from_precomputed(data,embs[indices],metadata=config),'precomputed')
                    st.session_state.source_texts={}
                    st.session_state.saved_cache_path=None
                except Exception as exc:
                    st.error(str(exc))
        elif mode=='Saved corpora':
            caches=list_corpus_caches(STORAGE/'corpora')
            with st.expander('Open a corpus cache from another location'):
                external=st.text_input('Cache directory',help='The directory containing index.sqlite and vectors.npy.')
            selected=None
            if caches:
                selected=st.selectbox('Available corpora',caches,format_func=lambda c:f"{c['name']} · {c['sentences']:,} sentences · {c['model']['model_id']} · layer {c['model']['layer']}")
            elif not external:
                st.info('Prepare reusable embeddings from pasted or uploaded corpora first.')
            path=external.strip() or (selected['path'] if selected else None)
            representation=st.radio('Representation',['word','sentence'],horizontal=True,key='cache_repr',
                                    help='Word representations require a token cache. Sentence representations average the stored token vectors.')
            if st.button('Load word from saved corpora',type='primary',disabled=path is None):
                try:
                    cache=CorpusCache(path)
                    data,embedded=cache.select(word,forms,representation=representation,max_per_corpus=cap)
                    _set_loaded(data,embedded,'saved')
                    st.session_state.source_texts=cache.source_texts()
                    st.session_state.source_lines=cache.info['splitter']=='lines_v1'
                    st.session_state.saved_cache_path=str(path)
                except Exception as exc:
                    st.error(str(exc))

        if mode not in {'Saved corpora','Pre-computed embeddings'}:
            st.divider()
            _embedding_controls(corpus_input if mode in {'Paste sentences','Upload raw corpora (.txt)'} else None)

    data=st.session_state.target_word_data
    if data is not None:
        with st.expander('Review loaded occurrences'):
            readable_table(pd.DataFrame({'Corpus':data.corpus_labels,'Occurrences':[len(data.indices_for_corpus(c)) for c in data.corpus_labels]}))
            index=st.number_input('Occurrence to inspect',min_value=1,max_value=len(data),value=1,key=f'preview_{st.session_state.data_revision}')
            show_usage(data.usages[int(index)-1])
            st.caption('The highlighted text is the exact occurrence used for word embeddings.')
    if mode in {'Saved corpora','Saved word embeddings','Saved analysis','Pre-computed embeddings'} and st.session_state.embedded_usages is not None:
        with st.expander('Save word embeddings'):
            _save_word_embeddings()


def _save_word_embeddings():
    data=st.session_state.target_word_data
    embedded=st.session_state.embedded_usages
    if embedded is None:
        st.info('Embed the loaded usages first, choosing Target word or Whole sentence above.')
        return
    meta=embedded.metadata
    representation=meta.get('representation','unspecified')
    st.caption(f'Save exactly the {len(data):,} loaded occurrences of “{data.word}” and its selected forms. Includes their sentences and character positions. Selection caps apply; other corpus sentences are excluded.')
    st.write(f"Representation: **{representation}** · Model: **{meta.get('model_id') or 'unspecified'}** · Layer: **{meta.get('layer','unspecified')}**")
    st.caption('To save another representation or layer, embed the usages with those settings first. Reopen these files through Saved word embeddings.')
    name=st.text_input('Word embedding file name',value=f'{data.word}_{representation}')
    if st.button('Prepare word embedding file'):
        content=project_bytes(data,embedded,settings={'corpus_order':st.session_state.corpus_order,'scope':'word'})
        st.session_state['_prepared_word_embeddings']=(content,st.session_state.data_revision,id(embedded))
    prepared=st.session_state.get('_prepared_word_embeddings')
    if prepared and prepared[1:]==(st.session_state.data_revision,id(embedded)):
        content=prepared[0]
        st.info(f'File size before saving: {format_bytes(len(content))}. Includes the vectors, sentences and metadata.')
        safe=re.sub(r'[^\w.-]+','_',name).strip('.') or 'word_embeddings'
        st.caption('Saving with an existing file name replaces that saved word file.')
        if st.button('Save word embeddings locally',type='primary'):
            save_project(STORAGE/'words'/f'{safe}.semlens',content)
            st.success('Saved. Reopen through Data source → Saved word embeddings, including after restarting SemLens.')
        st.download_button('Download word embeddings',data=content,file_name=f'{safe}.semlens',mime='application/zip')


def _embedding_controls(corpus_input):
    st.subheader('Embeddings')
    representation=st.radio('Represent each usage by',['Target word','Whole sentence'],horizontal=True,help='Whole sentence uses the mean of the encoder’s text-token vectors; it includes surrounding topics and context.')
    rep='word' if representation=='Target word' else 'sentence'
    with st.expander('Advanced embedding settings'):
        model_id=st.text_input('Encoder model',value='pierluigic/xl-lexeme',key='encoder_model')
        st.caption('Hugging Face encoder model ID or a local model directory.')
        layer=int(st.number_input('Layer (-1 = final output)',min_value=-1,value=-1,key='encoder_layer',
                                  help='-1 preserves the current final output; 0 is the input embedding layer; 1 and higher select transformer layers.'))
        max_length=int(st.number_input('Tokens per context window',min_value=32,max_value=4096,value=256,step=32))
        batch_size=int(st.number_input('Inference batch size',min_value=1,max_value=128,value=16))
    if st.button('Embed loaded usages',type='primary',disabled=st.session_state.target_word_data is None):
        try:
            progress=st.progress(0.0)
            with st.spinner('Loading the encoder and embedding usages…'):
                model=get_model(model_id)
                embedded=embed_usages(model,st.session_state.target_word_data,representation=rep,layer=layer,max_length=max_length,
                                     batch_size=batch_size,progress_callback=lambda b,n:progress.progress(b/n))
            reset_downstream_of('embeddings')
            st.session_state.embedded_usages=embedded
            st.session_state.model_name=model.name
            st.session_state.embeddings_source='model'
            progress.empty()
            st.success(f'Created {embedded.dim}-dimensional {rep} embeddings. Explore all corpora in the other tabs.')
        except Exception as exc:
            st.error(str(exc))
    with st.expander('Prepare reusable corpus embeddings'):
        scope=st.radio('Embedding save scope',['Whole corpus','Current word only'],horizontal=True,
                       help='Whole corpus allows changing the target later. Current word saves only the loaded occurrences and the representation already computed.')
        if scope=='Current word only':
            _save_word_embeddings()
        elif corpus_input is None:
            st.info('Paste or upload complete text corpora to prepare a whole-corpus collection. Current word only can save the loaded usage embeddings.')
        else:
            names,texts,lines=corpus_input
            st.write('Save the complete corpora once, then reopen them and switch words in a later session. Source text and sentences are included.')
            name=st.text_input('Saved corpus collection name',value='My corpora')
            storage_choice=st.selectbox('What to save',['Word and sentence representations','Sentence representations only'])
            storage='tokens' if storage_choice.startswith('Word') else 'sentences'
            dtype=st.selectbox('Storage precision',['float32','float16'],help='float16 halves vector storage but rounds values. float32 preserves the computed precision.')
            signature=hashlib.sha256(json.dumps([texts,lines,model_id,layer,max_length,storage,dtype],sort_keys=True).encode()).hexdigest()
            if st.button('Estimate storage'):
                try:
                    validate_corpus_names(names)
                    records=corpus_sentences(texts,lines=lines)
                    from transformers import AutoConfig,AutoTokenizer
                    config=AutoConfig.from_pretrained(model_id)
                    tokenizer=AutoTokenizer.from_pretrained(model_id,use_fast=True)
                    dimension=getattr(config,'hidden_size',getattr(config,'d_model',0))
                    estimate=estimate_storage(records,tokenizer,dimension,dtype=dtype,storage=storage,raw_text_bytes=sum(len(t.encode()) for t in texts.values()))
                    st.session_state['_storage_estimate']=(signature,estimate)
                except Exception as exc:
                    st.error(str(exc))
            current=st.session_state.get('_storage_estimate')
            if current and current[0]==signature:
                estimate=current[1]
                st.info(f'Estimated disk space: {format_bytes(estimate.estimated_bytes)} for {estimate.sentences:,} sentences and {estimate.tokens:,} tokens. Includes source text and an allowance for the index; final size may differ.')
                if st.button('Compute & save corpus embeddings',type='primary'):
                    try:
                        validate_corpus_names(names)
                        progress=st.progress(0.0)
                        with st.spinner('Preparing reusable corpus embeddings…'):
                            model=get_model(model_id)
                            path=create_corpus_cache(STORAGE/'corpora',name,texts,model,storage=storage,dtype=dtype,layer=layer,
                                max_length=max_length,batch_size=batch_size,lines=lines,estimate=estimate,
                                progress_callback=lambda b,n:progress.progress(b/n))
                        progress.empty()
                        st.success(f'Saved {name}. Choose Saved corpora above to load any word, including after restarting SemLens.')
                        st.caption(str(path))
                    except Exception as exc:
                        st.error(str(exc))
            else:
                st.caption('Estimate storage with the current settings before computing and saving.')


def render_save_analysis():
    data=st.session_state.target_word_data
    embedded=st.session_state.embedded_usages
    if data is None or embedded is None:
        return
    with st.expander('Save or export this analysis',key='save_analysis_panel',on_change='rerun'):
        st.caption('Includes the current usages and vectors, definitions, annotations and available source text. Whole-corpus token caches stay in Saved corpora.')
        if '_analysis_save_name' not in st.session_state:
            st.session_state['_analysis_save_name']=data.word
        name=st.text_input('Analysis name',key='_analysis_save_name')
        settings={k:st.session_state[k] for k in ('definition_settings','chronological','corpus_order','distance_metric','sampling_mode','sampling_repetitions','sampling_seed','source_lines','palette_name')}
        if st.button('Prepare analysis file'):
            content=project_bytes(data,embedded,definitions=st.session_state.definitions_formatted,
                definition_embeddings=st.session_state.definition_embeddings,annotations=get_annotations(),
                sense_classes=get_sense_classes(),settings=settings,source_texts=st.session_state.source_texts)
            st.session_state['_prepared_analysis']=(content,st.session_state.data_revision,
                json.dumps([get_annotations(),get_sense_classes(),settings,st.session_state.definitions_formatted],sort_keys=True))
        prepared=st.session_state.get('_prepared_analysis')
        stamp=json.dumps([get_annotations(),get_sense_classes(),settings,st.session_state.definitions_formatted],sort_keys=True)
        if prepared and prepared[1:]==(st.session_state.data_revision,stamp):
            content=prepared[0]
            try:
                filename=project_filename(name)
            except ValueError as exc:
                st.error(str(exc))
                return
            path=STORAGE/'projects'/filename
            st.caption(f'File size: {format_bytes(len(content))}')
            replace=True
            if path.exists():
                if st.session_state.get('_active_analysis_path')==str(path):
                    st.caption(f'Saving updates “{path.stem}”. Change the name to save a separate copy.')
                else:
                    replace=st.checkbox(f'Replace existing saved analysis “{path.stem}”',key=f'replace_analysis_{filename}')
            if st.button('Save analysis locally',disabled=not replace):
                try:
                    save_project(path,content)
                    st.session_state['_active_analysis_path']=str(path)
                    st.success('Saved. Reopen through Data source → Saved analysis.')
                except OSError as exc:
                    st.error(str(exc))
            st.download_button('Download analysis',data=content,file_name=filename,mime='application/zip')
