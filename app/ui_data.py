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
                                  validate_corpus_names,validate_embeddings,decode_text,count_words)
from semlens.utils import keyword_in_context,usage_search_pattern
from semlens.embeddings import embed_usages,embedded_usages_from_precomputed
from semlens.corpus_store import (CorpusCache,list_corpus_caches,estimate_storage,create_corpus_cache,format_bytes)
from semlens.projects import project_bytes,load_project,save_project,project_filename,rename_project,delete_project
from semlens.spaces import project_to_definition_space
from app.state import set_data,reset_downstream_of
from app.annotation_store import get_annotations,get_sense_classes,restore_annotations
from app.common import ROOT,STORAGE,get_model,show_usage,ordered_labels
from app.presentation import readable_table,concordance_html


# Shipped inside the package so that an installed copy has it too; a test keeps it equal to testing_texts/.
EXAMPLE_PATH=Path(__file__).resolve().parent/'data'/'bubble_usages.csv'
EXAMPLE_SOURCE='Example: “bubble” in the Guardian, 1999–2026'
EXAMPLE_DESCRIPTION=('350 sentences containing *bubble* or *bubbles*, sampled from Guardian and Observer articles: 50 from each of seven periods '
                     'between 1999 and 2026, one per article. The sample was not chosen for particular senses, so it mixes soap bubbles, '
                     'financial bubbles, social and political bubbles, and the support bubbles of the Covid years.\n\n'
                     'Load it, then click **Embed loaded usages** below. Publication date, title and link come with each sentence and are returned in the annotation export.')
MODEL_PRESETS={
    'XL-LEXEME · trained to compare word meanings in context (recommended)':'pierluigic/xl-lexeme',
    'XLM-RoBERTa large · general purpose':'FacebookAI/xlm-roberta-large',
    'XLM-RoBERTa base · general purpose, smaller and faster':'FacebookAI/xlm-roberta-base',
    'BERT base · English only, small':'google-bert/bert-base-uncased',
    'Another model…':None,
}
GUIDE=(
    'SemLens shows how one word is used across collections of text, such as different periods, and helps you judge whether its meaning differs between them.\n\n'
    '1. **Data & Model** (this page). Give the word and one text per corpus. SemLens finds every sentence containing the word, '
    'and a language model turns each of those usages into numbers that capture how the word is used there.\n'
    '2. **Explore**. Every usage becomes a point; usages the model finds similar sit close together. Scores summarise how far apart two corpora are, '
    'and you can read the sentences behind any score.\n'
    '3. **Definitions**. Write the senses you expect as short definitions and see which usages lie nearest each one.\n'
    '4. **Annotation**. Label usages with your own sense classes, then read off how often each sense occurs in each corpus.\n\n'
    'The model’s similarity is evidence to inspect, not a verdict: the tool is built so that every number leads back to sentences you can read. '
    'To try it without your own data, choose **Example** under Data source. The **Glossary** in the sidebar explains the terms, and each **(?)** explains its control.')


def _forms(value):
    return [s.strip() for s in re.split(r'[,\n]',value) if s.strip()]


def _parse_cap(text):
    """'all' (or blank, or 0) keeps every occurrence; otherwise a positive whole number."""
    value=str(text).strip().casefold().replace(',','').replace('_','')
    if value in {'','all','0'}:
        return 0
    if not value.isdigit():
        raise ValueError('Maximum occurrences per corpus must be “all” or a positive whole number, such as 500.')
    return int(value)


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
    _notify('success',f'Loaded {len(data):,} occurrences from {len(data.corpus_labels)} corpora.')


def _notify(kind,message):
    st.session_state.setdefault('_load_notices',[]).append((kind,message))


def _finish_load():
    """Redraw from the top, so the sidebar summary and corpus order show the new data at once."""
    st.rerun()


def _show_load_notices():
    for kind,message in st.session_state.pop('_load_notices',[]):
        getattr(st,kind)(message)


def _load_snapshot(snapshot, *, analysis_path=None, analysis_name=None):
    set_data(snapshot['data'],snapshot['embedded'],source='saved',preserve_annotations=False)
    st.session_state.source_texts=snapshot['source_texts']
    st.session_state.definitions_raw=snapshot['definitions']
    st.session_state.definitions_formatted=snapshot['definitions']
    st.session_state.definition_embeddings=snapshot['definition_embeddings']
    if snapshot['definition_embeddings'] is not None:
        st.session_state.def_space_all=project_to_definition_space(snapshot['embedded'].embeddings,snapshot['definition_embeddings'])
    settings=snapshot['settings']
    for key in ('definition_settings','chronological','corpus_order','distance_metric','sampling_mode','sampling_repetitions','sampling_seed','source_lines','source_split','palette_name'):
        if key in settings:
            st.session_state[key]=settings[key]
    restore_annotations(snapshot['annotations'],snapshot['sense_classes'])
    st.session_state.pop('_corpus_stats',None)
    # Which input these source texts came from, so they are offered again only there.
    st.session_state['_source_kind']=(('Paste sentences' if st.session_state.source_lines else 'Upload raw corpora (.txt)')
                                      if snapshot['source_texts'] else None)
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
            _notify('success','Word embeddings and sentences restored.' if word_only else 'Analysis restored. Edit it in the workspaces, then save your changes below.')
            _finish_load()
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
    with st.expander('New here? How SemLens works',expanded=st.session_state.target_word_data is None):
        st.markdown(GUIDE)
    _show_load_notices()
    sources=['Upload raw corpora (.txt)','Paste sentences','Upload CSV / TSV']
    if EXAMPLE_PATH.is_file():
        sources.append(EXAMPLE_SOURCE)
    sources+=['Saved corpora','Saved word embeddings','Pre-computed embeddings','Saved analysis']
    # Remembered, so returning from another workspace does not reset the choice.
    remembered=st.session_state.get('_data_source')
    mode=st.selectbox('Data source',sources,index=sources.index(remembered) if remembered in sources else 0,
                      help='**Upload raw corpora**: one plain-text file per corpus; SemLens finds the word.\n\n'
                           '**Paste sentences**: type or paste text for each corpus.\n\n'
                           '**Upload CSV / TSV**: a table of sentences you have already selected, one row per usage.\n\n'
                           '**Example**: a small ready-made dataset to try the tool.\n\n'
                           '**Saved corpora / word embeddings / analysis**: work you saved earlier.\n\n'
                           '**Pre-computed embeddings**: vectors you produced with your own code.')
    st.session_state['_data_source']=mode
    corpus_input=None
    if mode==EXAMPLE_SOURCE:
        st.markdown(EXAMPLE_DESCRIPTION)
        if st.button('Load the example',type='primary'):
            try:
                data=load_from_csv(EXAMPLE_PATH,'bubble',forms=['bubbles'])
                _set_loaded(data)
                st.session_state.source_texts={}
                st.session_state.saved_cache_path=None
                st.session_state.pop('_corpus_stats',None)
                st.session_state.chronological=True
                _finish_load()
            except Exception as exc:
                st.error(str(exc))
        st.divider()
        _embedding_controls(None)
    elif mode in {'Saved analysis','Saved word embeddings'}:
        word_only=mode=='Saved word embeddings'
        if word_only:
            st.caption('Reopen vectors for a saved word and its forms, with the matching sentences and model/layer settings. To analyse other words, use Saved corpora.')
        _saved_file_controls(word_only)
        upload=st.file_uploader('Or open an exported embedding file' if word_only else 'Or open an exported analysis',type=['semlens'])
        if upload is not None and st.button('Open uploaded file'):
            try:
                _load_snapshot(load_project(upload),analysis_name=None if word_only else Path(upload.name).stem)
                _notify('success','Analysis restored.')
                _finish_load()
            except Exception as exc:
                st.error(str(exc))
    else:
        word=st.text_input('Target word',value=st.session_state.target_word,key=f'target_{mode}',
                           help='The word to study, in its dictionary form: run, bubble, Blase. Add its other forms below.').strip()
        forms=_forms(st.text_input('Other forms (optional)',key=f'forms_{mode}',placeholder='For example: runs, running, ran',
                                   help='Other spellings or inflected forms to count as the same word, separated by commas: runs, running, ran. '
                                        'End a form with * to match any ending: runn* finds running and runner. '
                                        'Only whole words are matched, so run never finds runway.'))
        with st.expander('Usage selection options'):
            cap_text=st.text_input('Maximum occurrences per corpus',value='all',key=f'cap_{mode}',
                                   help='“all” uses every occurrence. A number such as 500 selects a reproducible random subset of that size from each corpus before embedding; plots show the retained usages.')
            try:
                _parse_cap(cap_text)
            except ValueError as exc:
                st.error(str(exc))
            match_case=st.checkbox('Match upper and lower case exactly',value=False,key=f'case_{mode}',
                                   help='Off: Polish and polish are the same word. On: only the capitalisation you typed is found, '
                                        'which separates names from common nouns (May / may, Bush / bush).')
            split='auto'
            if mode=='Upload raw corpora (.txt)':
                divisions={'auto':'Automatically, at sentence punctuation','lines':'One sentence per line','paragraphs':'One paragraph per context'}
                split=st.selectbox('How the text is divided into contexts',list(divisions),format_func=divisions.get,key='raw_split',
                                   help='Each usage is read, and represented, inside one context.\n\n'
                                        '**Automatically**: sentences end at . ! ? and at blank lines. Titles such as Mr. and Dr. do not end a sentence.\n\n'
                                        '**One sentence per line**: for text already split into sentences, including text without punctuation.\n\n'
                                        '**One paragraph per context**: paragraphs separated by a blank line are kept whole, for when a sentence is too little context.')
        if mode in {'Paste sentences','Upload raw corpora (.txt)'}:
            texts={}
            lines=mode=='Paste sentences'
            # Texts loaded earlier in this session stay available, so another word can be
            # looked up after visiting the other workspaces without entering them again.
            kept=st.session_state.source_texts if st.session_state.get('_source_kind')==mode else {}
            if lines:
                split='lines'
                count=int(st.number_input('Number of corpora',min_value=1,max_value=10,value=max(2,len(kept)) if kept else 2,step=1))
                names=[]
                values=[]
                for i in range(count):
                    with st.expander(f'Corpus {i+1}',expanded=count<=2):
                        earlier=list(kept.items())[i] if i<len(kept) else (f'corpus_{i+1}','')
                        names.append(st.text_input('Corpus name',value=earlier[0],key=f'paste_name_{i}',
                                                   help='A short label shown in plots and tables, such as a period or a genre.').strip())
                        values.append(st.text_area('Sentences (one per line)',value=earlier[1],key=f'paste_text_{i}',height=140))
                texts=dict(zip(names,values))
            else:
                uploads=st.file_uploader('Corpus text files (up to 10)',type=['txt'],accept_multiple_files=True,
                                         help='One plain-text (.txt) file per corpus, for example one per period. Export from Word or a PDF as plain text first.')
                names=[]
                values=[]
                for i,f in enumerate(uploads):
                    names.append(st.text_input(f'Corpus name for {f.name}',value=Path(f.name).stem,key=f'raw_name_{i}').strip())
                    text,encoding=decode_text(f.getvalue())
                    values.append(text)
                    if encoding!='UTF-8':
                        st.caption(f'{f.name} was read as {encoding}. If accented letters look wrong in the review below, save the file as UTF-8 and upload it again.')
                texts=dict(zip(names,values))
                if not uploads and kept:
                    names,texts=list(kept),dict(kept)
                    st.caption(f'Using the {len(kept)} texts loaded earlier in this session: {", ".join(kept)}. '
                               'Enter another word and load again, or upload files to replace them.')
            if st.button('Load usages',type='primary'):
                try:
                    validate_corpus_names(names)
                    records=corpus_sentences(texts,split=split)
                    found=select_usages(records,word,forms,match_case=match_case)
                    data,_=_cap(found,_parse_cap(cap_text))
                    _set_loaded(data)
                    st.session_state.source_texts=texts
                    st.session_state.source_lines=lines
                    st.session_state.source_split=split
                    st.session_state['_source_kind']=mode
                    st.session_state.saved_cache_path=None
                    st.session_state['_corpus_stats']={name:{'words':count_words(text),'found':len(found.indices_for_corpus(name))}
                                                       for name,text in texts.items()}
                    missing=[c for c in names if c not in data.corpus_labels]
                    if missing:
                        _notify('warning','No selected forms found in: '+', '.join(missing))
                    _finish_load()
                except Exception as exc:
                    st.error(str(exc))
            # Store all source texts for precomputation even if no target is selected yet.
            corpus_input=(names,texts,split)
        elif mode=='Upload CSV / TSV':
            upload=st.file_uploader('Usage table',type=['csv','tsv'])
            st.caption('One row per sentence, with the columns **sentence** and **corpus**. Optional **start** and **end** give the character position of the '
                       'target in the sentence (counting from 0, end not included) and select exactly that occurrence; without them every occurrence of the word is found. '
                       'Any other columns, such as a date or a source, are kept and returned with your annotations.')
            if st.button('Load usages',type='primary',disabled=upload is None):
                try:
                    data=load_from_csv(io.StringIO(decode_text(upload.getvalue())[0]),word,
                                       delimiter='\t' if upload.name.endswith('.tsv') else ',',forms=forms,match_case=match_case)
                    data,_=_cap(data,_parse_cap(cap_text))
                    _set_loaded(data)
                    st.session_state.source_texts={}
                    st.session_state.saved_cache_path=None
                    st.session_state.pop('_corpus_stats',None)
                    _finish_load()
                except Exception as exc:
                    st.error(str(exc))
        elif mode=='Pre-computed embeddings':
            vectors=st.file_uploader('Embeddings',type=['pt','npy'])
            metadata=st.file_uploader('Matching sentence metadata',type=['csv','tsv'])
            st.caption('One vector per metadata row. Include start/end when a sentence contains multiple matching occurrences.')
            model_id=st.text_input('Original model ID (needed for definitions)',key='import_model')
            col1,col2=st.columns(2)
            layer=int(col1.number_input('Saved layer (-1 = final output)',value=-1,step=1,key='import_layer',
                                        help='-1 is the final output, -2 the penultimate layer, and so on. 0 is the input embedding layer.'))
            representation=col2.selectbox('Saved representation',['word','sentence'],key='import_repr')
            revision=st.text_input('Model revision (optional)',key='import_revision')
            if st.button('Load embeddings',type='primary',disabled=vectors is None or metadata is None):
                try:
                    if vectors.name.endswith('.npy'):
                        embs=torch.from_numpy(np.load(io.BytesIO(vectors.getvalue()),allow_pickle=False))
                    else:
                        embs=torch.load(io.BytesIO(vectors.getvalue()),weights_only=True,map_location='cpu')
                    data=load_from_csv(io.StringIO(decode_text(metadata.getvalue())[0]),word,
                                       delimiter='\t' if metadata.name.endswith('.tsv') else ',',forms=forms,all_occurrences=False,match_case=match_case)
                    embs=validate_embeddings(embs,len(data))
                    data,indices=_cap(data,_parse_cap(cap_text))
                    config={'model_id':model_id.strip(),'revision':revision.strip() or None,'layer':layer,
                            'representation':representation,'pooling':'mean','max_length':256}
                    _set_loaded(data,embedded_usages_from_precomputed(data,embs[indices],metadata=config),'precomputed')
                    st.session_state.source_texts={}
                    st.session_state.saved_cache_path=None
                    st.session_state.pop('_corpus_stats',None)
                    _finish_load()
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
                                    help='Word representations require a token cache. Sentence representations average the stored token vectors. '
                                         'Saved corpora are encoded once for every word, so XL-LEXEME target markers cannot be applied to them; '
                                         'embed loaded usages directly when you want marked targets.')
            if st.button('Load word from saved corpora',type='primary',disabled=path is None):
                try:
                    cache=CorpusCache(path)
                    data,embedded=cache.select(word,forms,representation=representation,max_per_corpus=_parse_cap(cap_text),match_case=match_case)
                    _set_loaded(data,embedded,'saved')
                    st.session_state.source_texts=cache.source_texts()
                    st.session_state.source_lines=cache.info['splitter']=='lines_v1'
                    st.session_state.pop('_corpus_stats',None)
                    st.session_state['_source_kind']=mode
                    st.session_state.saved_cache_path=str(path)
                    _finish_load()
                except Exception as exc:
                    st.error(str(exc))

        if mode not in {'Saved corpora','Pre-computed embeddings'}:
            st.divider()
            _embedding_controls(corpus_input if mode in {'Paste sentences','Upload raw corpora (.txt)'} else None)

    data=st.session_state.target_word_data
    if data is not None:
        with st.expander('Review loaded occurrences',key='review_occurrences',on_change='rerun') as panel:
            if panel.open:
                _review_occurrences(data)
    if mode in {'Saved corpora','Saved word embeddings','Saved analysis','Pre-computed embeddings'} and st.session_state.embedded_usages is not None:
        with st.expander('Save word embeddings'):
            _save_word_embeddings()


def _review_occurrences(data):
    """Counts, relative frequencies, word forms and a concordance of what was loaded."""
    order=ordered_labels(data)
    stats=st.session_state.get('_corpus_stats') or {}
    rows=[]
    for corpus in order:
        row={'Corpus':corpus,'Occurrences loaded':len(data.indices_for_corpus(corpus))}
        if corpus in stats and stats[corpus]['words']:
            found,words=stats[corpus]['found'],stats[corpus]['words']
            row.update({'Found in the corpus':found,'Corpus size (words)':words,'Per million words':round(found/words*1e6,1)})
        rows.append(row)
    readable_table(pd.DataFrame(rows))
    if stats:
        st.caption('Per million words is the number found divided by the size of the corpus, so corpora of different sizes can be compared. '
                   'Occurrences loaded is smaller than the number found when a maximum per corpus is set.')
    forms=pd.crosstab(pd.Series([u.matched_form.casefold() for u in data.usages],name='Form'),pd.Series(data.corpus_list,name='Corpus'))
    forms=forms.reindex(columns=order,fill_value=0)
    if len(forms)>1:
        forms['Total']=forms.sum(axis=1)
        st.write('Word forms')
        readable_table(forms.sort_values('Total',ascending=False).reset_index())
    st.write('Concordance')
    cols=st.columns([2,2,2])
    corpora=cols[0].multiselect('Corpora',order,default=order,key=f'kwic_corpora_{st.session_state.data_revision}')
    query=cols[1].text_input('Lines containing',key='kwic_search',placeholder='For example: burst, hous*',
                             help='Whole words or phrases, ignoring case. Separate alternatives with commas; end a term with * to match any ending.').strip()
    sorting=cols[2].selectbox('Sort lines by',['Order in the corpus','Word after the target','Word before the target'],key='kwic_sort',
                              help='Sorting by a neighbouring word groups recurring phrases, such as “bubble burst” or “housing bubble”.')
    pattern=usage_search_pattern(query) if query else None
    chosen=[i for i,u in enumerate(data.usages) if u.corpus in corpora and (pattern is None or pattern.search(u.sentence))]
    lines=[(i,)+keyword_in_context(data.usages[i].sentence,data.usages[i].char_span) for i in chosen]
    if sorting=='Word after the target':
        lines.sort(key=lambda line:[w.casefold() for w in re.findall(r'\w+',line[3])[:3]])
    elif sorting=='Word before the target':
        lines.sort(key=lambda line:[w.casefold() for w in reversed(re.findall(r'\w+',line[1])[-3:])])
    if not lines:
        st.caption('No loaded occurrence matches.')
    else:
        per_page=100
        page=1
        if len(lines)>per_page:
            page=int(st.number_input('Page',min_value=1,max_value=-(-len(lines)//per_page),value=1,key=f'kwic_page_{len(lines)}_{sorting}'))
        shown=lines[(page-1)*per_page:page*per_page]
        st.html(concordance_html([(data.usages[i].corpus,left,target,right) for i,left,target,right in shown]))
        st.caption(f'{len(lines):,} of {len(data):,} occurrences'+(f', lines {(page-1)*per_page+1:,}–{(page-1)*per_page+len(shown):,}' if len(lines)>per_page else '')
                   +'. The highlighted word is the exact occurrence that is represented. Hover a point in any plot to read its whole sentence.')


def _save_word_embeddings():
    data=st.session_state.target_word_data
    embedded=st.session_state.embedded_usages
    if embedded is None:
        st.info('Embed the loaded usages first, choosing Target word or Whole sentence above.')
        return
    meta=embedded.metadata
    representation=meta.get('representation','unspecified')
    st.caption(f'Save exactly the {len(data):,} loaded occurrences of “{data.word}” and its selected forms. Includes their sentences and character positions. Selection caps apply; other corpus sentences are excluded.')
    st.write(f"Representation: **{representation}** · Model: **{meta.get('model_id') or 'unspecified'}** · Layer: **{meta.get('layer','unspecified')}**"
             +(' · Target markers: **on**' if meta.get('target_markers') else ''))
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
    st.caption('The language model reads every loaded sentence and represents each usage as a list of numbers (an embedding). '
               'Usages it treats as similar in meaning get similar numbers; everything in the other workspaces is built on them.')
    representation=st.radio('Represent each usage by',['Target word','Whole sentence'],horizontal=True,
                            help='**Target word** (recommended): the model’s representation of the word itself, as coloured by its sentence. Best for word meaning.\n\n'
                                 '**Whole sentence**: the average over every word of the sentence. It also reflects what the sentence is about, '
                                 'so usages can differ because the topic differs.')
    rep='word' if representation=='Target word' else 'sentence'
    with st.expander('Advanced embedding settings'):
        preset=st.selectbox('Language model',list(MODEL_PRESETS),key='encoder_preset',
                            help='The model that reads each sentence and turns each usage into numbers. The first use of a model downloads it '
                                 '(about 2 GB for the large ones) and later uses start from the saved copy. All of these read many languages unless marked English.')
        model_id=MODEL_PRESETS[preset]
        if model_id is None:
            model_id=st.text_input('Encoder model',value='pierluigic/xl-lexeme',key='encoder_model')
            st.caption('A Hugging Face encoder model ID, such as FacebookAI/xlm-roberta-base, or the path of a model folder on this computer.')
        layer=int(st.number_input('Layer (-1 = final output)',value=-1,step=1,key='encoder_layer',
                                  help='Negative values count back from the top: -1 is the final output, -2 the penultimate layer, -3 the one before it. '
                                       '0 is the input embedding layer; 1 and higher count transformer layers from the bottom.'))
        max_length=int(st.number_input('Tokens per context window',min_value=32,max_value=4096,value=256,step=32,
                                       help='How much text the model reads at once, in word pieces (roughly 0.75 words each). Longer contexts are read in overlapping windows, so nothing is cut off.'))
        batch_size=int(st.number_input('Inference batch size',min_value=1,max_value=128,value=16,
                                       help='How many contexts are processed together. It affects speed and memory only, never the results. Lower it if the computer runs out of memory.'))
        use_markers=st.checkbox('Mark the target word when the model supports it',value=True,key='encoder_target_markers',
                                help='XL-LEXEME was trained with the target wrapped in two special tokens, `<t>` and `</t>`. With this on, SemLens adds them '
                                     'around each target before encoding Target word representations, and around the word in each definition. '
                                     'Models without these tokens, and Whole sentence representations, are unaffected. '
                                     'Turn it off to encode the plain sentence, as earlier versions of SemLens did.')
    if st.button('Embed loaded usages',type='primary',disabled=st.session_state.target_word_data is None):
        try:
            progress=st.progress(0.0)
            with st.spinner('Loading the encoder and embedding usages…'):
                model=get_model(model_id)
                embedded=embed_usages(model,st.session_state.target_word_data,representation=rep,layer=layer,max_length=max_length,
                                     batch_size=batch_size,progress_callback=lambda b,n:progress.progress(b/n),
                                     target_markers=None if use_markers else False)
            reset_downstream_of('embeddings')
            st.session_state.embedded_usages=embedded
            st.session_state.model_name=model.name
            st.session_state.embeddings_source='model'
            progress.empty()
            marked=' with the target marked by <t>…</t>' if embedded.metadata.get('target_markers') else ''
            st.success(f'Created {embedded.dim}-dimensional {rep} embeddings{marked}. Explore all corpora in the other tabs.')
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
            names,texts,split=corpus_input
            lines=split=='lines'
            st.write('Save the complete corpora once, then reopen them and switch words in a later session. Source text and sentences are included.')
            name=st.text_input('Saved corpus collection name',value='My corpora')
            storage_choice=st.selectbox('What to save',['Word and sentence representations','Sentence representations only'])
            storage='tokens' if storage_choice.startswith('Word') else 'sentences'
            dtype=st.selectbox('Storage precision',['float32','float16'],help='float16 halves vector storage but rounds values. float32 preserves the computed precision.')
            signature=hashlib.sha256(json.dumps([texts,split,model_id,layer,max_length,storage,dtype],sort_keys=True).encode()).hexdigest()
            if st.button('Estimate storage'):
                try:
                    validate_corpus_names(names)
                    records=corpus_sentences(texts,split=split)
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
                                max_length=max_length,batch_size=batch_size,split=split,estimate=estimate,
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
        settings={k:st.session_state[k] for k in ('definition_settings','chronological','corpus_order','distance_metric','sampling_mode','sampling_repetitions','sampling_seed','source_lines','source_split','palette_name')}
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
