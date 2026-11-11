"""Definition inputs and optional diagnostics around the shared corpus views."""
from __future__ import annotations
import json
import hashlib
import html
import textwrap
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from semlens.definitions import format_definitions,fetch_wiktionary_definitions,embed_definitions,definition_inputs
from semlens.spaces import project_to_definition_space
from app.common import get_model
from app.state import reset_downstream_of
from app.ui_explore import render_plot,render_comparison,render_definition_distribution
from app.presentation import plot_chart,readable_table
import pandas as pd


def _definition_editor_key(origin):
    return 'definition_draft_'+hashlib.sha256(json.dumps(origin).encode()).hexdigest()


def _remember_definition_draft(key):
    # Keep drafts outside widget state so navigating away does not discard edits.
    st.session_state.setdefault('_definition_drafts',{})[key]=st.session_state[key]


def _edit_definitions(origin,definitions=None,*,replace=False):
    key=_definition_editor_key(origin)
    drafts=st.session_state.setdefault('_definition_drafts',{})
    if replace or key not in drafts:
        if definitions is None:
            return []
        drafts[key]='\n'.join(definitions)
        st.session_state[key]=drafts[key]
    if key not in st.session_state:
        st.session_state[key]=drafts[key]
    text=st.text_area('Edit definitions (one per line)',height=200,key=key,
                      on_change=_remember_definition_draft,args=(key,))
    return [line.strip() for line in text.splitlines() if line.strip()]


def definition_heatmap(matrix,labels,title):
    # Short axes leave room for the matrix; complete labels remain in hover and key.
    ticks=[f'D{i+1}' for i in range(len(labels))]
    wrapped=['<br>'.join(html.escape(line) for line in textwrap.wrap(label,65)) for label in labels]
    custom=[[[wrapped[i],wrapped[j]] for j in range(len(labels))] for i in range(len(labels))]
    fig=go.Figure(go.Heatmap(z=matrix,x=ticks,y=ticks,customdata=custom,colorscale='RdBu',
                            zauto=True,hovertemplate='%{customdata[0]}<br>↔ %{customdata[1]}<br>%{z:.4f}<extra></extra>'))
    side=max(440,len(labels)*42+160)
    fig.update_layout(title=title,height=side,margin=dict(l=65,r=85,t=70,b=65),
                      xaxis=dict(tickmode='array',tickvals=ticks,tickangle=-45,tickfont=dict(size=12),constrain='domain'),
                      yaxis=dict(tickmode='array',tickvals=ticks,tickfont=dict(size=12),autorange='reversed',scaleanchor='x',scaleratio=1))
    return fig


def render_definitions():
    st.header('Definitions & definition space')
    eu=st.session_state.embedded_usages
    if eu is None:
        st.info('Load data and embed usages first.')
        return
    data=st.session_state.target_word_data
    source=st.selectbox('Definition source',['Manual input','Wiktionary','JSON file'])
    raw=[]
    if source=='Manual input':
        raw=[s.strip() for s in st.text_area('One definition per line',value='\n'.join(st.session_state.definitions_raw),
                                             height=160,key=f'def_text_{data.word}_{st.session_state.data_revision}').splitlines() if s.strip()]
    elif source=='Wiktionary':
        with st.expander('Dictionary settings'):
            edition=st.text_input('Wiktionary edition',value='en')
            language=st.text_input('Language section heading',value='English',help='The exact section heading in that Wiktionary edition, for example English on en.wiktionary.org.')
        origin=('wiktionary',data.word,edition,language)
        fetched=None
        replace=False
        if st.button('Fetch definitions'):
            try:
                fetched=fetch_wiktionary_definitions(data.word,language=edition,target_language_name=language)
                replace=True
                if not fetched:
                    st.warning('No definitions found for that language section. You can enter definitions below.')
            except Exception as exc:
                st.error(str(exc))
        legacy=st.session_state.pop('_fetched_definitions',None)
        if fetched is None and legacy and legacy[0]==data.word:
            fetched=legacy[1]
        raw=_edit_definitions(origin,fetched,replace=replace)
        st.caption('Edit, add or remove definitions, then click Create definition space to apply them. Fetching again replaces this draft.')
    else:
        upload=st.file_uploader('Definitions JSON',type=['json'])
        if upload:
            try:
                content=json.loads(upload.getvalue())
                raw=content.get(data.word,content.get(data.word.split('_')[0],[]))
                if not isinstance(raw,list) or not all(isinstance(s,str) for s in raw):
                    raise ValueError('Expected a JSON object mapping words to lists of definitions.')
                if not raw:
                    st.info(f'No definitions found for {data.word}.')
                origin=('json',data.word,hashlib.sha256(upload.getvalue()).hexdigest())
                raw=_edit_definitions(origin,raw)
                st.caption('Edit, add or remove definitions, then click Create definition space to apply them.')
            except Exception as exc:
                st.error(str(exc))
                raw=[]
    settings=st.session_state.definition_settings
    with st.expander('Advanced definition settings'):
        presets={'Word: definition':'{word}: {definition}','Definition: word':'{definition}: {word}',
                 'Definition only':'{definition}','Custom':'custom'}
        current=settings.get('template','{word}: {definition}')
        default=next((name for name,template in presets.items() if template==current),'Custom')
        preset=st.selectbox('Input format',list(presets),index=list(presets).index(default))
        template=st.text_input('Template',value=current,help='Use {word} and {definition}.') if preset=='Custom' else presets[preset]
        readouts=['Target word','Whole input']
        default_readout=settings.get('readout',eu.metadata.get('representation','word'))
        readout_label=st.selectbox('Representation to extract',readouts,index=1 if default_readout=='sentence' or preset=='Definition only' else 0,
                                  key=f'def_readout_{st.session_state.data_revision}_{eu.metadata.get("representation","word")}_{preset}',
                                  help='Defaults to the representation used for the usages. You can override it here.')
        readout='word' if readout_label=='Target word' else 'sentence'
        st.caption('Definitions use the same encoder and layer as the usages. Whole input averages the text-token vectors.')
        model_id=eu.metadata.get('model_id','')
        if not model_id:
            model_id=st.text_input('Original embedding model ID',help='Use the encoder that produced the uploaded usage vectors.')
        layer=int(eu.metadata.get('layer',-1))
        st.caption(f'Model: {model_id or "not specified"} · Layer: {layer}')
        if raw:
            try:
                preview,spans=definition_inputs(data.word,raw[:1],template=template,readout=readout)
                st.code(preview[0],language=None)
            except ValueError as exc:
                st.info(str(exc))
    if st.button('Create definition space',type='primary'):
        try:
            if not raw:
                raise ValueError('Add at least one definition.')
            if not model_id:
                raise ValueError('Specify the original encoder in Advanced definition settings.')
            formatted=format_definitions(data.word,raw)
            # Validate the template before loading a model.
            definition_inputs(data.word,formatted,template=template,readout=readout)
            with st.spinner('Embedding definitions…'):
                model=get_model(model_id,eu.metadata.get('revision'))
                vectors=embed_definitions(model,data.word,formatted,layer=layer,template=template,readout=readout,
                                         max_length=int(eu.metadata.get('max_length',256)))
                if vectors.shape[1]!=eu.dim:
                    raise ValueError('The selected model does not match the uploaded embedding dimensions.')
                projected=project_to_definition_space(eu.embeddings,vectors)
            reset_downstream_of('definitions')
            st.session_state.definitions_raw=raw
            st.session_state.definitions_formatted=formatted
            st.session_state.definition_embeddings=vectors
            st.session_state.def_space_all=projected
            st.session_state.definition_settings={'template':template,'readout':readout,'model_id':model.name,'layer':layer}
            st.success(f'Created {len(formatted)} definition dimensions.')
        except Exception as exc:
            st.error(str(exc))
    space=st.session_state.def_space_all
    if space is None:
        return
    if raw!=st.session_state.definitions_raw:
        st.caption('The plots use the last created definition space. Click Create definition space to apply your edits.')
    labels=[s.split(': ',1)[-1] if ': ' in s else s for s in st.session_state.definitions_formatted]
    st.divider()
    st.caption('Each coordinate is cosine distance to a definition. Smaller values mean closer to that definition.')
    render_plot(data,space,key='defs',definition_labels=labels)
    render_definition_distribution(data,space,labels)
    render_comparison(data,space,key='defs',definition_labels=labels)
    with st.expander('Definition similarity and correlation',key='definition_heatmaps',on_change='rerun') as panel:
        if panel.open:
            vectors=st.session_state.definition_embeddings.numpy()
            normalised=vectors/np.maximum(np.linalg.norm(vectors,axis=1,keepdims=True),1e-12)
            similarity=np.clip(normalised@normalised.T,-1,1)
            if st.session_state.distance_metric=='euclidean':
                similarity=1/(1+np.linalg.norm(vectors[:,None]-vectors[None,:],axis=2))
            matrices=[('Definition similarity',similarity)]
            if len(labels)>1 and len(space)>1:
                with np.errstate(invalid='ignore',divide='ignore'):
                    correlation=np.corrcoef(space.numpy(),rowvar=False)
                matrices.append(('Correlation of definition distances',correlation))
            for title,matrix in matrices:
                fig=definition_heatmap(matrix,labels,title)
                plot_chart(fig,width='stretch')
            readable_table(pd.DataFrame({'Axis label':[f'D{i+1}' for i in range(len(labels))],'Definition':labels}),text_columns=['Definition'])
            st.caption('D1, D2, … follow the definition input order. Hover a cell to read both complete definitions; use fullscreen for a larger matrix.')
            st.caption('Each colour scale adapts to the values in its matrix. Compare the numeric values and colour bars when reading different heatmaps.')
            st.caption('Empty correlation cells indicate a constant feature with undefined correlation.')
