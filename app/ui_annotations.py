"""Click-to-assign sense annotation with undo and readable export tables."""
from __future__ import annotations
import pandas as pd
import streamlit as st
from app.annotation_store import (get_annotations,get_sense_classes,get_active_sense_class,add_sense_class,
    set_active_sense_class,remove_sense_class,set_annotation,remove_annotation,clear_annotations,
    remember_annotations,undo_annotations)
from app.ui_explore import render_plot
from app.presentation import readable_table
from semlens.utils import get_palette


def assign_new_selection(result,active):
    """Consume only genuine chart events, never a stale selection on a rerun."""
    event=result.get('selection_event')
    if not event or not event[1]:
        return False
    consumed=st.session_state.setdefault('_annotation_events_consumed',{})
    if consumed.get(event[0],0)>=event[1]:
        return False
    consumed[event[0]]=event[1]
    selected=result['clicked_points']
    if not active or not any(get_annotations().get(i)!=active for i in selected):
        return False
    remember_annotations()
    for i in selected:
        set_annotation(i,active)
    return True


def render_annotations():
    st.header('Sense annotation')
    eu=st.session_state.embedded_usages
    if eu is None:
        st.info('Load data and embed usages first.')
        return
    data=st.session_state.target_word_data
    classes=get_sense_classes()
    annotations=get_annotations()
    with st.form('add_sense',clear_on_submit=True):
        c1,c2=st.columns([3,1])
        name=c1.text_input('New sense class')
        if c2.form_submit_button('Add class') and name.strip():
            add_sense_class(name.strip())
            set_active_sense_class(name.strip())
            st.rerun()
    if classes:
        active=get_active_sense_class()
        if active not in classes:
            active=classes[0]
            set_active_sense_class(active)
        palette=get_palette('Tol Bright (colourblind)')
        with st.container(horizontal=True):
            for i,sense in enumerate(classes):
                colour=palette[i%len(palette)]
                selected=sense==active
                st.html(f'''<style>.st-key-sense_{i} button {{
                    border:3px solid {colour if selected else 'var(--sl-border)'} !important;
                    background:var(--sl-surface) !important; color:var(--sl-text) !important;
                    box-shadow:{f'0 0 0 1px {colour}' if selected else 'none'} !important; }}
                    .st-key-sense_{i} button p {{ color:var(--sl-text) !important; }}</style>''')
                st.button(('✓ ' if selected else '')+sense,key=f'sense_{i}',
                          help=f'{sum(s==sense for s in annotations.values())} annotated occurrences',
                          on_click=set_active_sense_class,args=(sense,))
    else:
        active=''
        st.caption('Add a sense class, then select usages in the plot.')
    st.caption(f'{len(annotations):,} of {len(data):,} occurrences annotated. Choose a sense above, then click a point (or select a group) to assign it immediately.')
    has_defs=st.session_state.def_space_all is not None
    source=st.radio('Annotation space',['Full embeddings','Definition space'] if has_defs else ['Full embeddings'],horizontal=True)
    def_labels=None
    vectors=eu.embeddings
    if source=='Definition space':
        vectors=st.session_state.def_space_all
        def_labels=[s.split(': ',1)[-1] for s in st.session_state.definitions_formatted]
    result=render_plot(data,vectors,key='annotation',definition_labels=def_labels,annotation=True,
                       annotations=annotations,sense_classes=classes)
    selected=result['clicked_points']
    if assign_new_selection(result,active):
        st.rerun()
    st.caption(f'{len(selected)} occurrence(s) selected.')
    c2,c3=st.columns(2)
    if c2.button('Clear selected',disabled=not selected):
        remember_annotations()
        for i in selected:
            remove_annotation(i)
        st.rerun()
    if c3.button('Undo last annotation change'):
        undo_annotations()
        st.rerun()
    with st.expander('Manage classes and annotations'):
        if classes:
            remove=st.selectbox('Class to remove',classes)
            if st.button('Remove class and its annotations'):
                remove_sense_class(remove)
                st.rerun()
        confirm=st.checkbox('Delete all annotations')
        if confirm and st.button('Confirm deletion'):
            clear_annotations()
            st.rerun()
    rows=[]
    for i,u in enumerate(data.usages):
        rows.append({'usage_id':u.usage_id,'sentence_id':u.sentence_id,'lemma':u.word,'matched_form':u.matched_form,
                     'sentence':u.sentence,'corpus':u.corpus,'start':u.char_start,'end':u.char_end,'sense':annotations.get(i,'')})
    with st.expander('Review and export annotations'):
        include_all=st.checkbox('Include unannotated occurrences',value=False)
        df=pd.DataFrame(rows)
        if not include_all:
            df=df[df.sense!='']
        readable_table(df,text_columns=['sentence'],column_order=['sentence','sense','corpus','matched_form','start','end','usage_id','sentence_id','lemma'])
        st.download_button('Download annotation CSV',data=df.to_csv(index=False),file_name=f'{data.word}_annotations.csv',mime='text/csv')
