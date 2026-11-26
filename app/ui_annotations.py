"""Click-to-assign sense annotation with undo and readable export tables.

By default a usage gets one sense. Two options, both off by default and kept
in a collapsed box, let a usage carry several senses and let each sense be
rated on a scale (graded annotation).
"""
from __future__ import annotations
import html
import json
import re
import zipfile
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from app.annotation_store import (get_annotations,get_sense_classes,get_active_sense_class,add_sense_class,
    set_active_sense_class,remove_sense_class,set_label,remove_sense,remove_annotation,clear_annotations,
    remember_annotations,undo_annotations,rename_sense_class,get_options,set_option,get_active_rating,
    set_active_rating,keep_first_senses)
from semlens.annotations import (MAX_SCALE,MIN_SCALE,SENSE_SEPARATOR,counts_towards,highest_rating,label_text,
    normalise_annotations,parse_rating,with_several_senses)
from semlens.utils import keyword_in_context,usage_search_pattern
import plotly.graph_objects as go
from app.ui_explore import render_plot
from app.common import STORAGE,ordered_labels
from app.grouping import current_view,loaded_words,render_grouping_controls,words_active
from semlens.projects import load_project
from app.presentation import readable_table,plot_chart
from semlens.utils import get_palette

NO_SENSE='—'
# In the list with several graded senses: the usage has this sense but no rating yet.
UNRATED='✓'
SENSE_COLUMN_PREFIX='sense:'


def _graded():
    return bool(get_options()['graded'])


def _multiple():
    return bool(get_options()['multiple'])


def _scale():
    return int(get_options()['scale'])


def _asks_for_rating():
    return _graded() and get_options().get('rating_entry','ask')=='ask'


def _new_selection(result):
    """The usages picked by a genuine new chart event, or None for a stale one replayed on a rerun."""
    event=result.get('selection_event')
    if not event or not event[1]:
        return None
    consumed=st.session_state.setdefault('_annotation_events_consumed',{})
    if consumed.get(event[0],0)>=event[1]:
        return None
    consumed[event[0]]=event[1]
    return list(result['clicked_points'])


def apply_assignment(selected,active,rating=None,*,toggle=False):
    """Give the selected usages the active sense at a rating; True when anything changed.

    One sense replaces a usage's senses; with several senses allowed it is
    added instead. With ``toggle``, a usage that already has the sense at this
    rating loses it.
    """
    annotations=get_annotations()
    rating=rating if _graded() else None
    changes={}
    for i in selected:
        current=annotations.get(i,{})
        if toggle and active in current and current[active]==rating:
            new={s:r for s,r in current.items() if s!=active}
        elif _multiple():
            new={**current,active:rating}
        else:
            new={active:rating}
        if new!=current:
            changes[i]=new
    if not changes:
        return False
    remember_annotations()
    for i,label in changes.items():
        set_label(i,label)
    return True


def assign_new_selection(result,active,rating=None):
    """Apply a genuine chart selection, never a stale one on a rerun.

    When ratings are typed, the selection waits in ``_rating_request`` for the
    rating box instead. Otherwise, with several senses allowed, clicking a
    single usage that already has the active sense (at the same rating) takes
    it off again.
    """
    selected=_new_selection(result)
    if not selected or not active:
        return False
    if _asks_for_rating():
        st.session_state['_rating_request']={'indices':selected,'sense':active}
        return True
    rating=(rating if rating is not None else get_active_rating()) if _graded() else None
    toggle=_multiple() and len(selected)==1 and st.session_state.get('annotation_interaction','Click')=='Click'
    return apply_assignment(selected,active,rating,toggle=toggle)


def _cancel_rating_request():
    st.session_state.pop('_rating_request',None)


@st.dialog('Give a rating',width='medium',on_dismiss=_cancel_rating_request)
def _rating_dialog():
    """Ask for the rating of the usages just clicked or lassoed."""
    request=st.session_state.get('_rating_request')
    if not request:
        return
    indices,sense=request['indices'],request['sense']
    data=st.session_state.target_word_data
    scale=_scale()
    annotations=get_annotations()
    if len(indices)==1:
        usage=data.usages[indices[0]]
        corpus='' if st.session_state.get('blind_annotation') else f'<small>{html.escape(usage.corpus)}</small><br>'
        st.markdown(f'<div>{corpus}{html.escape(usage.sentence[:usage.char_start])}<mark>{html.escape(usage.matched_form)}</mark>'
                    f'{html.escape(usage.sentence[usage.char_end:])}</div>',unsafe_allow_html=True)
        current=annotations.get(indices[0],{})
        if sense in current:
            st.caption(f'This usage already has “{sense}”'+(f', rated {current[sense]}.' if current[sense] is not None else ', without a rating.'))
    else:
        st.markdown(f'**{len(indices):,} usages selected.** They all get the same rating.')
    with st.form('semlens_rating_form',border=False):
        # A text box rather than a number box: Enter in a number box can submit
        # the form before the typed digit is committed.
        typed=st.text_input(f'How clearly does this have the sense “{sense}”? 1 = barely, {scale} = clearly',
                            placeholder=f'Type a number from 1 to {scale}, then press Enter',key='semlens_rating_input')
        c1,c2=st.columns(2)
        saved=c1.form_submit_button('Save rating',type='primary',width='stretch')
        cancelled=c2.form_submit_button('Cancel',width='stretch')
    removing=False
    if len(indices)==1 and sense in annotations.get(indices[0],{}):
        removing=st.button(f'Remove “{sense}” from this usage',width='stretch')
    if saved:
        value=typed.strip()
        if not value.isdigit() or not 1<=int(value)<=scale:
            st.error(f'Type a whole number from 1 to {scale}, or press Cancel.' if value else f'Type a rating from 1 to {scale} first, or press Cancel.')
            return
        apply_assignment(indices,sense,int(value))
        _cancel_rating_request()
        st.rerun()
    if cancelled:
        _cancel_rating_request()
        st.rerun()
    if removing:
        remember_annotations()
        remove_sense(indices[0],sense)
        _cancel_rating_request()
        st.rerun()


def _option_changed(name,widget_key):
    """Copy an option from its widget to the store, refusing changes that would lose annotations."""
    value=st.session_state[widget_key]
    if name=='multiple' and not value and with_several_senses(get_annotations()):
        st.session_state[widget_key]=True
        st.session_state['_annotation_multiple_conflict']=True
        return
    if name=='scale':
        value=int(value)
        top=highest_rating(get_annotations())
        if value<top:
            st.session_state[widget_key]=top
            st.session_state['_annotation_scale_notice']=(f'Ratings up to {top} are already in use, so the scale cannot have fewer than {top} points. '
                                                          'Change or remove those ratings first.')
            value=top
    set_option(name,value)


def _option_widget(kind,label,name,**kwargs):
    widget_key=f'_annotation_option_{name}'
    # The store is the source of truth: it survives leaving this workspace and
    # is restored with a saved analysis, so the widget starts from it each time.
    st.session_state[widget_key]=get_options()[name]
    return kind(label,key=widget_key,on_change=_option_changed,args=(name,widget_key),**kwargs)


def render_annotation_options():
    """Several senses per usage and graded annotation: off by default, in a collapsed box."""
    with st.expander('Annotation options'):
        _option_widget(st.checkbox,'Allow several senses per usage','multiple',
                       help='Let one usage carry more than one sense. Clicking a usage then adds the active sense, or takes it off if the usage has it already; '
                            'a lasso adds it to every usage inside. Frequencies count a usage once under each of its senses.')
        if st.session_state.pop('_annotation_multiple_conflict',False):
            st.session_state['_annotation_multiple_conflict_open']=True
        if st.session_state.get('_annotation_multiple_conflict_open'):
            several=with_several_senses(get_annotations())
            if several:
                st.warning(f'{several:,} usages have more than one sense. To switch this off, keep only the first sense of each '
                           '(the one given first). Undo brings the others back.')
                if st.button('Keep only the first sense of each usage'):
                    keep_first_senses()
                    set_option('multiple',False)
                    st.session_state.pop('_annotation_multiple_conflict_open',None)
                    st.rerun()
            else:
                st.session_state.pop('_annotation_multiple_conflict_open',None)
        _option_widget(st.checkbox,'Graded annotation','graded',
                       help='Give each sense a rating, from 1 (the usage barely has this sense) to the top of the scale (it clearly does). '
                            'Switching this off hides the ratings without deleting them.')
        if _graded():
            _option_widget(st.number_input,'Points on the scale','scale',min_value=MIN_SCALE,max_value=MAX_SCALE,step=1,
                           help='How many ratings to choose from. 4 is common in semantic-change annotation: 1 = barely, 4 = clearly.')
            _option_widget(st.radio,'Giving ratings','rating_entry',options=['ask','default'],
                           format_func={'ask':'Type a rating after each click or lasso',
                                        'default':'Use a default rating, set with the buttons above the plot'}.get,
                           help='Typing: each click or lasso opens a small box; type the rating and press Enter. '
                                'Default rating: every click or lasso gives the rating chosen above the plot, which is quicker for long runs of similar usages. '
                                'Ratings can be changed later in either way, or in the list below the plot.')
            notice=st.session_state.pop('_annotation_scale_notice',None)
            if notice:
                st.warning(notice)


def _sense_buttons(classes,annotations):
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
                      help=f'{sum(sense in label for label in annotations.values())} annotated usages',
                      on_click=set_active_sense_class,args=(sense,))
    return active


def _hide_guide(for_good):
    st.session_state['_annotation_guide_hidden']=True
    if for_good:
        st.session_state['annotation_guide_off']=True


def _render_guide():
    """Four steps for a first annotation, ticked off as they are done. Shown until “Don't show this again”."""
    if st.session_state.get('annotation_guide_off') or st.session_state.get('_annotation_guide_hidden'):
        return
    senses,annotations=get_sense_classes(),get_annotations()
    steps=[(bool(senses),'**Name a sense** in the box below, such as *financial*, and click **Add sense**. Add one for each sense you want to tell apart.'),
           (bool(senses),'**Choose it**: click its button. The chosen sense has a coloured border and a ✓.'),
           (bool(annotations),'**Click a point** in the plot to give that usage the chosen sense, or switch Selection to **Lasso** and circle a group. '
                              'Hover a point first to read its sentence.'),
           (bool(annotations),'**Read the counts** below the plot: how often each sense occurs in each corpus. Undo is under the plot.')]
    with st.container(border=True,key='annotation_guide'):
        st.markdown('**How to annotate, step by step**')
        for number,(done,text) in enumerate(steps,1):
            st.markdown(f'{":green[:material/check_circle:]" if done else f"**{number}.**"} {text}')
        with st.container(horizontal=True):
            st.button('Hide for now',type='tertiary',on_click=_hide_guide,args=(False,),key='annotation_guide_hide',
                      help='It comes back next time SemLens opens.')
            st.button('Don’t show this again',type='tertiary',on_click=_hide_guide,args=(True,),key='annotation_guide_off_button',
                      help='Remembered on this computer. Restore the default settings (bottom of the sidebar) to see it again.')


def _rating_picker():
    scale=_scale()
    chosen=st.segmented_control('Default rating',list(range(1,scale+1)),default=get_active_rating(),
                                key=f'annotation_rating_{scale}',
                                help=f'Clicking or lassoing usages gives them the active sense at this rating: 1 = barely, {scale} = clearly. '
                                     'To type a rating for each click instead, change “Giving ratings” in Annotation options.')
    if chosen is not None:
        set_active_rating(chosen)
    return get_active_rating()


def _hover_notes(data,annotations):
    graded=_graded()
    notes=[]
    for i in range(len(data)):
        label=annotations.get(i)
        if label:
            notes.append(('Senses: ' if len(label)>1 else 'Sense: ')+label_text(label,graded=graded))
        else:
            notes.append(None)
    return notes


def render_annotations():
    st.header('Sense annotation')
    st.caption('Define the senses you want to distinguish, then assign usages to them: by clicking or lassoing points in the plot, or line by line in the list below it. '
               'SemLens never assigns a sense for you.')
    eu=st.session_state.embedded_usages
    if eu is None:
        st.info('Annotation needs usages and their embeddings: load data and embed it in Data & Model, or open a saved analysis below.')
        render_previous_annotations(None)
        return
    data=st.session_state.target_word_data
    _render_guide()
    render_grouping_controls(data,'annotation')
    notice=st.session_state.pop('_annotation_notice',None)
    if notice:
        st.success(notice)
    # On screen: the senses, then the plot, then the options. The options are drawn
    # first in the code, because the plot depends on them.
    senses_area=st.container()
    plot_area=st.container()
    options_area=st.container()
    has_defs=st.session_state.def_space_all is not None
    with options_area:
        st.markdown('**Options**')
        c1,c2=st.columns(2)
        source=c1.radio('Annotate on',['Full embeddings','Definition space'] if has_defs else ['Full embeddings'],horizontal=True,
                        key='annotation_space',
                        format_func={'Full embeddings':'The usage map (full embeddings)','Definition space':'The definitions map (definition space)'}.get,
                        help='**The usage map** (full embeddings): the usages as in Explore, placed by everything the model gives them.\n\n'
                             '**The definitions map** (definition space): the usages placed by their distance to each definition, '
                             'once it is created in Definitions.')
        with c2:
            blind=st.toggle('Blind annotation',key='blind_annotation',
                            help='Draw every usage in the same colour and leave the corpus out of the hover text, so you can annotate without knowing '
                                 'which corpus a usage comes from. Corpus names stay in the legend: click one to hide or show its usages.')
            remaining_only=st.toggle('Show only usages without a sense',key='annotation_remaining_only',
                                     help='Leaves annotated usages out of the plot, so what is left to do stands out. They return when you switch this off.')
        view=current_view(data)
        # Blind annotation hides the corpus. Words, which the sentence shows anyway, keep their colours.
        hide_groups=blind and view.group_by=='corpus'
        if blind:
            st.caption('Corpus colours, hover labels and time views are hidden. LDA is unavailable while blind, because it arranges usages by corpus.'
                       if hide_groups else 'Corpus names and time views are hidden. Words keep their colours, since each sentence shows its word anyway.')
        render_annotation_options()
        render_previous_annotations(data)
    classes=get_sense_classes()
    annotations=get_annotations()
    with senses_area:
        with st.form('add_sense',clear_on_submit=True):
            c1,c2=st.columns([3,1])
            name=c1.text_input('New sense',placeholder='A short label, such as: financial',
                               help='A sense, or any other category you want to label usages with. Each sense becomes a button; add as many as you need.')
            if c2.form_submit_button('Add sense') and name.strip():
                add_sense_class(name.strip())
                set_active_sense_class(name.strip())
                st.rerun()
        active=_sense_buttons(classes,annotations) if classes else ''
        if not classes:
            st.caption('Add a sense, then select usages in the plot.')
        rating=_rating_picker() if _graded() and classes and not _asks_for_rating() else None
        several=with_several_senses(annotations)
        if _asks_for_rating():
            how=('click a point or lasso a group and type the rating; clicking a usage that has the sense also lets you remove it'
                 if _multiple() else 'click a point (or lasso a group) and type the rating')
        else:
            how=('click a point to add the active sense or take it off again, or lasso a group to add it to all of them'
                 if _multiple() else 'click a point (or select a group) to assign it immediately')
        st.caption(f'{len(annotations):,} of {len(data):,} usages annotated'+(f', {several:,} with several senses' if several else '')
                   +f'. Choose a sense above, then {how}.')
    def_labels=None
    vectors=eu.embeddings
    if source=='Definition space':
        vectors=st.session_state.def_space_all
        def_labels=[s.split(': ',1)[-1] for s in st.session_state.definitions_formatted]
    # The plot and the list show the view's usages; annotations are stored by
    # position in the loaded data, so positions are mapped back before saving.
    vdata=view.data
    vann=view.annotations(annotations)
    notes=_hover_notes(vdata,vann)
    group_notes=view.hover_notes(hide_corpus=blind)
    if group_notes:
        notes=[a if not b else (b if not a else f'{a}\n{b}') for a,b in zip(notes,group_notes)]
    with plot_area:
        result=render_plot(vdata,view.embeddings(vectors),key='annotation',definition_labels=def_labels,annotation=True,
                           annotations=vann,sense_classes=classes,blind=blind,blind_groups=hide_groups,
                           omit_indices=sorted(vann) if remaining_only else (),
                           annotation_scale=_scale() if _graded() else None,hover_notes=notes,
                           download_columns=view.download_columns(),noun=view.noun)
        selected=view.to_original(result['clicked_points'])
        if assign_new_selection({**result,'clicked_points':selected},active,rating) and not st.session_state.get('_rating_request'):
            st.rerun()
        if st.session_state.get('_rating_request'):
            _rating_dialog()
        st.caption(f'{len(selected)} usage(s) selected.')
        if _multiple():
            c1,c2,c3=st.columns(3)
            if c1.button('Remove the active sense from selected',disabled=not selected or not active):
                remember_annotations()
                for i in selected:
                    remove_sense(i,active)
                st.rerun()
            if c2.button('Remove all senses from selected',disabled=not selected):
                remember_annotations()
                for i in selected:
                    remove_annotation(i)
                st.rerun()
        else:
            c1,c3=st.columns(2)
            if c1.button('Clear selected',disabled=not selected):
                remember_annotations()
                for i in selected:
                    remove_annotation(i)
                st.rerun()
        if c3.button('Undo last annotation change'):
            undo_annotations()
            st.rerun()
    st.divider()
    render_annotation_list(vdata,vann,classes,hide_groups,view)
    with st.expander('Manage senses and annotations'):
        if classes:
            with st.form('rename_sense',clear_on_submit=True):
                cols=st.columns([2,2,1])
                old=cols[0].selectbox('Sense to rename',classes)
                new=cols[1].text_input('New name',help='Giving it the name of another sense merges the two.')
                if cols[2].form_submit_button('Rename') and new.strip():
                    rename_sense_class(old,new)
                    st.rerun()
            remove=st.selectbox('Sense to remove',classes)
            if st.button('Remove the sense and its annotations'):
                remove_sense_class(remove)
                st.rerun()
        confirm=st.checkbox('Delete all annotations')
        if confirm and st.button('Confirm deletion'):
            clear_annotations()
            st.rerun()
    render_sense_frequencies(vdata,vann,classes,hide_groups,view.noun)
    with st.expander('Review and export annotations'):
        include_all=st.checkbox('Include unannotated usages',value=False)
        df=annotation_table(data,annotations,get_options(),classes,loaded_words(data)[0] if words_active(data) else None)
        if not include_all:
            df=df[df.sense!='']
        standard=['sentence','sense','rating',*[c for c in df.columns if c.startswith(SENSE_COLUMN_PREFIX)],
                  'corpus','word','time','matched_form','start','end','usage_id','sentence_id','lemma']
        standard=[c for c in standard if c in df.columns]
        readable_table(df,text_columns=['sentence'],column_order=standard)
        extra=[c for c in df.columns if c not in standard]
        if extra:
            st.caption('The download also includes the other columns of your usage table: '+', '.join(extra)+'.')
        if _multiple() or _graded():
            st.caption('Several senses are joined by semicolons in the sense column. Each sense:<name> column holds '
                       +('its rating (blank when the usage does not have that sense).' if _graded() else '1 when the usage has that sense and 0 otherwise.'))
        st.download_button('Download annotation CSV',data=df.to_csv(index=False),file_name=f'{data.word}_annotations.csv',mime='text/csv')


STANDARD_COLUMNS=('usage_id','sentence_id','lemma','matched_form','sentence','corpus','word','time','start','end','sense','rating')


def annotation_table(data,annotations,options=None,classes=None,words=None):
    """One row per occurrence, with its senses and any extra columns from the usage table.

    ``sense`` joins a usage's senses with semicolons. Graded annotation of one
    sense per usage adds ``rating``; with several senses or grading there is a
    ``sense:<name>`` column per class (the rating, or 1/0 when ungraded).
    """
    options=options or {}
    graded,multiple=bool(options.get('graded')),bool(options.get('multiple'))
    annotations=normalise_annotations(annotations)
    classes=list(classes) if classes is not None else list(dict.fromkeys(s for label in annotations.values() for s in label))
    rows=[]
    for i,u in enumerate(data.usages):
        label=annotations.get(i,{})
        row={'usage_id':u.usage_id,'sentence_id':u.sentence_id,'lemma':u.word,'matched_form':u.matched_form,
             'sentence':u.sentence,'corpus':u.corpus,'start':u.char_start,'end':u.char_end,'sense':SENSE_SEPARATOR.join(label)}
        if words is not None:
            row['word']=words[i]
        if u.time is not None:
            row['time']=u.time
        if graded and not multiple:
            rating=next(iter(label.values()),None)
            row['rating']='' if rating is None else rating
        if graded or multiple:
            for sense in classes:
                if graded:
                    value=label.get(sense)
                    row[SENSE_COLUMN_PREFIX+sense]='' if value is None else value
                else:
                    row[SENSE_COLUMN_PREFIX+sense]=int(sense in label)
        for name,value in u.metadata.items():
            # A source column that shares a name with one of ours is kept under a prefix.
            row[f'source_{name}' if name in STANDARD_COLUMNS or name.startswith(SENSE_COLUMN_PREFIX) else name]=value
        rows.append(row)
    return pd.DataFrame(rows).fillna('')


def sense_frequencies(data,annotations,classes,order,threshold=1):
    """Counts of each sense in each corpus, in the given sense and corpus order.

    A usage with several senses counts once under each. With ratings, a sense
    counts only from ``threshold`` upwards; unrated senses always count.
    """
    counts=pd.DataFrame(0,index=list(classes),columns=list(order))
    for i,label in normalise_annotations(annotations).items():
        if not 0<=i<len(data):
            continue
        for sense,rating in label.items():
            if sense in counts.index and counts_towards(rating,threshold):
                counts.loc[sense,data.usages[i].corpus]+=1
    return counts


def mean_ratings(data,annotations,classes,order):
    """The mean rating of each sense in each corpus, over the usages rated for it."""
    totals=pd.DataFrame(0.0,index=list(classes),columns=list(order))
    counts=pd.DataFrame(0,index=list(classes),columns=list(order))
    for i,label in normalise_annotations(annotations).items():
        if not 0<=i<len(data):
            continue
        for sense,rating in label.items():
            if sense in totals.index and rating is not None:
                totals.loc[sense,data.usages[i].corpus]+=rating
                counts.loc[sense,data.usages[i].corpus]+=1
    return totals/counts.where(counts>0),counts


def render_sense_frequencies(data,annotations,classes,blind,noun=('corpus','corpora')):
    if not annotations or not classes:
        return
    st.subheader(f'Sense frequencies by {noun[0]}')
    if blind:
        st.caption('Hidden while blind annotation is on, so the table cannot suggest which corpus a usage belongs to. Turn blind annotation off to see it.')
        return
    order=ordered_labels(data)
    graded=_graded()
    threshold=1
    if graded:
        threshold=st.select_slider('Count a sense from rating',options=list(range(1,_scale()+1)),value=1,key='sense_frequency_threshold',
                                   help='Only ratings at or above this count towards the frequencies. At 1, every rated sense counts.')
    counts=sense_frequencies(data,annotations,classes,order,threshold)
    annotated=pd.Series({corpus:0 for corpus in order})
    several=pd.Series({corpus:0 for corpus in order})
    for i,label in annotations.items():
        if 0<=i<len(data):
            annotated[data.usages[i].corpus]+=1
            several[data.usages[i].corpus]+=len(label)>1
    show_several=_multiple() or several.sum()>0
    loaded=pd.Series({corpus:len(data.indices_for_corpus(corpus)) for corpus in order})
    shares=counts.div(annotated.where(annotated>0),axis=1)
    percentages=st.checkbox(f'Show percentages of each {noun[0]}’s annotated usages',value=True,key='sense_frequency_percentages',
                            help=f'Percentages are out of the usages you have annotated in that {noun[0]}, not all of its usages. Counts are shown otherwise.')
    table=(shares*100).round(1) if percentages else counts.copy()
    if show_several:
        table.loc['Usages with several senses']=several
    table.loc['Annotated usages']=annotated
    table.loc['Loaded usages']=loaded
    display=table.rename_axis('Sense').reset_index()
    if percentages:
        for corpus in order:
            display[corpus]=[f'{value:.1f}%' if sense in counts.index and pd.notna(value) else ('—' if pd.isna(value) else f'{int(value):,}')
                             for sense,value in zip(display['Sense'],display[corpus])]
    else:
        display['Total']=list(counts.sum(axis=1))+([int(several.sum())] if show_several else [])+[int(annotated.sum()),int(loaded.sum())]
    readable_table(display)
    notes=[]
    if show_several:
        notes.append('A usage with several senses counts once under each, so a corpus’s senses can add up to more than 100% '
                     '(and a bar in the chart can reach above it).')
    if graded and threshold>1:
        notes.append(f'Senses rated below {threshold} are not counted; unrated senses are.')
    partial=[f'{corpus} ({int(annotated[corpus])} of {int(loaded[corpus])})' for corpus in order if annotated[corpus]<loaded[corpus]]
    if partial:
        notes.append('Not every usage is annotated yet: '+', '.join(partial)+'. Frequencies describe the annotated usages only; '
                     'if you annotated the easy or clustered usages first, they are not yet a fair sample of the corpus.')
    if notes:
        st.caption(' '.join(notes))
    palette=get_palette('Tol Bright (colourblind)')
    totals=st.checkbox('Show totals instead of shares in the chart',value=False,key='sense_frequency_totals',
                       help='Shares: each bar stands for one corpus’s annotated usages, so every bar reaches the same height. '
                            'Totals: bars count usages, so a corpus with more annotated usages has a taller bar.')
    fig=go.Figure()
    for i,sense in enumerate(classes):
        values=(counts.loc[sense] if totals else shares.loc[sense]).tolist()
        fig.add_trace(go.Bar(x=order,y=values,name=sense,marker_color=palette[i%len(palette)],
                             customdata=[[int(counts.loc[sense,corpus]),int(annotated[corpus])] for corpus in order],
                             hovertemplate='%{x}<br>'+sense.replace('%','%%')+': %{customdata[0]} of %{customdata[1]} annotated usages<extra></extra>'))
    fig.update_layout(barmode='stack',height=380,margin=dict(l=60,r=30,t=30,b=60),
                      yaxis_title='Annotated usages' if totals else 'Share of annotated usages',
                      legend=dict(title='Sense',orientation='h',yanchor='bottom',y=1.02,xanchor='left',x=0))
    fig.update_xaxes(type='category',categoryorder='array',categoryarray=order)
    fig.update_yaxes(tickformat=',d' if totals else '.0%')
    plot_chart(fig,width='stretch',key='sense_frequency_chart')
    exported=counts.copy()
    if show_several:
        exported.loc['Usages with several senses']=several
    exported.loc['Annotated usages']=annotated
    exported.loc['Loaded usages']=loaded
    st.download_button('Download sense frequency table (counts)',data=exported.rename_axis('sense').to_csv(),
                       file_name=f'{data.word}_sense_frequencies.csv',mime='text/csv')
    if graded:
        means,rated=mean_ratings(data,annotations,classes,order)
        if rated.values.sum():
            st.markdown(f'**Mean rating by {noun[0]}**')
            shown=means.map(lambda v:'—' if pd.isna(v) else f'{v:.2f}').rename_axis('Sense').reset_index()
            readable_table(shown)
            st.caption(f'The mean rating of each sense, over the usages rated for it in that corpus (1 = barely, {_scale()} = clearly). '
                       '— means no usage in that corpus is rated for the sense.')
            out=means.copy()
            out.columns=[f'{c} mean' for c in out.columns]
            for corpus in order:
                out[f'{corpus} rated']=rated[corpus]
            st.download_button('Download mean ratings (CSV)',data=out.rename_axis('sense').to_csv(),
                               file_name=f'{data.word}_mean_ratings.csv',mime='text/csv',key='mean_ratings_download')


def _read_label(row,sense_columns,graded):
    """The senses in one imported row, from sense:<name> columns or the sense (and rating) columns."""
    if sense_columns:
        label={}
        for column,sense in sense_columns.items():
            value=str(row.get(column,'')).strip()
            if value in ('','0'):
                continue
            label[sense]=parse_rating(value) if graded else None
        return label
    senses=[s.strip() for s in str(row.get('sense','')).split(';') if s.strip()]
    rating=parse_rating(row.get('rating')) if graded and 'rating' in row else None
    return {s:rating for s in senses}


def usage_finder(data):
    """Find a loaded usage from how a file identifies it.

    Tried in turn: the usage ID; the sentence, corpus and target position; the
    sentence and position alone (so annotations survive a change of corpus
    names or periods); the sentence alone, when it holds the word once.
    Anything ambiguous is left unmatched rather than guessed.
    """
    by_id={u.usage_id:i for i,u in enumerate(data.usages)}
    by_position={(u.sentence,u.corpus,u.char_start,u.char_end):i for i,u in enumerate(data.usages)}
    by_span,by_sentence={},{}
    for i,u in enumerate(data.usages):
        by_span.setdefault((u.sentence,u.char_start,u.char_end),[]).append(i)
        by_sentence.setdefault(u.sentence,[]).append(i)

    def find(usage_id=None,sentence=None,corpus=None,start=None,end=None):
        if usage_id and usage_id in by_id:
            return by_id[usage_id]
        if not isinstance(sentence,str) or not sentence:
            return None
        try:
            span=(int(str(start).strip()),int(str(end).strip()))
        except (TypeError,ValueError):
            span=None
        if span:
            if corpus and (sentence,corpus,*span) in by_position:
                return by_position[(sentence,corpus,*span)]
            found=by_span.get((sentence,*span),[])
        else:
            found=by_sentence.get(sentence,[])
        return found[0] if len(found)==1 else None
    return find


def apply_loaded_annotations(data,records,classes=(),options=None,replace=False):
    """Put annotations from a file onto the loaded usages.

    ``records`` holds (identity, label) pairs, the identity being any of
    usage_id, sentence, corpus, start and end. Where a usage already has
    annotations, the file's replace them; with ``replace``, every earlier
    annotation goes. Returns (matched, unmatched, new classes, notes). Options
    the file needs (several senses, ratings) are switched on.
    """
    find=usage_finder(data)
    assignments={}
    unmatched=0
    for identity,label in records:
        if not label:
            continue
        index=find(**identity)
        if index is None:
            unmatched+=1
        else:
            assignments[index]=label
    if not assignments:
        raise ValueError('No row with a sense matches a loaded usage. Annotations can only be loaded onto the same sentences, '
                         'so load the data they were made on first.')
    notes=[]
    if any(len(label)>1 for label in assignments.values()) and not _multiple():
        set_option('multiple',True)
        notes.append('Some usages have several senses, so “Allow several senses per usage” is now on.')
    top=max((r for label in assignments.values() for r in label.values() if r is not None),default=0)
    if top and not _graded():
        set_option('graded',True)
        set_option('scale',min(MAX_SCALE,max(MIN_SCALE,top,int((options or {}).get('scale') or 0))))
        notes.append(f'The annotations have ratings, so graded annotation is now on, with {_scale()} points.')
    elif top>_scale():
        set_option('scale',min(MAX_SCALE,top))
        notes.append(f'The rating scale now has {_scale()} points to fit the file.')
    remember_annotations()
    if replace:
        get_annotations().clear()
    added=[]
    for sense in [*classes,*(s for label in assignments.values() for s in label)]:
        if sense not in get_sense_classes():
            add_sense_class(sense)
            added.append(sense)
    for index,label in assignments.items():
        set_label(index,label)
    return len(assignments),unmatched,added,notes


def import_annotations(data,table,replace=False):
    """Apply senses from an annotation table to the loaded usages.

    Returns (matched rows, unmatched rows, new classes, notes). An imported
    file that needs several senses or ratings switches those options on.
    """
    sense_columns={c:c[len(SENSE_COLUMN_PREFIX):].strip() for c in table.columns if str(c).startswith(SENSE_COLUMN_PREFIX)}
    if 'sense' not in table.columns and not sense_columns:
        raise ValueError('The file needs a column named sense, or sense:<name> columns.')
    if 'usage_id' not in table.columns and 'sentence' not in table.columns:
        raise ValueError('The file needs a usage_id or a sentence column to identify each usage.')
    rows=table.to_dict('records')
    values=[str(r.get(c,'')).strip() for r in rows for c in sense_columns]
    graded=_graded() or ('rating' in table.columns and any(str(r.get('rating','')).strip() for r in rows)) \
        or any(v not in ('','0','1') for v in values)
    records=[({'usage_id':row.get('usage_id'),'sentence':row.get('sentence'),'corpus':row.get('corpus'),
               'start':row.get('start'),'end':row.get('end')},_read_label(row,sense_columns,graded)) for row in rows]
    return apply_loaded_annotations(data,records,replace=replace)


def load_analysis_annotations(data,snapshot,replace=False):
    """Apply the annotations of a saved analysis to the loaded usages."""
    saved=snapshot['data']
    records=[({'usage_id':u.usage_id,'sentence':u.sentence,'corpus':u.corpus,'start':u.char_start,'end':u.char_end},
              snapshot['annotations'].get(i,{})) for i,u in enumerate(saved.usages)]
    return apply_loaded_annotations(data,records,classes=snapshot.get('sense_classes') or (),
                                    options=snapshot.get('annotation_options'),replace=replace)


def _saved_analyses():
    folder=STORAGE/'projects'
    return sorted((p for p in folder.glob('*.semlens') if p.is_file() and not p.is_symlink()),
                  key=lambda p:p.stat().st_mtime,reverse=True)


@st.cache_data(show_spinner=False,max_entries=64)
def _saved_summary(path,modified):
    """Word, usages and annotated usages in a saved analysis, read from its manifest only."""
    try:
        with zipfile.ZipFile(path) as z:
            manifest=json.loads(z.read('project.json'))
        return manifest.get('word',''),len(manifest.get('usages',[])),len(manifest.get('annotations') or {})
    except Exception:
        return None


def _describe_saved(path):
    summary=_saved_summary(str(path),path.stat().st_mtime)
    when=datetime.fromtimestamp(path.stat().st_mtime).strftime('%d %b %Y, %H:%M')
    if summary is None:
        return f'{path.stem} · saved {when}'
    word,usages,annotated=summary
    return f'{path.stem} · “{word}”, {usages:,} usages, {annotated:,} annotated · saved {when}'


def render_previous_annotations(data):
    """Load annotations made earlier: from a saved analysis, an analysis file or an annotation CSV.

    With nothing loaded yet, the whole analysis is opened, since annotations
    need their usages and vectors.
    """
    loaded=data is not None
    with st.expander('Load previous annotations',expanded=not loaded):
        saved=_saved_analyses()
        if loaded:
            st.caption('Annotations saved with an analysis, or downloaded as an annotation CSV, are matched to the usages loaded now: '
                       'by usage ID, or else by sentence and position. Annotations of usages that are not loaded are skipped.')
        else:
            st.caption('Open an analysis you saved before: its usages, vectors, definitions and annotations, without going back to Data & Model. '
                       'An annotation CSV can be loaded once the usages it was made on are loaded.')
        sources=(['A saved analysis on this computer'] if saved else [])+['A file']
        source=st.radio('Load from',sources,horizontal=True,key='previous_annotations_source') if len(sources)>1 else sources[0]
        choice=upload=None
        if source=='A file':
            upload=st.file_uploader('Analysis (.semlens) or annotation table (.csv)' if loaded else 'Analysis (.semlens)',
                                    type=['semlens','csv'] if loaded else ['semlens'],key=f'previous_annotations_file_{int(loaded)}',
                                    help='A .semlens file saved or downloaded from “Save or export this analysis”, or a CSV downloaded from '
                                         '“Review and export annotations” (perhaps edited in a spreadsheet, or made by a colleague on the same data). '
                                         'A CSV needs a sense column (several senses separated by semicolons) or sense:<name> columns, and a usage_id '
                                         'or sentence column; start and end help when a sentence has the word more than once.')
        else:
            choice=st.selectbox('Saved analysis',saved,format_func=_describe_saved,key='previous_annotations_saved')
        replace=False
        if loaded and get_annotations():
            replace=st.checkbox('Replace the annotations made so far',value=False,key='previous_annotations_replace',
                                help='Off: the loaded annotations are added to the ones made so far; a usage annotated in both keeps the loaded ones. '
                                     'On: annotations made so far are removed first. Undo restores them either way.')
        if st.button('Load annotations' if loaded else 'Open analysis',type='primary',disabled=choice is None and upload is None,
                     key='previous_annotations_load'):
            try:
                if not loaded:
                    from app.ui_data import _load_snapshot
                    snapshot=load_project(choice if upload is None else upload)
                    _load_snapshot(snapshot,analysis_path=choice,analysis_name=None if upload is None else Path(upload.name).stem)
                    name=(choice.stem if upload is None else Path(upload.name).stem)
                    st.session_state['_annotation_notice']=(f'Opened “{name}”: {len(snapshot["data"]):,} usages, '
                                                            f'{len(snapshot["annotations"]):,} of them annotated.')
                else:
                    if upload is not None and upload.name.lower().endswith('.csv'):
                        matched,unmatched,added,notes=import_annotations(data,pd.read_csv(upload,dtype=str,keep_default_na=False),replace=replace)
                    else:
                        matched,unmatched,added,notes=load_analysis_annotations(data,load_project(choice if upload is None else upload),replace=replace)
                    st.session_state['_annotation_notice']=(f'Loaded {matched:,} annotations'
                                                            +(f'; {unmatched:,} did not match a loaded usage' if unmatched else '')
                                                            +(f'; new senses: {", ".join(added)}' if added else '')+'. '
                                                            +''.join(n+' ' for n in notes)+'Undo restores the previous state.')
                st.rerun()
            except Exception as exc:
                st.error(str(exc))


def _rating_options():
    return [NO_SENSE]+[str(r) for r in range(1,_scale()+1)]


def render_annotation_list(data,annotations,classes,blind,view=None):
    """Annotate by reading: a concordance with a sense to choose on each line."""
    with st.expander('Annotate in a list',key='annotation_list',on_change='rerun') as panel:
        if not panel.open:
            return
        if not classes:
            st.caption('Add a sense above first.')
            return
        multiple,graded=_multiple(),_graded()
        how=('Tick each sense a usage has' if multiple and not graded else
             'Give a rating under each sense a usage has (— not that sense, ✓ that sense without a rating)' if multiple else
             'Choose a sense and its rating in the first columns' if graded else 'Choose a sense in the first column')
        st.caption(f'Each line is one usage with its target in the middle. {how}; the plot and the frequencies update with it. '
                   'Sort by a neighbouring word to bring recurring phrases together.')
        order=ordered_labels(data)
        cols=st.columns([2,2,2])
        query=cols[0].text_input('Lines containing',key='annotation_list_search',placeholder='For example: burst, hous*',
                                 help='Whole words or phrases, ignoring case. Separate alternatives with commas; end a term with * to match any ending.').strip()
        sorting=cols[1].selectbox('Sort lines by',['Order in the corpus','Word after the target','Word before the target'],key='annotation_list_sort')
        only_open=cols[2].checkbox('Only usages without a sense',key='annotation_list_open')
        corpora=order
        if not blind and len(order)>1:
            plural=view.noun[1] if view is not None else 'corpora'
            corpora=st.multiselect(plural.capitalize(),order,default=order,key=f'annotation_list_{plural}_{st.session_state.data_revision}')
        pattern=usage_search_pattern(query) if query else None
        chosen=[i for i,u in enumerate(data.usages) if u.corpus in corpora and (pattern is None or pattern.search(u.sentence))
                and not (only_open and i in annotations)]
        lines=[(i,)+keyword_in_context(data.usages[i].sentence,data.usages[i].char_span,60) for i in chosen]
        if sorting=='Word after the target':
            lines.sort(key=lambda line:[w.casefold() for w in re.findall(r'\w+',line[3])[:3]])
        elif sorting=='Word before the target':
            lines.sort(key=lambda line:[w.casefold() for w in reversed(re.findall(r'\w+',line[1])[-3:])])
        if not lines:
            st.caption('No usage matches.')
            return
        per_page=100
        page=1
        if len(lines)>per_page:
            page=int(st.number_input('Page',min_value=1,max_value=-(-len(lines)//per_page),value=1,key=f'annotation_list_page_{sorting}_{query}_{only_open}'))
        shown=lines[(page-1)*per_page:page*per_page]
        index=[i for i,*_ in shown]
        columns={}
        config={}
        sense_keys={}
        if multiple:
            for k,sense in enumerate(classes):
                column=f'{SENSE_COLUMN_PREFIX}{k}'
                sense_keys[column]=sense
                if graded:
                    # A sense given before grading was on has no rating yet: ✓.
                    columns[column]=[(UNRATED if annotations[i][sense] is None else str(annotations[i][sense])) if sense in annotations.get(i,{})
                                     else NO_SENSE for i in index]
                    config[column]=st.column_config.SelectboxColumn(sense,options=[NO_SENSE,UNRATED]+_rating_options()[1:],required=True,width='small',
                                                                    help=f'— the usage does not have this sense; ✓ it has it, unrated; 1 (barely) to {_scale()} (clearly).')
                else:
                    columns[column]=[sense in annotations.get(i,{}) for i in index]
                    config[column]=st.column_config.CheckboxColumn(sense,width='small')
        else:
            columns['Sense']=[next(iter(annotations.get(i,{})),NO_SENSE) for i in index]
            config['Sense']=st.column_config.SelectboxColumn('Sense',options=[NO_SENSE]+list(classes),required=True,width='medium')
            if graded:
                columns['Rating']=[NO_SENSE if not annotations.get(i) or next(iter(annotations[i].values())) is None
                                   else str(next(iter(annotations[i].values()))) for i in index]
                config['Rating']=st.column_config.SelectboxColumn('Rating',options=_rating_options(),required=True,width='small')
        columns.update({'Before':[left for _,left,_,_ in shown],'Target':[target for _,_,target,_ in shown],
                        'After':[right for _,_,_,right in shown]})
        config.update({'Before':st.column_config.TextColumn('Before',width='large'),
                       'Target':st.column_config.TextColumn('Target',width='small'),
                       'After':st.column_config.TextColumn('After',width='large')})
        table=pd.DataFrame(columns,index=index)
        if not blind:
            table[(view.noun[0] if view is not None else 'corpus').capitalize()]=[data.usages[i].corpus for i in index]
        editable=[c for c in table.columns if c in sense_keys or c in ('Sense','Rating')]
        # A new editor after every applied change, so an edit is never applied twice.
        version=st.session_state.get('_annotation_list_version',0)
        edited=st.data_editor(table,hide_index=True,width='stretch',key=f'annotation_list_editor_{version}',
                              disabled=[c for c in table.columns if c not in editable],column_config=config)
        changes=list_edits(table,edited,annotations,sense_keys,multiple,graded,None if _asks_for_rating() else get_active_rating())
        if changes:
            remember_annotations()
            for i,label in changes.items():
                set_label(view.original(i) if view is not None else i,label)
            st.session_state['_annotation_list_version']=version+1
            st.rerun()
        st.caption(f'{len(lines):,} usages'+(f', lines {(page-1)*per_page+1:,}–{(page-1)*per_page+len(shown):,}' if len(lines)>per_page else '')
                   +'. Hover a point in the plot to read a whole sentence.')


def list_edits(table,edited,annotations,sense_keys,multiple,graded,active_rating):
    """The new label of every usage changed in the list editor, by usage position.

    ``sense_keys`` maps the editor's per-sense columns to sense names (several
    senses); otherwise the editor has a Sense column and, when graded, a Rating
    column. An empty label means the usage lost all its senses.
    """
    editable=[c for c in table.columns if c in sense_keys or c in ('Sense','Rating')]
    changes={}
    for position,i in enumerate(table.index):
        before=table.iloc[position]
        after=edited.iloc[position]
        if all(before[c]==after[c] for c in editable):
            continue
        if multiple:
            label={}
            for column,sense in sense_keys.items():
                value=after[column]
                if graded:
                    if value==UNRATED:
                        label[sense]=None
                    elif value!=NO_SENSE:
                        label[sense]=int(value)
                elif bool(value):
                    label[sense]=None
            # Keep the order in which the usage's senses were given.
            current=annotations.get(int(i),{})
            label={**{s:label[s] for s in current if s in label},**{s:r for s,r in label.items() if s not in current}}
        else:
            sense=after['Sense']
            if sense==NO_SENSE:
                label={}
            else:
                rating=None
                if graded:
                    if after['Rating']!=NO_SENSE:
                        rating=int(after['Rating'])
                    elif sense!=before['Sense']:
                        # A sense chosen without a rating takes the default rating, if one is in use.
                        rating=active_rating
                label={sense:rating}
        changes[int(i)]=label
    return changes
