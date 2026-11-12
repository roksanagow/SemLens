"""Click-to-assign sense annotation with undo and readable export tables."""
from __future__ import annotations
import re
import pandas as pd
import streamlit as st
from app.annotation_store import (get_annotations,get_sense_classes,get_active_sense_class,add_sense_class,
    set_active_sense_class,remove_sense_class,set_annotation,remove_annotation,clear_annotations,
    remember_annotations,undo_annotations,rename_sense_class)
from semlens.utils import keyword_in_context,usage_search_pattern
import plotly.graph_objects as go
from app.ui_explore import render_plot
from app.common import ordered_labels
from app.presentation import readable_table,plot_chart
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
    st.caption('Define the senses you want to distinguish, then assign usages to them: by clicking or lassoing points in the plot, or line by line in the list below it. '
               'SemLens never assigns a sense for you.')
    eu=st.session_state.embedded_usages
    if eu is None:
        st.info('Load data and embed usages first.')
        return
    data=st.session_state.target_word_data
    classes=get_sense_classes()
    annotations=get_annotations()
    with st.form('add_sense',clear_on_submit=True):
        c1,c2=st.columns([3,1])
        name=c1.text_input('New sense class',placeholder='A short label, such as: financial',
                           help='A sense, or any other category you want to label usages with. Each class becomes a button; add as many as you need.')
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
    blind=st.toggle('Blind annotation',key='blind_annotation',
                    help='Draw every usage in the same colour and leave the corpus out of the hover text, so you can annotate without knowing '
                         'which corpus a usage comes from. Corpus names stay in the legend: click one to hide or show its usages.')
    if blind:
        st.caption('Corpus colours and hover labels are hidden. LDA is unavailable while blind, because it arranges usages by corpus.')
    remaining_only=st.toggle('Show only usages without a sense',key='annotation_remaining_only',
                             help='Leaves annotated usages out of the plot, so what is left to do stands out. They return when you switch this off.')
    def_labels=None
    vectors=eu.embeddings
    if source=='Definition space':
        vectors=st.session_state.def_space_all
        def_labels=[s.split(': ',1)[-1] for s in st.session_state.definitions_formatted]
    result=render_plot(data,vectors,key='annotation',definition_labels=def_labels,annotation=True,
                       annotations=annotations,sense_classes=classes,blind=blind,
                       omit_indices=sorted(annotations) if remaining_only else ())
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
    render_annotation_list(data,annotations,classes,blind)
    with st.expander('Manage classes and annotations'):
        if classes:
            with st.form('rename_sense',clear_on_submit=True):
                cols=st.columns([2,2,1])
                old=cols[0].selectbox('Class to rename',classes)
                new=cols[1].text_input('New name',help='Giving it the name of another class merges the two.')
                if cols[2].form_submit_button('Rename') and new.strip():
                    rename_sense_class(old,new)
                    st.rerun()
            remove=st.selectbox('Class to remove',classes)
            if st.button('Remove class and its annotations'):
                remove_sense_class(remove)
                st.rerun()
        upload=st.file_uploader('Import annotations (CSV)',type=['csv'],key='annotation_import',
                                help='A file downloaded from “Review and export annotations”, perhaps edited in a spreadsheet or made by a colleague on the same data. '
                                     'It needs a sense column, and either usage_id or sentence, corpus, start and end to identify each usage.')
        if upload is not None and st.button('Add these annotations'):
            try:
                matched,unmatched,added=import_annotations(data,pd.read_csv(upload,dtype=str,keep_default_na=False))
                st.session_state['_annotation_notice']=(f'Imported {matched:,} annotations'+(f'; {unmatched:,} rows matched no loaded usage' if unmatched else '')
                                                        +(f'; new classes: {", ".join(added)}' if added else '')+'. Undo restores the previous state.')
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
        notice=st.session_state.pop('_annotation_notice',None)
        if notice:
            st.success(notice)
        confirm=st.checkbox('Delete all annotations')
        if confirm and st.button('Confirm deletion'):
            clear_annotations()
            st.rerun()
    render_sense_frequencies(data,annotations,classes,blind)
    with st.expander('Review and export annotations'):
        include_all=st.checkbox('Include unannotated occurrences',value=False)
        df=annotation_table(data,annotations)
        if not include_all:
            df=df[df.sense!='']
        standard=['sentence','sense','corpus','matched_form','start','end','usage_id','sentence_id','lemma']
        readable_table(df,text_columns=['sentence'],column_order=standard)
        extra=[c for c in df.columns if c not in standard]
        if extra:
            st.caption('The download also includes the other columns of your usage table: '+', '.join(extra)+'.')
        st.download_button('Download annotation CSV',data=df.to_csv(index=False),file_name=f'{data.word}_annotations.csv',mime='text/csv')


STANDARD_COLUMNS=('usage_id','sentence_id','lemma','matched_form','sentence','corpus','start','end','sense')


def annotation_table(data,annotations):
    """One row per occurrence, with its sense and any extra columns from the usage table."""
    rows=[]
    for i,u in enumerate(data.usages):
        row={'usage_id':u.usage_id,'sentence_id':u.sentence_id,'lemma':u.word,'matched_form':u.matched_form,
             'sentence':u.sentence,'corpus':u.corpus,'start':u.char_start,'end':u.char_end,'sense':annotations.get(i,'')}
        for name,value in u.metadata.items():
            # A source column that shares a name with one of ours is kept under a prefix.
            row[f'source_{name}' if name in STANDARD_COLUMNS else name]=value
        rows.append(row)
    return pd.DataFrame(rows).fillna('')


def sense_frequencies(data,annotations,classes,order):
    """Counts of each sense in each corpus, in the given sense and corpus order."""
    counts=pd.DataFrame(0,index=list(classes),columns=list(order))
    for i,sense in annotations.items():
        if sense in counts.index and 0<=i<len(data):
            counts.loc[sense,data.usages[i].corpus]+=1
    return counts


def render_sense_frequencies(data,annotations,classes,blind):
    if not annotations or not classes:
        return
    st.subheader('Sense frequencies by corpus')
    if blind:
        st.caption('Hidden while blind annotation is on, so the table cannot suggest which corpus a usage belongs to. Turn blind annotation off to see it.')
        return
    order=ordered_labels(data)
    counts=sense_frequencies(data,annotations,classes,order)
    annotated=counts.sum(axis=0)
    loaded=pd.Series({corpus:len(data.indices_for_corpus(corpus)) for corpus in order})
    shares=counts.div(annotated.where(annotated>0),axis=1)
    percentages=st.checkbox('Show percentages of each corpus’s annotated usages',value=True,key='sense_frequency_percentages',
                            help='Percentages are out of the usages you have annotated in that corpus, not all of its usages. Counts are shown otherwise.')
    table=(shares*100).round(1) if percentages else counts.copy()
    table.loc['Annotated usages']=annotated
    table.loc['Loaded usages']=loaded
    display=table.rename_axis('Sense').reset_index()
    if percentages:
        for corpus in order:
            display[corpus]=[f'{value:.1f}%' if sense in counts.index and pd.notna(value) else ('—' if pd.isna(value) else f'{int(value):,}')
                             for sense,value in zip(display['Sense'],display[corpus])]
    else:
        display['Total']=list(counts.sum(axis=1))+[int(annotated.sum()),int(loaded.sum())]
    readable_table(display)
    partial=[f'{corpus} ({int(annotated[corpus])} of {int(loaded[corpus])})' for corpus in order if annotated[corpus]<loaded[corpus]]
    if partial:
        st.caption('Not every usage is annotated yet: '+', '.join(partial)+'. Frequencies describe the annotated usages only; '
                   'if you annotated the easy or clustered usages first, they are not yet a fair sample of the corpus.')
    palette=get_palette('Tol Bright (colourblind)')
    fig=go.Figure()
    for i,sense in enumerate(classes):
        values=(shares.loc[sense] if percentages else counts.loc[sense]).tolist()
        fig.add_trace(go.Bar(x=order,y=values,name=sense,marker_color=palette[i%len(palette)],
                             customdata=[[int(counts.loc[sense,corpus]),int(annotated[corpus])] for corpus in order],
                             hovertemplate='%{x}<br>'+sense.replace('%','%%')+': %{customdata[0]} of %{customdata[1]} annotated usages<extra></extra>'))
    fig.update_layout(barmode='stack',height=380,margin=dict(l=60,r=30,t=30,b=60),
                      yaxis_title='Share of annotated usages' if percentages else 'Annotated usages',
                      legend=dict(title='Sense',orientation='h',yanchor='bottom',y=1.02,xanchor='left',x=0))
    fig.update_xaxes(type='category',categoryorder='array',categoryarray=order)
    fig.update_yaxes(tickformat='.0%' if percentages else ',d')
    plot_chart(fig,width='stretch',key='sense_frequency_chart')
    exported=counts.copy()
    exported.loc['Annotated usages']=annotated
    exported.loc['Loaded usages']=loaded
    st.download_button('Download sense frequency table (counts)',data=exported.rename_axis('sense').to_csv(),
                       file_name=f'{data.word}_sense_frequencies.csv',mime='text/csv')


def import_annotations(data,table):
    """Apply senses from a table to the loaded usages; returns (matched, unmatched, new classes)."""
    if 'sense' not in table.columns:
        raise ValueError('The file needs a column named sense.')
    by_id={u.usage_id:i for i,u in enumerate(data.usages)}
    by_position={(u.sentence,u.corpus,str(u.char_start),str(u.char_end)):i for i,u in enumerate(data.usages)}
    has_position=all(column in table.columns for column in ('sentence','corpus','start','end'))
    if 'usage_id' not in table.columns and not has_position:
        raise ValueError('The file needs usage_id, or sentence, corpus, start and end, to identify each usage.')
    assignments={}
    unmatched=0
    for row in table.to_dict('records'):
        sense=str(row['sense']).strip()
        if not sense:
            continue
        index=by_id.get(row.get('usage_id'))
        if index is None and has_position:
            index=by_position.get((row['sentence'],row['corpus'],str(row['start']).strip(),str(row['end']).strip()))
        if index is None:
            unmatched+=1
        else:
            assignments[index]=sense
    if not assignments:
        raise ValueError('No row with a sense matches a loaded usage. Annotations can only be imported onto the same sentences and word positions.')
    remember_annotations()
    added=[]
    for index,sense in assignments.items():
        if sense not in get_sense_classes():
            add_sense_class(sense)
            added.append(sense)
        set_annotation(index,sense)
    return len(assignments),unmatched,added


NO_SENSE='—'


def render_annotation_list(data,annotations,classes,blind):
    """Annotate by reading: a concordance with a sense to choose on each line."""
    with st.expander('Annotate in a list',key='annotation_list',on_change='rerun') as panel:
        if not panel.open:
            return
        if not classes:
            st.caption('Add a sense class above first.')
            return
        st.caption('Each line is one usage with its target in the middle. Choose a sense in the first column; the plot and the frequencies update with it. '
                   'Sort by a neighbouring word to bring recurring phrases together.')
        order=ordered_labels(data)
        cols=st.columns([2,2,2])
        query=cols[0].text_input('Lines containing',key='annotation_list_search',placeholder='For example: burst, hous*',
                                 help='Whole words or phrases, ignoring case. Separate alternatives with commas; end a term with * to match any ending.').strip()
        sorting=cols[1].selectbox('Sort lines by',['Order in the corpus','Word after the target','Word before the target'],key='annotation_list_sort')
        only_open=cols[2].checkbox('Only usages without a sense',key='annotation_list_open')
        corpora=order
        if not blind and len(order)>1:
            corpora=st.multiselect('Corpora',order,default=order,key=f'annotation_list_corpora_{st.session_state.data_revision}')
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
        table=pd.DataFrame({'Sense':[annotations.get(i,NO_SENSE) for i,*_ in shown],
                            'Before':[left for _,left,_,_ in shown],'Target':[target for _,_,target,_ in shown],
                            'After':[right for _,_,_,right in shown]},index=[i for i,*_ in shown])
        if not blind:
            table['Corpus']=[data.usages[i].corpus for i,*_ in shown]
        # A new editor after every applied change, so an edit is never applied twice.
        version=st.session_state.get('_annotation_list_version',0)
        edited=st.data_editor(table,hide_index=True,width='stretch',key=f'annotation_list_editor_{version}',
                              disabled=[c for c in table.columns if c!='Sense'],
                              column_config={'Sense':st.column_config.SelectboxColumn('Sense',options=[NO_SENSE]+list(classes),required=True,width='medium'),
                                             'Before':st.column_config.TextColumn('Before',width='large'),
                                             'Target':st.column_config.TextColumn('Target',width='small'),
                                             'After':st.column_config.TextColumn('After',width='large')})
        changes={int(i):sense for i,sense,before in zip(table.index,edited['Sense'],table['Sense']) if sense!=before}
        if changes:
            remember_annotations()
            for index,sense in changes.items():
                if sense==NO_SENSE:
                    remove_annotation(index)
                else:
                    set_annotation(index,sense)
            st.session_state['_annotation_list_version']=version+1
            st.rerun()
        st.caption(f'{len(lines):,} usages'+(f', lines {(page-1)*per_page+1:,}–{(page-1)*per_page+len(shown):,}' if len(lines)>per_page else '')
                   +'. Hover a point in the plot to read a whole sentence.')
