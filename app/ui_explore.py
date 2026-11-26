"""Shared plots of every corpus, with optional comparisons and usage evidence."""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import html
import textwrap
from sklearn.decomposition import PCA
from semlens.lda import lda_projection,lda_definition_weights,lda_transform_new_points
from semlens.metrics import compute_per_definition_metrics,nearest_counterparts,closest_usages
from semlens.spaces import project_to_definition_space
from semlens.utils import find_usages
from app.components.scatter import render_scatter,render_lda_weights_chart
from app.components.corpus_visibility import hidden_corpora
from app.grouping import current_view,loaded_words,render_grouping_controls,words_active
from app.costs import cost_check
from semlens.resources import comparison_cost
from app.presentation import plot_chart,readable_table,sentence_pair_table,nearest_usage_html
from app.common import (cached_projection,cached_comparison,cached_corpus_scores,cached_within_amd,ordered_labels,number,show_usage,
                        corpus_colours,model_name,layer_name,METRIC_HELP,METRIC_NAMES,METRIC_NOTES,DIRECTIONAL_AMD_FORMULA,PROJECTION_HELP,SAMPLING_HELP,advanced)
from plotly.subplots import make_subplots


@st.cache_data(show_spinner=False,max_entries=24)
def _lda(array,labels,order,features):
    return lda_projection(array,labels,feature_names=features,corpus_order=order)


@st.cache_data(show_spinner=False,max_entries=24)
def _pca_axes(array,anchors=None):
    if len(array)==1:
        return np.zeros((1,2)), None if anchors is None else np.zeros((len(anchors),2))
    n=min(2,*array.shape)
    pca=PCA(n_components=n).fit(array)
    def pad(x):
        return np.pad(x,((0,0),(0,2-x.shape[1])))
    return pad(pca.transform(array)),None if anchors is None else pad(pca.transform(anchors))


SEARCH_HELP=('Rings every usage whose sentence contains the text, so you can see where a collocation or topic sits in the plot. '
             'Whole words or phrases, ignoring case. Separate alternatives with commas; end a term with * to match any ending '
             '(specul* finds speculation and speculative).')


@st.cache_data(show_spinner=False,max_entries=24)
def _pca_variance(array):
    """Share of the total variation shown by PC1 and PC2."""
    if len(array)<2:
        return (0.0,0.0)
    ratios=PCA(n_components=min(2,*array.shape)).fit(array).explained_variance_ratio_.tolist()
    return tuple(ratios+[0.0]*(2-len(ratios)))


def _pca_axis_labels(array,suffix=''):
    return tuple(f'PC{i+1}{suffix} · {share:.0%} of the variation' for i,share in enumerate(_pca_variance(array)))


@st.cache_data(show_spinner=False,max_entries=24)
def _pca_pair(array,components,anchors=None):
    """Coordinates on two chosen principal components (numbered from 1), with their shares of the variation."""
    count=min(max(components),*array.shape)
    pca=PCA(n_components=count).fit(array)
    def pick(points):
        projected=pca.transform(points)
        return np.column_stack([projected[:,c-1] if c<=count else np.zeros(len(points)) for c in components])
    shares=[float(pca.explained_variance_ratio_[c-1]) if c<=count else 0.0 for c in components]
    return pick(array),None if anchors is None else pick(anchors),shares


SAMPLING_MODES={'balanced':'Equal sizes','unequal':'All usages (unequal sizes allowed)','repeated':'Repeated equal-size samples'}


def _copy_setting(widget,target):
    st.session_state[target]=st.session_state[widget]


def _setting(widget_type,label,target,place,**kwargs):
    """A widget for a setting shared by several places: it shows the shared value and writes back to it."""
    widget=f'_{target}_{place}'
    st.session_state[widget]=st.session_state[target]
    return widget_type(label,key=widget,on_change=_copy_setting,args=(widget,target),**kwargs)


def render_score_settings(place):
    """How scores measure distance and sample usages, next to the scores; shown with Advanced options."""
    if not advanced():
        return
    with st.expander('Score settings'):
        c1,c2=st.columns(2)
        with c1:
            _setting(st.selectbox,'Vector distance','distance_metric',place,options=['cosine','euclidean'],
                     help='Used to compare full embedding or definition-space vectors. Individual definition dimensions always use absolute distance.')
        with c2:
            _setting(st.selectbox,'Score sampling','sampling_mode',place,options=list(SAMPLING_MODES),format_func=SAMPLING_MODES.get,help=SAMPLING_HELP)
        c1,c2=st.columns(2)
        with c1:
            _setting(st.number_input,'Sampling seed','sampling_seed',place,min_value=0,
                     help='Fixes which usages are drawn when samples are taken, so that scores can be reproduced. Any whole number will do.')
        if st.session_state.sampling_mode=='repeated':
            with c2:
                _setting(st.number_input,'Repetitions','sampling_repetitions',place,min_value=2,max_value=100,
                         help='How many equal-size samples are drawn for each pair; the scores are their means.')


def _projection_settings(method,key,count,dimensions,distance):
    """Optional settings for the current view, with defaults that suit most data.

    UMAP's and t-SNE's own settings are shown with Advanced options; otherwise their defaults are used.
    """
    settings={}
    if method in {'UMAP','t-SNE'} and not advanced():
        if method=='UMAP':
            settings.update(n_neighbors=max(2,min(15,count-1)),min_dist=0.1,random_state=42)
        else:
            settings.update(perplexity=float(min(30.0,max(1.0,(count-1)//3))),random_state=42)
        settings['metric']=distance
    elif method=='PCA':
        limit=max(2,min(10,count-1,dimensions))
        cols=st.columns(2)
        first=int(cols[0].number_input('Horizontal component',min_value=1,max_value=limit,value=1,key=f'{key}_pc_x',
                                       help='PC1 shows the most variation, PC2 the next most, and so on. Later components can separate usages that the first two leave mixed.'))
        second=int(cols[1].number_input('Vertical component',min_value=1,max_value=limit,value=min(2,limit),key=f'{key}_pc_y'))
        settings['components']=(first,second)
    elif method=='UMAP':
        cols=st.columns(3)
        settings['n_neighbors']=int(cols[0].number_input('Neighbours',min_value=2,max_value=max(2,min(200,count-1)),value=max(2,min(15,count-1)),key=f'{key}_umap_n',
            help='How many nearby usages each usage is laid out with. Small values show fine local groups; large values show the overall shape.'))
        settings['min_dist']=float(cols[1].number_input('Minimum distance',min_value=0.0,max_value=0.99,value=0.1,step=0.05,key=f'{key}_umap_d',
            help='How tightly points may be packed. Smaller values give denser clumps.'))
        settings['random_state']=int(cols[2].number_input('Seed',min_value=0,value=42,key=f'{key}_umap_seed',
            help='UMAP starts from a random layout. The same seed gives the same picture again; another seed shows how much of the picture is chance.'))
        settings['metric']=distance
    elif method=='t-SNE':
        cols=st.columns(2)
        ceiling=max(1.0,float(count-1))
        settings['perplexity']=float(cols[0].number_input('Perplexity',min_value=1.0,max_value=ceiling,value=float(min(30.0,max(1.0,(count-1)//3))),key=f'{key}_tsne_p',
            help='Roughly, how many neighbours each usage attends to. Typical values are 5 to 50; it must be smaller than the number of usages.'))
        settings['random_state']=int(cols[1].number_input('Seed',min_value=0,value=42,key=f'{key}_tsne_seed',
            help='t-SNE starts from a random layout. The same seed gives the same picture again; another seed shows how much of the picture is chance.'))
        settings['metric']=distance
    elif method=='Over time':
        cols=st.columns(2)
        settings['trend']=cols[0].checkbox('Show a moving average',value=True,key=f'{key}_time_trend',
            help='A line through the average vertical position of nearby usages in time (each point averages about a tenth of the usages shown), '
                 'so a drift up or down stands out from the scatter.')
        settings['jitter']=cols[1].checkbox('Spread out usages with the same time',value=True,key=f'{key}_time_jitter',
            help='When many usages share a year, they would sit on one vertical line. This moves each a little sideways, '
                 'by less than half the gap between times, so they can be told apart.')
    if method in {'UMAP','t-SNE'}:
        st.caption(f'Neighbours are found by {distance} distance'+(', the vector distance chosen under Score settings.' if advanced() else '.'))
        if count>20000:
            st.warning(f'{method} on {count:,} usages can take many minutes, and the app waits until it finishes. '
                       'PCA is immediate; a maximum number of usages per corpus in Data & Model also makes this faster.')
    return settings


def _time_text(usage,label):
    """How a usage's time reads in hover text: the original date when there is one."""
    if usage.time is None:
        return None
    original=usage.metadata.get(label) if label else None
    if original:
        # 2020-05-10T02:14:27Z reads as 2020-05-10: the time of day is noise here.
        text=str(original)
        if len(text)>10 and text[4:5]=='-' and text[10:11] in {'T',' '}:
            text=text[:10]
        return f'{label}: {text}'
    value=int(usage.time) if float(usage.time).is_integer() else round(usage.time,2)
    return f'{label or "Time"}: {value}'


def _spread_times(times,seed=0):
    """Small sideways offsets for usages that share a time, smaller than half the gap between times."""
    known=np.unique(times[np.isfinite(times)])
    gap=float(np.min(np.diff(known))) if len(known)>1 else 1.0
    return np.random.default_rng(seed).uniform(-0.3*gap,0.3*gap,size=len(times))


def _moving_average(times,values,shown):
    """Average position over time: the mean of about a tenth of the usages shown, centred on each one."""
    keep=[i for i in shown if np.isfinite(times[i]) and np.isfinite(values[i])]
    if len(keep)<5:
        return None
    order=sorted(keep,key=lambda i:times[i])
    window=max(5,len(order)//10)
    frame=pd.DataFrame({'t':times[order],'y':values[order]})
    smooth=frame.rolling(window,center=True,min_periods=max(3,window//2)).mean().dropna()
    return go.Scatter(x=smooth['t'].tolist(),y=smooth['y'].tolist(),mode='lines',name='Moving average',
                      line=dict(color='#f59e0b',width=3),hoverinfo='skip')


def render_plot(data,embeddings,*,key,definition_labels=None,annotation=False,annotations=None,sense_classes=None,blind=False,
                omit_indices=(),annotation_scale=None,hover_notes=None,blind_groups=None,download_columns=None,noun=('corpus','corpora')):
    """Every point is included; pair selectors never filter this shared plot.

    Corpora switched off in the legend stay off across views. ``blind`` hides
    which corpus a usage belongs to, for annotation.
    """
    order=ordered_labels(data)
    # Blind annotation hides the corpus. When the groups are words, which a
    # reader sees in the sentence anyway, their colours and LDA stay.
    hide_groups=blind if blind_groups is None else blind_groups
    options=['PCA','UMAP','t-SNE']
    # LDA arranges usages by group, which would give hidden groups away.
    if len(order)>1 and not hide_groups:
        options.append('LDA')
    # Time would give the corpora away in blind annotation, like LDA.
    timed=data.has_time and not blind
    if timed:
        options.append('Over time')
    if definition_labels:
        if annotation:
            options.append('Definition axes')
        else:
            options.insert(0,'Definition axes')
    view_key=f'{key}_view'
    if definition_labels and st.session_state.get(view_key)=='Choose axes':
        st.session_state[view_key]='Definition axes'
    if st.session_state.get(view_key) not in options:
        st.session_state.pop(view_key,None)
    help_text=f'All {noun[1]} share one plot. '+PROJECTION_HELP
    if timed:
        help_text+=('\n\n**Over time** puts each usage\'s time on the horizontal axis and a chosen position (PC1, PC2'
                    +(' or the distance to a definition' if definition_labels else '')+') on the vertical one.')
    if definition_labels:
        help_text='**Definition axes** puts a chosen definition on each axis (the distance of every usage to that definition), or combines a definition with PC1/PC2.\n\n'+help_text
    # Plain names first, the method in brackets; the help explains each method.
    names={'PCA':'Main directions of variation (PCA)','UMAP':'Neighbourhoods (UMAP)','t-SNE':'Close neighbours (t-SNE)',
           'LDA':f'What separates the {noun[1]} (LDA)','Definition axes':'Distance to two definitions (definition axes)','Over time':'Over time'}
    method=st.selectbox('Plot view',options,key=view_key,help=help_text,format_func=lambda v:names.get(v,v))
    x=embeddings.detach().cpu().numpy()
    distance='euclidean' if x.shape[1]==1 else st.session_state.distance_metric
    plot_options=st.expander('Plot options')
    with plot_options:
        settings=_projection_settings(method,key,len(x),x.shape[1],distance)
        colour_by_time=False
        if timed and method!='Over time':
            colour_by_time=st.radio('Colour points by',['Corpus','Time'],horizontal=True,key=f'{key}_colour_by',
                                    help='Time shades every usage along one scale, from earliest (dark) to latest (light). '
                                         'Clicking a corpus in the legend still hides and shows it.')=='Time'
        search=''
        if not annotation:
            search=st.text_input('Highlight usages containing',key=f'{key}_search',placeholder='For example: AI, dot-com, specul*',help=SEARCH_HELP).strip()
    anchors=None
    anchor_labels=None
    if definition_labels and not annotation:
        show=st.checkbox('Show definition anchors',key=f'{key}_anchors',
                         help='The definitions projected into the same space as the usages, drawn as stars labelled D1, D2, … in the order of the definition list. Hover a star to read its definition.')
        if show:
            if method in {'PCA','LDA','Definition axes'}:
                defs=st.session_state.definition_embeddings
                anchors=project_to_definition_space(defs,defs).numpy()
                anchor_labels=definition_labels
            else:
                st.caption('Definition anchors are available in PCA, LDA and chosen-axis views.')
    lda_result=None
    extra_traces=[]
    try:
        if method=='Definition axes':
            options_axes=[f'Definition {i+1}: {label}' for i,label in enumerate(definition_labels)]+['PC1','PC2']
            c1,c2=st.columns(2)
            x_choice=c1.selectbox('Horizontal axis',options_axes,index=0,key=f'{key}_x')
            y_choice=c2.selectbox('Vertical axis',options_axes,index=1,key=f'{key}_y')
            pcs,anchor_pcs=_pca_axes(x,anchors) if any(choice in {'PC1','PC2'} for choice in (x_choice,y_choice)) else (None,None)
            def axis(choice,array,pc_array):
                if choice in {'PC1','PC2'}:
                    return pc_array[:,int(choice[-1])-1],_pca_axis_labels(x,' of definition space')[int(choice[-1])-1]
                index=options_axes.index(choice)
                return array[:,index],f'Distance to: {definition_labels[index]}'
            xx,xlabel=axis(x_choice,x,pcs)
            yy,ylabel=axis(y_choice,x,pcs)
            coords=np.column_stack([xx,yy])
            anchor_coords=None if anchors is None else np.column_stack([axis(x_choice,anchors,anchor_pcs)[0],axis(y_choice,anchors,anchor_pcs)[0]])
            axes=(xlabel,ylabel)
            st.caption('Definition axes show cosine distance: smaller values mean closer to that definition. PC axes summarise variation across all definition dimensions.')
        elif method=='Over time':
            times=np.array([np.nan if t is None else float(t) for t in data.times])
            choices=['PC1','PC2']+([f'Definition {i+1}: {label}' for i,label in enumerate(definition_labels)] if definition_labels else [])
            y_choice=st.selectbox('Vertical axis',choices,key=f'{key}_time_y',
                                  help='PC1 and PC2 are the directions along which the usages differ most. A definition puts each usage\'s distance to it on this axis.')
            if y_choice in {'PC1','PC2'}:
                k=int(y_choice[-1])-1
                yy=_pca_axes(x)[0][:,k]
                ylabel=_pca_axis_labels(x,' of definition space' if definition_labels else '')[k]
            else:
                k=choices.index(y_choice)-2
                yy=x[:,k]
                ylabel=f'Distance to: {definition_labels[k]}'
            xx=times+(_spread_times(times) if settings.get('jitter') else 0)
            coords=np.column_stack([xx,yy])
            axes=(data.time_label or 'Time',ylabel)
            anchor_coords=None
            undated=int(np.isnan(times).sum())
            if undated:
                st.caption(f'{undated:,} of {len(data):,} usages have no time and are not shown in this view.')
            if settings.get('trend'):
                hidden=set(hidden_corpora(order))
                shown=[i for i,c in enumerate(data.corpus_list) if c not in hidden and i not in set(omit_indices)]
                trend=_moving_average(times,yy,shown)
                if trend is not None:
                    extra_traces.append(trend)
        elif method=='LDA':
            lda_result=_lda(x,data.corpus_list,order,definition_labels)
            coords=lda_result.coords_2d
            axes=lda_result.axis_labels
            anchor_coords=None if anchors is None else lda_transform_new_points(lda_result,anchors)
            st.caption('LDA uses corpus labels to seek separation. Separation in this view is exploratory, not independent evidence of semantic change.')
            if lda_result.reducer is not None:
                st.caption(f'To keep LDA from separating the corpora by chance, it is fitted on the first {lda_result.reducer.n_components_} principal components '
                           f'of the {lda_result.n_features} dimensions (about one per 20 usages). Fitted on every dimension, it would separate even randomly assigned corpus labels.')
        elif method=='PCA' and settings.get('components',(1,2))!=(1,2):
            coords,anchor_coords,shares=_pca_pair(x,settings['components'],anchors)
            axes=tuple(f'PC{c} · {share:.0%} of the variation' for c,share in zip(settings['components'],shares))
        elif method=='PCA':
            coords,anchor_coords=_pca_axes(x,anchors)
            axes=_pca_axis_labels(x)
        else:
            coords=cached_projection(x,'tsne' if method=='t-SNE' else 'umap',**settings)
            anchor_coords=None
            axes=(f'{method} 1',f'{method} 2')
            if method=='UMAP' and len(x)<4:
                st.caption('Fewer than four usages: showing PCA coordinates instead of UMAP.')
                axes=_pca_axis_labels(x)
    except (ValueError,ImportError,np.linalg.LinAlgError) as exc:
        st.info(f'{exc} Showing PCA instead.')
        coords,anchor_coords=_pca_axes(x,anchors)
        axes=_pca_axis_labels(x)
        lda_result=None
    highlight=None
    nearest=None
    if definition_labels and not annotation:
        nearest_indices=embeddings.argmin(1).tolist()
        nearest=[definition_labels[i] for i in nearest_indices]
        selected=st.selectbox('Highlight nearest definition',['None']+definition_labels,key=f'{key}_highlight',help='The nearest available definition is a model suggestion, not a verified sense annotation.')
        if selected!='None':
            target=definition_labels.index(selected)
            highlight=[i for i,n in enumerate(nearest_indices) if n==target]
    interaction='click'
    highlight_label='Highlighted'
    if annotation:
        interaction=st.radio('Selection',['Click','Lasso'],horizontal=True,key=f'{key}_interaction').lower()
        search=st.text_input('Highlight usages containing',key=f'{key}_search',placeholder='For example: AI, dot-com, specul*',help=SEARCH_HELP).strip()
    if search:
        # A typed search takes the place of the nearest-definition highlight.
        highlight=find_usages(data.sentences,search)
        highlight_label='Contains “'+(search if len(search)<=28 else search[:27].rstrip(' ,')+'…')+'”'
        per_corpus=', '.join(f'{corpus}: {sum(data.corpus_list[i]==corpus for i in highlight)}' for corpus in order) if not hide_groups else ''
        st.caption(f'{len(highlight):,} of {len(data):,} usages contain “{search}”.'+(f' {per_corpus}.' if per_corpus and highlight else ''))
    # Actual coordinates determine the selection widget identity, preventing old selections
    # from being applied after a view/data change while keeping positions stable on annotation.
    notes=list(hover_notes) if hover_notes else [None]*len(data)
    if timed and data.time_label!='corpus':
        for i,u in enumerate(data.usages):
            text=_time_text(u,data.time_label)
            if text:
                notes[i]=text if not notes[i] else f'{notes[i]}\n{text}'
    import hashlib
    signature=hashlib.sha256(coords.tobytes()+str(axes).encode()).hexdigest()[:12]
    result=render_scatter(coords,data.corpus_list,data.sentences,data.word,
        palette_name=st.session_state.palette_name,title=f'{data.word} · {len(order)} {noun[1] if len(order)!=1 else noun[0]}',axis_labels=axes,
        char_spans=data.char_spans,corpus_order=order,definition_labels_per_point=nearest,
        definition_anchor_coords=anchor_coords,definition_anchor_labels=anchor_labels,
        show_definition_anchors=anchor_coords is not None,highlight_indices=highlight,highlight_label=highlight_label,
        annotations=annotations,sense_classes=sense_classes,enable_selection=annotation,interaction_mode=interaction,
        hidden_corpora=hidden_corpora(order),blind=hide_groups,omit_indices=omit_indices,
        annotation_scale=annotation_scale,hover_notes=notes if any(notes) else None,
        colour_values=data.times if colour_by_time else None,colour_title=data.time_label or 'Time',
        extra_traces=extra_traces,
        key=f'{key}_chart_{st.session_state.data_revision}_{signature}_{interaction}_{int(colour_by_time)}')
    if not blind:
        with plot_options:
            table=pd.DataFrame({'usage_id':[u.usage_id for u in data.usages],noun[0]:data.corpus_list,
                                **(download_columns or {}),
                                'matched_form':[u.matched_form for u in data.usages],'sentence':data.sentences,
                                'start':[u.char_start for u in data.usages],'end':[u.char_end for u in data.usages],
                                'x':coords[:,0],'y':coords[:,1],'x_axis':axes[0],'y_axis':axes[1]})
            if data.has_time:
                table.insert(2,'time',data.times)
            if annotations is not None:
                table['sense']=['; '.join(annotations.get(i,{})) for i in range(len(data))]
            st.download_button('Download plot data (CSV)',data=table.to_csv(index=False),file_name=f'{data.word}_plot_{method.lower().replace(" ","_")}.csv',
                               mime='text/csv',key=f'{key}_plot_download',
                               help='One row per usage with its sentence, corpus and position in this plot, for drawing the figure in other software.')
    if lda_result is not None and definition_labels and not annotation:
        with st.expander('LDA definition weights'):
            render_lda_weights_chart(lda_definition_weights(lda_result,definition_labels),palette_name=st.session_state.palette_name)
            st.caption('Weights act on distance features: positive means greater distance increases LD1. Read them alongside corpus positions and mean distances; they are not sense frequencies.')
    return result


AMD_IN_WORDS=('For each usage, the distance to the closest usage of the other {0}; averaged both ways. '
              'Higher means more usages without a close counterpart in the other {0}.')


def _directions_in_words(summary,a,b,noun):
    """Which way the counterparts are weaker, in a sentence."""
    forward,backward=float(summary.loc[f'amd_{a}_to_{b}','mean']),float(summary.loc[f'amd_{b}_to_{a}','mean'])
    if not (np.isfinite(forward) and np.isfinite(backward)):
        return ''
    if abs(forward-backward)<=0.05*max(forward,backward,1e-12):
        return (f'Usages of **{a}** and of **{b}** find about equally close counterparts in the other {noun[0]} '
                f'({number(forward)} and {number(backward)}).')
    weaker,stronger=(a,b) if forward>backward else (b,a)
    return (f'Usages of **{weaker}** are further from their closest counterparts in {stronger} ({number(max(forward,backward))}) '
            f'than the other way round ({number(min(forward,backward))}): {weaker} may have usages unlike any in {stronger}, or more varied ones '
            f'(see **Compared with AMD within each {noun[0]}**). Read the usages under **Inspect the usage pairs behind AMD**.')


WITHIN_IN_WORDS=('Within a {0}, each usage’s distance to its closest *other* usage of the same {0} is averaged, '
                 'on the same sampled usages as the score. Near 1×, usages find counterparts in the other {0} about as close as in their own; '
                 'well above 1×, the two differ by more than usages vary within each.')


def _ratio_in_words(amd,baseline,noun):
    if not (np.isfinite(amd) and np.isfinite(baseline)) or baseline<=0:
        return f'AMD cannot be compared with AMD within each {noun[0]}: a {noun[0]} needs two usages that are not identical.'
    return f'AMD is **{amd/baseline:.2f}×** the AMD within each {noun[0]}.'


def _within_ratios(data,embeddings,labels,shown,matrices):
    """For each pair shown: AMD between the two, divided by the mean AMD within each, on the same sampled usages."""
    array=embeddings.detach().cpu().numpy()
    n=len(shown)
    ratios=np.full((n,n),np.nan)
    for a in range(n):
        for b in range(a+1,n):
            i,j=shown[a],shown[b]
            _,_,baseline=cached_within_amd(array[data.indices_for_corpus(labels[i])],array[data.indices_for_corpus(labels[j])],
                                           (labels[i],labels[j]),st.session_state.distance_metric,st.session_state.sampling_mode,
                                           st.session_state.sampling_repetitions,st.session_state.sampling_seed)
            amd=float(matrices['amd']['mean'][i,j])
            if baseline>0 and np.isfinite(baseline):
                ratios[a,b]=ratios[b,a]=amd/baseline
    return ratios


def ratio_heatmap(ratios,labels,noun):
    n=len(labels)
    ticks=['<br>'.join(html.escape(line) for line in textwrap.wrap(textwrap.shorten(label,width=32,placeholder='…'),16)) for label in labels]
    text=[['' if i==j or not np.isfinite(ratios[i,j]) else f'{ratios[i,j]:.2f}×' for j in range(n)] for i in range(n)]
    custom=[[[html.escape(labels[i]),html.escape(labels[j])] for j in range(n)] for i in range(n)]
    fig=go.Figure(go.Heatmap(z=ratios,x=list(range(n)),y=list(range(n)),text=text,texttemplate='%{text}',customdata=custom,
                            colorscale='Viridis',hoverongaps=False,colorbar=dict(title='×',thickness=12),
                            hovertemplate='%{customdata[0]} ↔ %{customdata[1]}<br>AMD is %{z:.2f}× the AMD within each '+noun[0]+'<extra></extra>'))
    fig.update_layout(title=f'AMD ÷ AMD within each {noun[0]}',height=max(370,32*n+170),margin=dict(l=95,r=50,t=55,b=100),
                      xaxis=dict(tickmode='array',tickvals=list(range(n)),ticktext=ticks,tickangle=-35),
                      yaxis=dict(tickmode='array',tickvals=list(range(n)),ticktext=ticks,autorange='reversed'))
    return fig


def _pairs_in_words(matrix,visible,noun):
    """'2020–21 and 1999–2003 differ most (AMD 0.43); … least (0.31).'"""
    pairs=[(float(matrix[i,j]),visible[i],visible[j]) for i in range(len(visible)) for j in range(i+1,len(visible)) if np.isfinite(matrix[i,j])]
    if not pairs:
        return ''
    if len(pairs)==1:
        value,first,second=pairs[0]
        return f'AMD between **{first}** and **{second}**: {number(value)}.'
    high,low=max(pairs),min(pairs)
    return (f'Of the {noun[1]} shown, **{high[1]}** and **{high[2]}** differ most (AMD {number(high[0])}); '
            f'**{low[1]}** and **{low[2]}** differ least ({number(low[0])}).')


def render_comparison(data,embeddings,*,key,definition_labels=None,noun=('corpus','corpora')):
    labels=ordered_labels(data)
    if len(labels)<2:
        st.info(f'One {noun[0]} is shown. You can explore and annotate its usages; change scores need a second {noun[0]}.')
        return
    # Words and corpora keep their own choices. A browser sends a widget's last
    # value back, so a choice that is no longer among the groups (filtered out,
    # say) cannot be cleared: the selectors are given fresh keys instead.
    base=key if noun[0]=='corpus' else f'{key}_words'
    generation=st.session_state.get(f'_{base}_generation',0)
    key=base if not generation else f'{base}_g{generation}'
    if any(st.session_state.get(f'{key}_{side}') not in (None,*labels) for side in ('from','to','pairs_from','pairs_to')):
        generation+=1
        st.session_state[f'_{base}_generation']=generation
        key=f'{base}_g{generation}'
    st.subheader(f'Compare {noun[1]}')
    render_score_settings(key)
    st.caption(f'The plot includes all {noun[1]}. Select the two {noun[1]} whose scores and counterparts you want to inspect. '
               f'A higher score means the usages of the two {noun[1]} are further apart; what counts as high depends on the model and the word, '
               'so compare scores with each other and read the sentences behind them.')
    cols=st.columns(2)
    a=cols[0].selectbox(f'From {noun[0]}',labels,key=f'{key}_from')
    b=cols[1].selectbox(f'To {noun[0]}',[c for c in labels if c!=a],key=f'{key}_to')
    ia,ib=data.indices_for_corpus(a),data.indices_for_corpus(b)
    # Checked at the sizes the sidebar's sampling compares (the pair inspector reads the same samples).
    if not comparison_allowed([(len(ia),len(ib))],f'{key}_pair',f'Comparing {a} and {b}'):
        return
    xa,xb=embeddings[ia],embeddings[ib]
    distance='absolute' if definition_labels and len(definition_labels)==1 else st.session_state.distance_metric
    with st.spinner('Comparing usages…'):
        comparison=cached_comparison(xa.numpy(),xb.numpy(),(a,b),distance,
                    st.session_state.sampling_mode,st.session_state.sampling_repetitions,st.session_state.sampling_seed)
    summary=comparison.summary
    na,nb=map(len,comparison.draws[0])
    st.caption(f'Scores use {na:,} / {len(ia):,} usages from {a} and {nb:,} / {len(ib):,} from {b}; '
               f'{len(comparison.draws)} sample(s). Distances: {distance}. All retained usages remain in the plot.')
    if distance=='absolute':
        st.caption('With one definition the representation is a scalar, so scores use absolute distance.')
    if st.session_state.sampling_mode=='unequal' and na!=nb:
        st.caption(f'SAMD matches {min(na,nb):,} pairs; {abs(na-nb):,} surplus usages remain unmatched.')
    # AMD first, with its two directions; the other scores on request.
    cols=st.columns(3)
    cols[0].metric('AMD',number(summary.loc['amd','mean']),help=METRIC_HELP['amd'])
    for col,(source,target) in zip(cols[1:],((a,b),(b,a))):
        metric=f'amd_{source}_to_{target}'
        col.metric(f'AMD: {source} → {target}',number(summary.loc[metric,'mean']),
                   help=f'**Directional average minimum distance**\n\n{DIRECTIONAL_AMD_FORMULA}, with $A$ = {source} and $B$ = {target}\n\n'
                        f'For each usage in {source}, the distance to its closest counterpart in {target}, averaged. Higher means weaker counterparts. '
                        'Corpus composition and sample size affect this score.')
    st.markdown(_directions_in_words(summary,a,b,noun))
    with st.expander(f'Compared with AMD within each {noun[0]}',key=f'{key}_within',on_change='rerun') as panel:
        if panel.open:
            with st.spinner(f'Comparing usages within each {noun[0]}…'):
                in_a,in_b,baseline=cached_within_amd(xa.numpy(),xb.numpy(),(a,b),distance,st.session_state.sampling_mode,
                                                     st.session_state.sampling_repetitions,st.session_state.sampling_seed)
            st.markdown(_ratio_in_words(float(summary.loc['amd','mean']),baseline,noun))
            st.caption(f'AMD within {a}: {number(in_a)} · within {b}: {number(in_b)} · their average: {number(baseline)}. '
                       +WITHIN_IN_WORDS.format(noun[0]))
    with st.expander('More scores: APD, PRT and SAMD'):
        cols=st.columns(3)
        for col,metric in zip(cols,('apd','prt','samd')):
            col.metric(metric.upper(),number(summary.loc[metric,'mean']),help=METRIC_HELP[metric])
            col.caption(METRIC_NOTES[metric])
    if st.session_state.chronological:
        earlier,later=sorted([a,b],key=labels.index)
        st.caption(f'Chronology: {earlier} precedes {later}. Inspect weak counterparts for evidence of possible loss or emergence.')
    if len(comparison.draws)>1:
        readable_table(summary.rename_axis('Metric').reset_index())
        st.caption('Min–max describes variation across the sampled comparisons, not a confidence interval. Equal-size corpora produce identical repetitions.')
    space_name='definitions map (definition space)' if definition_labels else 'usage map (full embeddings)'
    with st.expander(f'Inspect the usage pairs behind AMD — {space_name}',key=f'{key}_pairs',on_change='rerun') as panel:
        if panel.open:
            views=['Weakest counterparts between two corpora','Closest usages to a chosen sentence']
            view=st.radio('Show',views,horizontal=True,key=f'{key}_pairs_view',label_visibility='collapsed')
            if view==views[0]:
                _render_weakest_counterparts(data,embeddings,comparison,labels,(a,b),(ia,ib),distance,space_name,key)
            else:
                _render_closest_usages(data,embeddings,labels,distance,space_name,key)
    if definition_labels:
        with st.expander('Per-definition scores (absolute distance)',key=f'{key}_perdef',on_change='rerun') as panel:
            if panel.open:
                frames=[compute_per_definition_metrics(xa[i],xb[j],definition_labels,(a,b)).set_index('dim') for i,j in comparison.draws]
                numeric=[c for c in frames[0].columns if c!='definition']
                means=sum(f[numeric] for f in frames)/len(frames)
                means.insert(0,'definition',frames[0]['definition'])
                names={'definition':'Definition','apd':'APD','prt':'PRT','amd':'AMD','samd':'SAMD',
                       f'amd_{a}_to_{b}':f'AMD: {a} → {b}',f'amd_{b}_to_{a}':f'AMD: {b} → {a}',
                       f'avg_cos_dist_{a}':f'Mean distance: {a}',f'avg_cos_dist_{b}':f'Mean distance: {b}',
                       'mean_distance_difference':f'Mean difference ({b} − {a})'}
                table=means.sort_values('amd',ascending=False)[list(names)].rename(columns=names)
                readable_table(table,text_columns=['Definition'],max_text_width=300)
                st.caption('The definition column stays in place while the scores scroll; double-click a definition to read it in full. '
                           'Means use the same samples as the scores above; a smaller mean distance means closer to the definition.')
                st.caption('Each definition gives one number per usage (its cosine distance to that definition), so these scores always compare '
                           'usages by the absolute difference |x − y|. The vector distance under Score settings cannot change that: in one dimension '
                           'Euclidean distance is the absolute difference, and cosine distance between two positive numbers is always zero. '
                           'APD is the mean difference over all cross-corpus pairs; PRT is the difference between the two corpus means.')


def _set_pair(key,changed):
    """Choosing a pair inside the inspector moves the comparison above with it."""
    other='to' if changed=='from' else 'from'
    new=st.session_state[f'{key}_pairs_{changed}']
    previous=st.session_state.get(f'{key}_{changed}')
    if new==st.session_state.get(f'{key}_{other}'):
        st.session_state[f'{key}_{other}']=previous  # picking the other side's corpus swaps the pair
    st.session_state[f'{key}_{changed}']=new


def _render_weakest_counterparts(data,embeddings,comparison,labels,pair,indices,distance,space_name,key):
    a,b=pair
    ia,ib=indices
    if len(labels)>2:
        # Mirror the selectors above, so the pair can be changed without leaving this panel.
        st.session_state[f'{key}_pairs_from'],st.session_state[f'{key}_pairs_to']=a,b
        cols=st.columns(2)
        cols[0].selectbox('Inspect usages from',labels,key=f'{key}_pairs_from',on_change=_set_pair,args=(key,'from'))
        cols[1].selectbox('Closest counterparts in',[c for c in labels if c!=a],key=f'{key}_pairs_to',on_change=_set_pair,args=(key,'to'))
    st.caption(f'Ranked by weakest nearest counterpart in {space_name}. Search either sentence. These are the usages behind the scores above; another space can give different counterparts.')
    repetition=0
    if len(comparison.draws)>1:
        repetition=int(st.number_input('Sample to inspect',min_value=1,max_value=len(comparison.draws),value=1,key=f'{key}_sample'))-1
    da,db=comparison.draws[repetition]
    aa,bb=embeddings[np.asarray(ia)[da]],embeddings[np.asarray(ib)[db]]
    ab,ba=nearest_counterparts(aa,bb,distance)
    query=st.text_input('Find a sentence',placeholder='Type or paste part of a sentence…',key=f'{key}_pair_search').strip()
    for title,t,src,dst in ((f'{a} → {b}',ab,np.asarray(ia)[da],np.asarray(ib)[db]),(f'{b} → {a}',ba,np.asarray(ib)[db],np.asarray(ia)[da])):
        st.write(title)
        rows=[]
        for r in t.itertuples():
            usage,counterpart=data.usages[int(src[int(r.source_index)])],data.usages[int(dst[int(r.counterpart_index)])]
            rows.append({'Distance':r.distance,'Usage':usage.sentence,'Closest counterpart':counterpart.sentence,
                         'Usage span':usage.char_span,'Counterpart span':counterpart.char_span})
        table=pd.DataFrame(rows).sort_values('Distance',ascending=False,kind='stable')
        if query:
            table=table[table['Usage'].str.contains(query,case=False,regex=False)|table['Closest counterpart'].str.contains(query,case=False,regex=False)]
        sentence_pair_table(table,key=f'{key}_pairs_page_{title}_{repetition}_{query}',compact=True)
        st.caption(f'{len(table):,} of {len(rows):,} pairs, weakest first · scroll down for more, and inside either sentence cell to read it in full.')


def _usage_label(usage,width=70):
    """One line for a selector: the corpus, then the target with context on both sides."""
    left,right=usage.sentence[:usage.char_start],usage.sentence[usage.char_end:]
    left=('…'+left[-width:]) if len(left)>width else left
    right=(right[:width]+'…') if len(right)>width else right
    return f'{usage.corpus} · {left}{usage.matched_form}{right}'


def _render_closest_usages(data,embeddings,labels,distance,space_name,key):
    st.caption(f'Choose one usage to see the usages nearest to it in each of the other corpora, in {space_name}.')
    query=st.text_input('Find the sentence to look up',placeholder='Type or paste part of a sentence…',key=f'{key}_nearest_search').strip()
    matches=[i for i,u in enumerate(data.usages) if query.casefold() in u.sentence.casefold()]
    if not matches:
        st.caption('No loaded sentence contains that text.')
        return
    limit=300
    cols=st.columns([4,1])
    chosen=cols[0].selectbox('Sentence',matches[:limit],format_func=lambda i:_usage_label(data.usages[i]),
                             key=f'{key}_nearest_choice_{st.session_state.data_revision}',
                             help='Type in the box to narrow the list. The target word is shown with its surrounding context.')
    count=int(cols[1].number_input('Closest per corpus',min_value=1,max_value=20,value=3,key=f'{key}_nearest_count'))
    if len(matches)>limit:
        st.caption(f'Listing the first {limit:,} of {len(matches):,} matching usages. Add more of the sentence above to narrow them.')
    usage=data.usages[chosen]
    show_usage(usage,prefix='Chosen usage · ')
    rows=[]
    for corpus in labels:
        if corpus==usage.corpus:
            continue
        candidates=np.asarray(data.indices_for_corpus(corpus))
        positions,distances=closest_usages(embeddings[chosen],embeddings[candidates],count,distance)
        for p,d in zip(positions,distances):
            neighbour=data.usages[int(candidates[int(p)])]
            rows.append((corpus,float(d),neighbour.sentence,neighbour.char_span))
    st.html(nearest_usage_html(pd.DataFrame(rows,columns=['Corpus','Distance','Closest usage','Span'])))
    st.caption(f'Nearest first within each corpus · {distance} distance in {space_name}. Every loaded usage is searched, '
               'including any left out of an equal-size score sample.')


def render_definition_distribution(data,embeddings,definition_labels):
    with st.expander('Definition distributions and descriptive statistics',key='definition_distribution',on_change='rerun') as panel:
        if not panel.open:
            return
        index=st.selectbox('Definition',list(range(len(definition_labels))),format_func=lambda i:definition_labels[i])
        normalised=st.checkbox('Normalise within each corpus',value=True,help='Show proportions instead of counts, so differently sized corpora can be compared.')
        order=ordered_labels(data)
        colours=corpus_colours(order,st.session_state.palette_name)
        fig=go.Figure()
        rows=[]
        # Bins span the distances actually observed, shared by every corpus, with
        # enough usages per bin for the smallest corpus to give a readable line.
        column=embeddings[:,index].numpy()
        low,high=float(column.min()),float(column.max())
        if high-low<1e-9:
            low,high=max(0.0,low-.05),low+.05
        smallest=min(len(data.indices_for_corpus(corpus)) for corpus in order)
        bins=int(np.clip(round(2*np.sqrt(smallest)),8,30))
        edges=np.linspace(low,high,bins+1)
        centres=(edges[:-1]+edges[1:])/2
        dashes=['solid','dash','dot','dashdot','longdash','longdashdot']
        for position,corpus in enumerate(order):
            values=embeddings[data.indices_for_corpus(corpus),index].numpy()
            counts,_=np.histogram(values,bins=edges)
            heights=counts/len(values) if normalised else counts
            custom=[[float(left),float(right),int(count),len(values)] for left,right,count in zip(edges[:-1],edges[1:],counts)]
            fig.add_trace(go.Scatter(x=centres.tolist(),y=heights.tolist(),name=corpus,
                mode='lines+markers',fill='none',
                line=dict(color=colours[corpus],width=2.5,dash=dashes[position%len(dashes)]),
                marker=dict(color=colours[corpus],size=4),customdata=custom,
                hovertemplate='Distance bin: %{customdata[0]:.3f}–%{customdata[1]:.3f}'
                    '<br>Usages: %{customdata[2]} / %{customdata[3]}'
                    +('<br>Proportion: %{y:.1%}' if normalised else '')+'<extra>%{fullData.name}</extra>'))
            rows.append({'Corpus':corpus,'Usages':len(values),'Mean distance':float(values.mean()),
                         'Median distance':float(np.median(values)),'Standard deviation':float(values.std())})
        fig.update_layout(xaxis_title='Cosine distance to definition (smaller = closer)',
                          yaxis_title='Proportion of usages' if normalised else 'Usages',
                          hovermode='x unified',height=440,
                          legend=dict(title='Corpus',orientation='h',yanchor='bottom',y=1.02,xanchor='left',x=0))
        margin=(high-low)*.04
        fig.update_xaxes(range=[low-margin,high+margin])
        fig.update_yaxes(rangemode='tozero',tickformat='.0%' if normalised else ',d')
        plot_chart(fig,width='stretch')
        st.caption(f'Each line connects the values in {bins} shared distance bins spanning the observed distances ({low:.3f}–{high:.3f}), with no smoothing. '
                   'Click a corpus in the legend to hide or show it; double-click to isolate it.')
        readable_table(pd.DataFrame(rows))
        st.caption('These describe distances to this definition; they do not estimate the proportion of usages that belong to a sense.')
        distances=pd.DataFrame({'usage_id':[u.usage_id for u in data.usages],'corpus':data.corpus_list,
                                'matched_form':[u.matched_form for u in data.usages],'sentence':data.sentences})
        for position,label in enumerate(definition_labels):
            distances[f'D{position+1}: {label}']=embeddings[:,position].numpy()
        st.download_button('Download every usage’s distance to every definition (CSV)',data=distances.to_csv(index=False),
                           file_name=f'{data.word}_definition_distances.csv',mime='text/csv',key='definition_distances_download',
                           help='One row per usage and one column per definition, for statistics or figures in other software.')


def corpus_score_heatmap(stats,counts,labels,metric):
    n=len(labels)
    ticks=['<br>'.join(html.escape(line) for line in textwrap.wrap(textwrap.shorten(label,width=32,placeholder='…'),16)) for label in labels]
    custom=[[[html.escape(labels[i]),html.escape(labels[j]),number(stats['mean'][i,j]),
              number(stats['min'][i,j]),number(stats['max'][i,j]),int(counts[i,j]),int(counts[j,i])]
             for j in range(n)] for i in range(n)]
    # Fewer digits and smaller type as the grid grows, so every cell keeps its number.
    digits,size=(3,12) if n<=6 else (2,11) if n<=8 else (2,9)
    text=[['' if i==j else (f'{stats["mean"][i,j]:.{digits}g}' if np.isfinite(stats['mean'][i,j]) else 'n/a')
           for j in range(n)] for i in range(n)]
    fig=go.Figure(go.Heatmap(z=stats['mean'],x=list(range(n)),y=list(range(n)),customdata=custom,
                            colorscale='Viridis',text=text,texttemplate='%{text}',textfont=dict(size=size),
                            hoverongaps=False,colorbar=dict(title=metric.upper(),thickness=12),
                            hovertemplate='%{customdata[0]} ↔ %{customdata[1]}<br>'+metric.upper()+': %{customdata[2]}'
                            '<br>Sample min–max: %{customdata[3]}–%{customdata[4]}'
                            '<br>Usages per sample: %{customdata[5]} / %{customdata[6]}<extra></extra>'))
    fig.update_layout(title=f'{METRIC_NAMES[metric]} ({metric.upper()})',height=max(370,32*n+170),margin=dict(l=95,r=50,t=55,b=100),
                      xaxis=dict(tickmode='array',tickvals=list(range(n)),ticktext=ticks,tickangle=-35),
                      yaxis=dict(tickmode='array',tickvals=list(range(n)),ticktext=ticks,autorange='reversed'))
    return fig


def _pair_sizes(na,nb):
    """The sizes compared for two groups under the score sampling."""
    if st.session_state.sampling_mode=='unequal':
        return na,nb
    smaller=min(na,nb)
    return smaller,smaller


def comparison_allowed(sizes,key,action):
    """Hold back comparisons whose distance matrices could exhaust memory."""
    cost=comparison_cost([_pair_sizes(a,b) for a,b in sizes],action)
    if cost.memory_bytes<5e8:
        return True
    return cost_check(cost,key,remedy='Set a maximum number of usages per corpus in Data & Model, '
                                       'or choose Equal sizes under Score settings (shown with Advanced options in the sidebar).')


def _all_pair_scores(data,embeddings,noun,key):
    """The four scores for every pair of corpora shown in the plot legend, or None (with a note) when they cannot be computed."""
    labels=ordered_labels(data)
    # Corpora switched off in the plot legend are left out here too.
    hidden=set(hidden_corpora(labels))
    shown=[i for i,label in enumerate(labels) if label not in hidden]
    if hidden:
        st.caption(f'Showing {len(shown)} of {len(labels)} {noun[1]}. Switched off in the plot legend (Map): {", ".join(l for l in labels if l in hidden)}. '
                   'Click a name in the legend to bring it back.')
    if len(shown)<2:
        st.info(f'Switch on at least two {noun[1]} in the plot legend (Map) to compare them.')
        return None
    indices=tuple(tuple(data.indices_for_corpus(label)) for label in labels)
    if not comparison_allowed([(len(indices[i]),len(indices[j])) for a,i in enumerate(shown) for j in shown[a+1:]],
                              key,f'Comparing every pair of {noun[1]}'):
        return None
    with st.spinner(f'Comparing all pairs of {noun[1]}…'):
        matrices,counts=cached_corpus_scores(embeddings.detach().cpu().numpy(),indices,tuple(labels),st.session_state.distance_metric,
                                             st.session_state.sampling_mode,st.session_state.sampling_repetitions,st.session_state.sampling_seed)
    return labels,shown,matrices,counts


def render_corpus_scores(data,embeddings,noun=('corpus','corpora')):
    labels=ordered_labels(data)
    if len(labels)<2:
        st.info(f'One {noun[0]} is shown: scores compare two or more.')
        return
    st.subheader(f'Scores for “{data.word}” across all {noun[1]}')
    render_score_settings('all_pairs')
    mode=st.session_state.sampling_mode
    distance=st.session_state.distance_metric
    descriptions={'balanced':'equal-size samples for each pair','unequal':'all loaded usages',
                  'repeated':f'means over {st.session_state.sampling_repetitions} equal-size samples per pair'}
    st.caption(f'{distance.capitalize()} distance · {descriptions[mode]}. Scores use the full usage embeddings, before 2D projection. Hover a cell for its value and sample size.')
    found=_all_pair_scores(data,embeddings,noun,'all_pairs')
    if found is None:
        return
    labels,shown,matrices,counts=found
    pick=np.ix_(shown,shown)
    visible=[labels[i] for i in shown]
    # AMD is the headline: how far apart the corpora are, in a sentence and a heatmap.
    st.markdown(f'**Average minimum distance (AMD)**',help=METRIC_HELP['amd'])
    st.markdown(_pairs_in_words(matrices['amd']['mean'][pick],visible,noun))
    if len(shown)>2:
        stats={stat:matrix[pick] for stat,matrix in matrices['amd'].items()}
        plot_chart(corpus_score_heatmap(stats,counts[pick],visible,'amd'),width='stretch',key='all_pairs_amd')
    st.caption(AMD_IN_WORDS.format(noun[0])+(f' The diagonal is left out: within-{noun[0]} comparisons are not change scores. '
                                             f'Sample sizes can differ between pairs of {noun[1]}.' if len(shown)>2 else ''))
    with st.expander(f'Compared with AMD within each {noun[0]}',key='within_scores',on_change='rerun') as panel:
        if panel.open:
            with st.spinner(f'Comparing usages within each {noun[0]}…'):
                ratios=_within_ratios(data,embeddings,labels,shown,matrices)
            if len(shown)==2:
                amd=float(matrices['amd']['mean'][shown[0],shown[1]])
                st.markdown(_ratio_in_words(amd,amd/ratios[0,1] if np.isfinite(ratios[0,1]) else float('nan'),noun))
            else:
                plot_chart(ratio_heatmap(ratios,visible,noun),width='stretch',key='all_pairs_within')
            st.caption(WITHIN_IN_WORDS.format(noun[0]))
    with st.expander('More scores: APD, PRT and SAMD',key='more_scores',on_change='rerun') as panel:
        if panel.open:
            if len(shown)==2:
                # A heatmap of two groups has one cell: a plain table reads better.
                i,j=shown
                readable_table(pd.DataFrame({'Score':[f'{METRIC_NAMES[m]} ({m.upper()})' for m in ('apd','prt','samd')],
                                             f'{labels[i]} ↔ {labels[j]}':[number(float(matrices[m]['mean'][i,j])) for m in ('apd','prt','samd')]}))
                for metric in ('apd','prt','samd'):
                    st.caption(f'**{metric.upper()}**: {METRIC_NOTES[metric]}',help=METRIC_HELP[metric])
            else:
                for metrics in [('apd','prt'),('samd',)]:
                    cols=st.columns(2)
                    for col,metric in zip(cols,metrics):
                        with col:
                            stats={stat:matrix[pick] for stat,matrix in matrices[metric].items()}
                            plot_chart(corpus_score_heatmap(stats,counts[pick],visible,metric),width='stretch',key=f'all_pairs_{metric}')
                            st.caption(METRIC_NOTES[metric],help=METRIC_HELP[metric])
                st.caption('Each heatmap has its own colour scale.')
            if not np.isfinite(matrices['prt']['mean'][pick][~np.eye(len(shown),dtype=bool)]).all():
                st.caption('PRT cells marked n/a are undefined: a corpus centroid has no direction under cosine distance.')
    rows=[]
    for a in range(len(shown)):
        for b in range(a+1,len(shown)):
            i,j=shown[a],shown[b]
            row={f'{noun[0]}_a':labels[i],f'{noun[0]}_b':labels[j],'usages_a':int(counts[i,j]),'usages_b':int(counts[j,i])}
            for metric in ('apd','prt','amd','samd'):
                row[metric]=float(matrices[metric]['mean'][i,j])
                if mode=='repeated':
                    row[f'{metric}_min'],row[f'{metric}_max']=float(matrices[metric]['min'][i,j]),float(matrices[metric]['max'][i,j])
            rows.append({**row,'distance':distance,'sampling':mode,'seed':st.session_state.sampling_seed})
    st.download_button(f'Download scores for all {noun[0]} pairs (CSV)',data=pd.DataFrame(rows).to_csv(index=False),
                       file_name=f'{data.word}_scores.csv',mime='text/csv',key='scores_download',
                       help='One row per pair of corpora, with the four scores, the number of usages compared and the settings used.')


def render_time_scores(data,embeddings):
    """The scores read along the corpus order: each period against the one before it and against the first."""
    found=_all_pair_scores(data,embeddings,('corpus','corpora'),'time_pairs')
    if found is None:
        return
    labels,shown,matrices,_=found
    if len(shown)<3:
        st.info('Change over time needs at least three periods switched on in the plot legend (Map).')
        return
    render_change_over_time(matrices,[labels[i] for i in shown],shown)


def change_over_time_figure(matrices,labels,positions,metrics=('apd','prt','amd','samd')):
    """Each period against the one before it, and against the first, for the given scores (two panels to a row)."""
    columns=min(2,len(metrics)) if len(metrics)!=3 else 3
    rows=-(-len(metrics)//columns)
    fig=make_subplots(rows=rows,cols=columns,shared_xaxes=True,vertical_spacing=.16,horizontal_spacing=.08,
                      subplot_titles=[f'{METRIC_NAMES[m]} ({m.upper()})' for m in metrics])
    later=labels[1:]
    series=[('From the period before','#38bdf8','solid',[(positions[k-1],positions[k]) for k in range(1,len(labels))],
             [labels[k-1] for k in range(1,len(labels))]),
            (f'From the first period ({labels[0]})','#f59e0b','dash',[(positions[0],positions[k]) for k in range(1,len(labels))],
             [labels[0]]*len(later))]
    for index,metric in enumerate(metrics):
        for name,colour,dash,pairs,earlier in series:
            values=[float(matrices[metric]['mean'][i,j]) for i,j in pairs]
            fig.add_trace(go.Scatter(x=later,y=values,name=name,legendgroup=name,showlegend=index==0,mode='lines+markers',
                                     line=dict(color=colour,dash=dash,width=2.5),marker=dict(size=7,color=colour),customdata=earlier,
                                     hovertemplate='%{customdata} → %{x}<br>'+metric.upper()+': %{y:.4g}<extra></extra>'),
                          row=index//columns+1,col=index%columns+1)
    fig.update_xaxes(type='category',categoryorder='array',categoryarray=later,tickangle=-30)
    fig.update_layout(height=170+250*rows if rows>1 else 400,margin=dict(l=60,r=30,t=70,b=70),
                      legend=dict(orientation='h',yanchor='bottom',y=1.08,xanchor='left',x=0))
    return fig


def render_change_over_time(matrices,labels,positions):
    st.subheader('Change over time')
    st.caption('AMD read along the corpus order: each period compared with the one before it, and with the first period. '
               'A rise against the period before marks a step change; a steady climb against the first period marks gradual drift.')
    plot_chart(change_over_time_figure(matrices,labels,positions,('amd',)),width='stretch',key='change_over_time',)
    with st.expander('More scores: APD, PRT and SAMD',key='more_time_scores',on_change='rerun') as panel:
        if panel.open:
            plot_chart(change_over_time_figure(matrices,labels,positions,('apd','prt','samd')),width='stretch',key='change_over_time_more')
            st.caption('Each panel has its own scale.')


def words_by_corpus(data,words,word_order,corpus_order):
    """Usages of each word (rows) in each corpus (columns)."""
    counts=pd.DataFrame(0,index=list(word_order),columns=list(corpus_order))
    for word,usage in zip(words,data.usages):
        if word in counts.index and usage.corpus in counts.columns:
            counts.loc[word,usage.corpus]+=1
    return counts


def render_words_by_corpus(data,words,word_order):
    """How often each word occurs in each corpus: counts, shares, per million words, and a stacked chart."""
    order=ordered_labels(data)
    counts=words_by_corpus(data,words,word_order,order)
    totals=counts.sum(axis=0)
    shares=counts.div(totals.where(totals>0),axis=1)
    st.subheader('Words by corpus')
    st.caption('How often each word occurs among the loaded usages of each corpus. Shares are out of the corpus’s loaded usages of all the words, '
               'so they show how the words divide the corpus between them.')
    percentages=st.checkbox('Show percentages of each corpus’s usages',value=True,key='words_by_corpus_percentages')
    table=(shares*100).round(1) if percentages else counts.copy()
    table.loc['All words']=totals if not percentages else totals.where(totals==0,100.0)
    display=table.rename_axis('Word').reset_index()
    if percentages:
        for corpus in order:
            display[corpus]=[('—' if pd.isna(v) else f'{v:.1f}%') for v in display[corpus]]
    else:
        display['Total']=list(counts.sum(axis=1))+[int(totals.sum())]
    readable_table(display)
    stats=st.session_state.get('_corpus_stats') or {}
    complete=stats and all(c in stats and stats[c].get('words') and stats[c].get('found')==len(data.indices_for_corpus(c)) for c in order)
    if complete:
        per_million=counts.astype(float)
        for corpus in order:
            per_million[corpus]=counts[corpus]/stats[corpus]['words']*1e6
        st.markdown('**Per million words**')
        readable_table(per_million.round(1).rename_axis('Word').reset_index())
        st.caption('Usages of each word divided by the size of the corpus, so corpora of different sizes can be compared.')
    elif stats:
        st.caption('Per million words is not shown, because a maximum number of usages per corpus was set: the loaded usages are a sample.')
    palette=corpus_colours(list(word_order),st.session_state.palette_name)
    totals_chart=st.checkbox('Show totals instead of shares in the chart',value=False,key='words_by_corpus_totals',
                             help='Shares: every corpus reaches the same height. Totals: bars count usages, so a corpus with more usages has a taller bar.')
    fig=go.Figure()
    for word in word_order:
        fig.add_trace(go.Bar(x=order,y=(counts.loc[word] if totals_chart else shares.loc[word]).tolist(),name=word,marker_color=palette[word],
                             customdata=[[int(counts.loc[word,c]),int(totals[c])] for c in order],
                             hovertemplate='%{x}<br>'+word.replace('%','%%')+': %{customdata[0]} of %{customdata[1]} usages<extra></extra>'))
    fig.update_layout(barmode='stack',height=360,margin=dict(l=60,r=30,t=30,b=60),
                      yaxis_title='Usages' if totals_chart else 'Share of the corpus’s usages',
                      legend=dict(title='Word',orientation='h',yanchor='bottom',y=1.02,xanchor='left',x=0))
    fig.update_xaxes(type='category',categoryorder='array',categoryarray=order)
    fig.update_yaxes(tickformat=',d' if totals_chart else '.0%')
    plot_chart(fig,width='stretch',key='words_by_corpus_chart')
    st.download_button('Download words by corpus (counts)',data=counts.rename_axis('word').to_csv(),
                       file_name=f'{data.word}_words_by_corpus.csv',mime='text/csv',key='words_by_corpus_download')


def _group_score(array,ia,ib,labels,metric):
    """One score between two sets of usages, with the score settings; NaN when either is empty."""
    if not len(ia) or not len(ib):
        return float('nan')
    comparison=cached_comparison(array[ia],array[ib],labels,st.session_state.distance_metric,st.session_state.sampling_mode,
                                 st.session_state.sampling_repetitions,st.session_state.sampling_seed)
    return float(comparison.summary.loc[metric,'mean'])


def word_change_series(array,data,words,word_order,corpus_order,metric,baseline='previous',reference=None):
    """Scores for the words over time.

    Returns ``(each_word, between)``: for each word, its score in each later
    period against the period before (or the first); and, for each other word,
    its score against ``reference`` within each period.
    """
    groups={}
    for i,(word,usage) in enumerate(zip(words,data.usages)):
        groups.setdefault((word,usage.corpus),[]).append(i)
    each={}
    for word in word_order:
        values=[]
        for k in range(1,len(corpus_order)):
            earlier=corpus_order[k-1] if baseline=='previous' else corpus_order[0]
            values.append(_group_score(array,groups.get((word,earlier),[]),groups.get((word,corpus_order[k]),[]),
                                       (f'{word} · {earlier}',f'{word} · {corpus_order[k]}'),metric))
        each[word]=values
    between={}
    if reference is not None:
        for word in word_order:
            if word==reference:
                continue
            between[word]=[_group_score(array,groups.get((reference,c),[]),groups.get((word,c),[]),
                                        (f'{reference} · {c}',f'{word} · {c}'),metric) for c in corpus_order]
    return each,between


def render_word_change(data,words,word_order,embeddings):
    """Each word over the periods, and the distance between words period by period."""
    order=ordered_labels(data)
    if not st.session_state.chronological or len(order)<3:
        return
    st.subheader('Words over time')
    st.caption('The same scores as above, computed for each word on its own usages. One score at a time keeps the charts readable; '
               'a gap means a word has no usages in that period.')
    st.caption('Scores also depend on how many usages are compared: a word with few usages in each period gets higher scores from that alone. '
               'Read each word’s line against itself over time; the levels of two words are not directly comparable unless their counts are similar '
               '(see Words by corpus).')
    metric=st.radio('Score',['amd','apd','prt','samd'],index=0,horizontal=True,key='word_change_metric',
                    format_func=lambda m:f'{METRIC_NAMES[m]} ({m.upper()})',help=METRIC_HELP['amd'])
    cols=st.columns(2)
    baseline=cols[0].radio('Compare each period with',['previous','first'],horizontal=True,key='word_change_baseline',
                           format_func={'previous':'the period before','first':f'the first period ({order[0]})'}.get)
    reference=cols[1].selectbox('Distance between words: from',word_order,key='word_change_reference',
                                help='Each other word is compared with this one, within each period.')
    groups={}
    for word,usage in zip(words,data.usages):
        groups[(word,usage.corpus)]=groups.get((word,usage.corpus),0)+1
    sizes=[(groups.get((w,a),0),groups.get((w,b),0)) for w in word_order for a in order for b in order if a<b]
    if not comparison_allowed(sizes,'word_change','Comparing the words in each period'):
        return
    with st.spinner('Comparing the words in each period…'):
        each,between=word_change_series(embeddings.detach().cpu().numpy(),data,words,word_order,order,metric,baseline,reference)
    colours=corpus_colours(list(word_order),st.session_state.palette_name)
    later=order[1:]
    fig=go.Figure()
    for word,values in each.items():
        fig.add_trace(go.Scatter(x=later,y=values,name=word,mode='lines+markers',connectgaps=False,
                                 line=dict(color=colours[word],width=2.5),marker=dict(size=7,color=colours[word]),
                                 hovertemplate=word.replace('%','%%')+' · %{x}<br>'+metric.upper()+': %{y:.4g}<extra></extra>'))
    fig.update_layout(title=f'Each word against {"the period before" if baseline=="previous" else "the first period"}',height=380,
                      margin=dict(l=60,r=30,t=70,b=70),yaxis_title=metric.upper(),
                      legend=dict(title='Word',orientation='h',yanchor='bottom',y=1.02,xanchor='left',x=0))
    fig.update_xaxes(type='category',categoryorder='array',categoryarray=later,tickangle=-30)
    plot_chart(fig,width='stretch',key='word_change_each')
    if between:
        fig=go.Figure()
        for word,values in between.items():
            fig.add_trace(go.Scatter(x=order,y=values,name=f'{word}',mode='lines+markers',connectgaps=False,
                                     line=dict(color=colours[word],width=2.5),marker=dict(size=7,color=colours[word]),
                                     hovertemplate=f'{reference} ↔ {word}'.replace('%','%%')+' · %{x}<br>'+metric.upper()+': %{y:.4g}<extra></extra>'))
        fig.update_layout(title=f'Distance between “{reference}” and each other word, period by period',height=380,
                          margin=dict(l=60,r=30,t=70,b=70),yaxis_title=metric.upper(),
                          legend=dict(title=f'Compared with {reference}',orientation='h',yanchor='bottom',y=1.02,xanchor='left',x=0))
        fig.update_xaxes(type='category',categoryorder='array',categoryarray=order,tickangle=-30)
        plot_chart(fig,width='stretch',key='word_change_between')
        st.caption('A falling line means the two words are used more alike in later periods; a rising line, that they drift apart.')


EXPLORE_VIEW='explore_view'


def explore_tabs(data,noun):
    """Explore's tabs: Over time only when the corpora are periods in order, three or more."""
    tabs=['Map','Scores',f'Compare two {noun[1]}']
    if st.session_state.chronological and len(ordered_labels(data))>=3 and (noun[0]=='corpus' or words_active(data)):
        tabs.append('Over time')
    return tabs


def render_full_space():
    st.header('Explore all corpora')
    eu=st.session_state.embedded_usages
    if eu is None:
        st.info('Load data and embed usages in Data & Model first.')
        return
    data=st.session_state.target_word_data
    render_grouping_controls(data,'explore')
    view=current_view(data)
    noun=view.noun
    vectors=view.embeddings(eu.embeddings)
    active=words_active(data)
    tabs=explore_tabs(data,noun)
    current=st.session_state.get(EXPLORE_VIEW)
    if current not in tabs:
        # Switching between corpora and words renames the comparison tab: stay on it.
        st.session_state[EXPLORE_VIEW]=tabs[2] if str(current).startswith('Compare two') else tabs[0]
    # Only the open tab is computed: the others cost nothing until they are opened.
    panels=dict(zip(tabs,st.tabs(tabs,key=EXPLORE_VIEW,on_change='rerun')))
    with panels['Map']:
        if panels['Map'].open:
            st.caption('Each point is one usage of the word. Points lie close together when the language model represents those usages similarly, '
                       f'which usually means similar meaning or context. Hover a point to read its sentence; click a {noun[0]} in the legend to hide it.')
            if len(view.data.corpus_labels)>1:
                st.markdown(f'How far apart the {noun[1]} are: the **Scores** tab compares every pair, and **{tabs[2]}** one pair, '
                            'with the sentences behind its scores.')
            meta=eu.metadata
            reading='whole sentence' if meta.get('representation')=='sentence' else 'target word'
            st.caption(f"Details: {model_name(meta.get('model_id'))}, {layer_name(meta.get('layer',-1))}, {reading} · {eu.dim} dimensions · "
                       f"{len(view.data):,} usages in {len(view.data.corpus_labels)} {noun[1] if len(view.data.corpus_labels)!=1 else noun[0]}")
            render_plot(view.data,vectors,key='full',hover_notes=view.hover_notes(),download_columns=view.download_columns(),noun=noun)
    with panels['Scores']:
        if panels['Scores'].open:
            if active:
                words,word_order=loaded_words(data)
                render_words_by_corpus(data,words,word_order)
            render_corpus_scores(view.data,vectors,noun)
    compare=tabs[2]
    with panels[compare]:
        if panels[compare].open:
            render_comparison(view.data,vectors,key='full',noun=noun)
    if 'Over time' in panels:
        with panels['Over time']:
            if panels['Over time'].open:
                if noun[0]=='corpus':
                    render_time_scores(view.data,vectors)
                if active:
                    words,word_order=loaded_words(data)
                    render_word_change(data,words,word_order,eu.embeddings)
                if any(u.time is not None for u in data.usages):
                    st.caption('Each usage can also be placed by its own date: choose **Over time** under Plot view, in Map.')
