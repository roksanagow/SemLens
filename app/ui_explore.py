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
from app.presentation import plot_chart,readable_table,sentence_pair_table,nearest_usage_html
from app.common import (cached_projection,cached_comparison,cached_corpus_scores,ordered_labels,number,show_usage,
                        corpus_colours,METRIC_HELP,METRIC_NAMES,DIRECTIONAL_AMD_FORMULA,PROJECTION_HELP)
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


def _projection_settings(method,key,count,dimensions,distance):
    """Optional settings for the current view, with defaults that suit most data."""
    settings={}
    if method=='PCA':
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
    if method in {'UMAP','t-SNE'}:
        st.caption(f'Neighbours are found by {distance} distance, the vector distance chosen in the sidebar.')
    return settings


def render_plot(data,embeddings,*,key,definition_labels=None,annotation=False,annotations=None,sense_classes=None,blind=False,
                omit_indices=()):
    """Every point is included; pair selectors never filter this shared plot.

    Corpora switched off in the legend stay off across views. ``blind`` hides
    which corpus a usage belongs to, for annotation.
    """
    order=ordered_labels(data)
    options=['PCA','UMAP','t-SNE']
    # LDA arranges usages by corpus, which would give the corpora away.
    if len(order)>1 and not blind:
        options.append('LDA')
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
    help_text='All corpora share one plot. '+PROJECTION_HELP
    if definition_labels:
        help_text='**Definition axes** puts a chosen definition on each axis (the distance of every usage to that definition), or combines a definition with PC1/PC2.\n\n'+help_text
    method=st.selectbox('Plot view',options,key=view_key,help=help_text)
    x=embeddings.detach().cpu().numpy()
    distance='euclidean' if x.shape[1]==1 else st.session_state.distance_metric
    plot_options=st.expander('Plot options')
    with plot_options:
        settings=_projection_settings(method,key,len(x),x.shape[1],distance)
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
        highlight_label=f'Contains “{search}”'
        per_corpus=', '.join(f'{corpus}: {sum(data.corpus_list[i]==corpus for i in highlight)}' for corpus in order) if not blind else ''
        st.caption(f'{len(highlight):,} of {len(data):,} usages contain “{search}”.'+(f' {per_corpus}.' if per_corpus and highlight else ''))
    # Actual coordinates determine the selection widget identity, preventing old selections
    # from being applied after a view/data change while keeping positions stable on annotation.
    import hashlib
    signature=hashlib.sha256(coords.tobytes()+str(axes).encode()).hexdigest()[:12]
    result=render_scatter(coords,data.corpus_list,data.sentences,data.word,
        palette_name=st.session_state.palette_name,title=f'{data.word} · {len(order)} corpora',axis_labels=axes,
        char_spans=data.char_spans,corpus_order=order,definition_labels_per_point=nearest,
        definition_anchor_coords=anchor_coords,definition_anchor_labels=anchor_labels,
        show_definition_anchors=anchor_coords is not None,highlight_indices=highlight,highlight_label=highlight_label,
        annotations=annotations,sense_classes=sense_classes,enable_selection=annotation,interaction_mode=interaction,
        hidden_corpora=hidden_corpora(order),blind=blind,omit_indices=omit_indices,
        key=f'{key}_chart_{st.session_state.data_revision}_{signature}_{interaction}')
    if not blind:
        with plot_options:
            table=pd.DataFrame({'usage_id':[u.usage_id for u in data.usages],'corpus':data.corpus_list,
                                'matched_form':[u.matched_form for u in data.usages],'sentence':data.sentences,
                                'start':[u.char_start for u in data.usages],'end':[u.char_end for u in data.usages],
                                'x':coords[:,0],'y':coords[:,1],'x_axis':axes[0],'y_axis':axes[1]})
            if annotations is not None:
                table['sense']=[annotations.get(i,'') for i in range(len(data))]
            st.download_button('Download plot data (CSV)',data=table.to_csv(index=False),file_name=f'{data.word}_plot_{method.lower().replace(" ","_")}.csv',
                               mime='text/csv',key=f'{key}_plot_download',
                               help='One row per usage with its sentence, corpus and position in this plot, for drawing the figure in other software.')
    if lda_result is not None and definition_labels and not annotation:
        with st.expander('LDA definition weights'):
            render_lda_weights_chart(lda_definition_weights(lda_result,definition_labels),palette_name=st.session_state.palette_name)
            st.caption('Weights act on distance features: positive means greater distance increases LD1. Read them alongside corpus positions and mean distances; they are not sense frequencies.')
    return result


def render_comparison(data,embeddings,*,key,definition_labels=None):
    labels=ordered_labels(data)
    if len(labels)<2:
        st.info('One corpus is available. You can explore and annotate its usages; change scores need a second corpus.')
        return
    st.subheader('Compare corpora')
    st.caption('The plot includes all corpora. Select the two corpora whose scores and counterparts you want to inspect. '
               'A higher score means the usages of the two corpora are further apart; what counts as high depends on the model and the word, '
               'so compare scores with each other and read the sentences behind them.')
    cols=st.columns(2)
    a=cols[0].selectbox('From corpus',labels,key=f'{key}_from')
    b=cols[1].selectbox('To corpus',[c for c in labels if c!=a],key=f'{key}_to')
    ia,ib=data.indices_for_corpus(a),data.indices_for_corpus(b)
    xa,xb=embeddings[ia],embeddings[ib]
    distance='absolute' if definition_labels and len(definition_labels)==1 else st.session_state.distance_metric
    with st.spinner('Comparing usages…'):
        comparison=cached_comparison(xa.numpy(),xb.numpy(),(a,b),distance,
                    st.session_state.sampling_mode,st.session_state.sampling_repetitions,st.session_state.sampling_seed)
    summary=comparison.summary
    na,nb=map(len,comparison.draws[0])
    st.caption(f'Scores use {na:,} / {len(ia):,} occurrences from {a} and {nb:,} / {len(ib):,} from {b}; '
               f'{len(comparison.draws)} sample(s). Distances: {distance}. All retained occurrences remain in the plot.')
    if distance=='absolute':
        st.caption('With one definition the representation is a scalar, so scores use absolute distance.')
    if st.session_state.sampling_mode=='unequal' and na!=nb:
        st.caption(f'SAMD matches {min(na,nb):,} pairs; {abs(na-nb):,} surplus occurrences remain unmatched.')
    cols=st.columns(4)
    for col,metric in zip(cols,('apd','prt','amd','samd')):
        col.metric(metric.upper(),number(summary.loc[metric,'mean']),help=METRIC_HELP[metric])
    cols=st.columns(2)
    for col,(source,target) in zip(cols,((a,b),(b,a))):
        metric=f'amd_{source}_to_{target}'
        col.metric(f'AMD: {source} → {target}',number(summary.loc[metric,'mean']),
                   help=f'**Directional average minimum distance**\n\n{DIRECTIONAL_AMD_FORMULA}, with $A$ = {source} and $B$ = {target}\n\n'
                        f'For each usage in {source}, the distance to its closest counterpart in {target}, averaged. Higher means weaker counterparts. '
                        'Corpus composition and sample size affect this score.')
    if st.session_state.chronological:
        earlier,later=sorted([a,b],key=labels.index)
        st.caption(f'Chronology: {earlier} precedes {later}. Inspect weak counterparts for evidence of possible loss or emergence.')
    if len(comparison.draws)>1:
        readable_table(summary.rename_axis('Metric').reset_index())
        st.caption('Min–max describes variation across the sampled comparisons, not a confidence interval. Equal-size corpora produce identical repetitions.')
    space_name='definition space' if definition_labels else 'full embeddings'
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
                           'usages by the absolute difference |x − y|. The sidebar’s vector distance cannot change that: in one dimension '
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
    st.caption(f'Nearest first within each corpus · {distance} distance in {space_name}. Every loaded occurrence is searched, '
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
                    '<br>Occurrences: %{customdata[2]} / %{customdata[3]}'
                    +('<br>Proportion: %{y:.1%}' if normalised else '')+'<extra>%{fullData.name}</extra>'))
            rows.append({'Corpus':corpus,'Occurrences':len(values),'Mean distance':float(values.mean()),
                         'Median distance':float(np.median(values)),'Standard deviation':float(values.std())})
        fig.update_layout(xaxis_title='Cosine distance to definition (smaller = closer)',
                          yaxis_title='Proportion of occurrences' if normalised else 'Occurrences',
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


def render_corpus_scores(data,embeddings):
    labels=ordered_labels(data)
    if len(labels)<2:
        return
    st.subheader(f'Scores for “{data.word}” across all corpora')
    mode=st.session_state.sampling_mode
    distance=st.session_state.distance_metric
    descriptions={'balanced':'equal-size samples for each pair','unequal':'all loaded occurrences',
                  'repeated':f'means over {st.session_state.sampling_repetitions} equal-size samples per pair'}
    st.caption(f'{distance.capitalize()} distance · {descriptions[mode]}. Scores use the full usage embeddings, before 2D projection. Hover a cell for its value and sample size.')
    # Corpora switched off in the plot legend are left out here too.
    hidden=set(hidden_corpora(labels))
    shown=[i for i,label in enumerate(labels) if label not in hidden]
    if hidden:
        st.caption(f'Showing {len(shown)} of {len(labels)} corpora. Switched off in the plot legend: {", ".join(l for l in labels if l in hidden)}. Click a name in the legend to bring it back.')
    if len(shown)<2:
        st.info('Switch on at least two corpora in the plot legend to compare them.')
        return
    indices=tuple(tuple(data.indices_for_corpus(label)) for label in labels)
    with st.spinner('Comparing all corpus pairs…'):
        matrices,counts=cached_corpus_scores(embeddings.detach().cpu().numpy(),indices,tuple(labels),distance,mode,
                                             st.session_state.sampling_repetitions,st.session_state.sampling_seed)
    pick=np.ix_(shown,shown)
    visible=[labels[i] for i in shown]
    for metrics in [('apd','prt'),('amd','samd')]:
        cols=st.columns(2)
        for col,metric in zip(cols,metrics):
            with col:
                stats={stat:matrix[pick] for stat,matrix in matrices[metric].items()}
                plot_chart(corpus_score_heatmap(stats,counts[pick],visible,metric),width='stretch',key=f'all_pairs_{metric}')
                st.caption(METRIC_HELP[metric])
    st.caption('In the formulas, $A$ and $B$ are the usages of the two corpora and $d$ is the distance chosen in the sidebar. '
               'Each heatmap has its own colour scale. The diagonal is omitted: within-corpus comparisons are not change scores. Sample sizes can differ between corpus pairs.')
    if not np.isfinite(matrices['prt']['mean'][pick][~np.eye(len(shown),dtype=bool)]).all():
        st.caption('PRT cells marked n/a are undefined: a corpus centroid has no direction under cosine distance.')
    rows=[]
    for a in range(len(shown)):
        for b in range(a+1,len(shown)):
            i,j=shown[a],shown[b]
            row={'corpus_a':labels[i],'corpus_b':labels[j],'usages_a':int(counts[i,j]),'usages_b':int(counts[j,i])}
            for metric in ('apd','prt','amd','samd'):
                row[metric]=float(matrices[metric]['mean'][i,j])
                if mode=='repeated':
                    row[f'{metric}_min'],row[f'{metric}_max']=float(matrices[metric]['min'][i,j]),float(matrices[metric]['max'][i,j])
            rows.append({**row,'distance':distance,'sampling':mode,'seed':st.session_state.sampling_seed})
    st.download_button('Download scores for all corpus pairs (CSV)',data=pd.DataFrame(rows).to_csv(index=False),
                       file_name=f'{data.word}_scores.csv',mime='text/csv',key='scores_download',
                       help='One row per pair of corpora, with the four scores, the number of usages compared and the settings used.')
    if st.session_state.chronological and len(shown)>=3:
        render_change_over_time(matrices,visible,shown)


def change_over_time_figure(matrices,labels,positions):
    """Each period against the one before it, and against the first, for the four scores."""
    metrics=('apd','prt','amd','samd')
    fig=make_subplots(rows=2,cols=2,shared_xaxes=True,vertical_spacing=.16,horizontal_spacing=.1,
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
                          row=index//2+1,col=index%2+1)
    fig.update_xaxes(type='category',categoryorder='array',categoryarray=later,tickangle=-30)
    fig.update_layout(height=560,margin=dict(l=60,r=30,t=70,b=70),
                      legend=dict(orientation='h',yanchor='bottom',y=1.08,xanchor='left',x=0))
    return fig


def render_change_over_time(matrices,labels,positions):
    st.subheader('Change over time')
    st.caption('The same scores as the heatmaps, read along the corpus order: each period compared with the one before it, and with the first period. '
               'A rise against the period before marks a step change; a steady climb against the first period marks gradual drift. '
               'Each panel has its own scale.')
    plot_chart(change_over_time_figure(matrices,labels,positions),width='stretch',key='change_over_time')


def render_full_space():
    st.header('Explore all corpora')
    eu=st.session_state.embedded_usages
    if eu is None:
        st.info('Load data and embed usages in Data & Model first.')
        return
    data=st.session_state.target_word_data
    st.caption('Each point is one usage of the word. Points lie close together when the language model represents those usages similarly, '
               'which usually means similar meaning or context. Hover a point to read its sentence; click a corpus in the legend to hide it.')
    st.caption(f"Representation: {eu.metadata.get('representation','unspecified')} · {eu.dim} dimensions · all {len(data.corpus_labels)} corpora")
    render_plot(data,eu.embeddings,key='full')
    render_corpus_scores(data,eu.embeddings)
    render_comparison(data,eu.embeddings,key='full')
