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
from semlens.metrics import compute_per_definition_metrics,nearest_counterparts
from semlens.spaces import project_to_definition_space
from app.components.scatter import render_scatter,render_lda_weights_chart
from app.presentation import plot_chart,readable_table,sentence_pair_table
from app.common import (cached_projection,cached_comparison,cached_corpus_scores,ordered_labels,number,show_usage,
                        corpus_colours,METRIC_HELP)


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


def render_plot(data,embeddings,*,key,definition_labels=None,annotation=False,annotations=None,sense_classes=None):
    """Every point is included; pair selectors never filter this shared plot."""
    order=ordered_labels(data)
    options=['PCA','UMAP','t-SNE']
    if len(order)>1:
        options.append('LDA')
    if definition_labels:
        if annotation:
            options.append('Definition axes')
        else:
            options.insert(0,'Definition axes')
    view_key=f'{key}_view'
    if definition_labels and st.session_state.get(view_key)=='Choose axes':
        st.session_state[view_key]='Definition axes'
    help_text='All corpora share one plot. PCA shows global variation; UMAP and t-SNE emphasise neighbourhoods. LDA explicitly seeks corpus separation.'
    if definition_labels:
        help_text='Definition axes lets you put a chosen definition on each axis, or combine a definition with PC1/PC2. '+help_text
    method=st.selectbox('Plot view',options,key=view_key,help=help_text)
    x=embeddings.detach().cpu().numpy()
    anchors=None
    anchor_labels=None
    if definition_labels and not annotation:
        show=st.checkbox('Show definition anchors',key=f'{key}_anchors',help='The definitions projected into the same space as the usages.')
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
                    return pc_array[:,int(choice[-1])-1],f'{choice} (definition space)'
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
        elif method=='PCA':
            coords,anchor_coords=_pca_axes(x,anchors)
            axes=('PC1','PC2')
        else:
            coords=cached_projection(x,'tsne' if method=='t-SNE' else 'umap')
            anchor_coords=None
            axes=(f'{method} 1',f'{method} 2')
            if method=='UMAP' and len(x)<4:
                st.caption('Fewer than four usages: showing PCA coordinates instead of UMAP.')
                axes=('PC1','PC2')
    except (ValueError,ImportError,np.linalg.LinAlgError) as exc:
        st.info(f'{exc} Showing PCA instead.')
        coords,anchor_coords=_pca_axes(x,anchors)
        axes=('PC1','PC2')
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
    if annotation:
        interaction=st.radio('Selection',['Click','Lasso'],horizontal=True,key=f'{key}_interaction').lower()
    # Actual coordinates determine the selection widget identity, preventing old selections
    # from being applied after a view/data change while keeping positions stable on annotation.
    import hashlib
    signature=hashlib.sha256(coords.tobytes()+str(axes).encode()).hexdigest()[:12]
    result=render_scatter(coords,data.corpus_list,data.sentences,data.word,
        palette_name=st.session_state.palette_name,title=f'{data.word} · {len(order)} corpora',axis_labels=axes,
        char_spans=data.char_spans,corpus_order=order,definition_labels_per_point=nearest,
        definition_anchor_coords=anchor_coords,definition_anchor_labels=anchor_labels,
        show_definition_anchors=anchor_coords is not None,highlight_indices=highlight,
        annotations=annotations,sense_classes=sense_classes,enable_selection=annotation,interaction_mode=interaction,
        key=f'{key}_chart_{st.session_state.data_revision}_{signature}_{interaction}')
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
    st.caption('The plot includes all corpora. Select the two corpora whose scores and counterparts you want to inspect.')
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
                   help=f'Mean nearest-counterpart distance for usages in {source}, looking in {target}. Higher means weaker counterparts. Corpus composition and sample size affect this score.')
    if st.session_state.chronological:
        earlier,later=sorted([a,b],key=labels.index)
        st.caption(f'Chronology: {earlier} precedes {later}. Inspect weak counterparts for evidence of possible loss or emergence.')
    if len(comparison.draws)>1:
        readable_table(summary.rename_axis('Metric').reset_index())
        st.caption('Min–max describes variation across the sampled comparisons, not a confidence interval. Equal-size corpora produce identical repetitions.')
    space_name='definition space' if definition_labels else 'full embeddings'
    with st.expander(f'Inspect the usage pairs behind AMD — {space_name}',key=f'{key}_pairs',on_change='rerun') as panel:
        if panel.open:
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
                rows=[{'Distance':r.distance,'Usage':data.usages[int(src[int(r.source_index)])].sentence,
                       'Closest counterpart':data.usages[int(dst[int(r.counterpart_index)])].sentence} for r in t.itertuples()]
                table=pd.DataFrame(rows).sort_values('Distance',ascending=False,kind='stable')
                if query:
                    table=table[table['Usage'].str.contains(query,case=False,regex=False)|table['Closest counterpart'].str.contains(query,case=False,regex=False)]
                sentence_pair_table(table,key=f'{key}_pairs_page_{title}_{repetition}_{query}')
                st.caption(f'{len(table):,} of {len(rows):,} pairs · scroll inside either sentence cell to read it in full.')
    if definition_labels:
        with st.expander('Per-definition scores (absolute distance)',key=f'{key}_perdef',on_change='rerun') as panel:
            if panel.open:
                frames=[compute_per_definition_metrics(xa[i],xb[j],definition_labels,(a,b)).set_index('dim') for i,j in comparison.draws]
                numeric=[c for c in frames[0].columns if c!='definition']
                means=sum(f[numeric] for f in frames)/len(frames)
                means.insert(0,'definition',frames[0]['definition'])
                readable_table(means.sort_values('amd',ascending=False),text_columns=['definition'])
                st.caption('Absolute differences compare each scalar definition distance. The embedding-to-definition projection remains cosine distance. Means use the same samples as the scores above; smaller mean distance means closer to the definition.')


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
        edges=np.linspace(0,2,31)
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
        fig.update_xaxes(range=[0,2])
        fig.update_yaxes(rangemode='tozero',tickformat='.0%' if normalised else ',d')
        plot_chart(fig,width='stretch')
        st.caption('Each line connects the values in 30 shared distance bins, with no smoothing. Click a corpus in the legend to hide or show it; double-click to isolate it.')
        readable_table(pd.DataFrame(rows))
        st.caption('These describe distances to this definition; they do not estimate the proportion of usages that belong to a sense.')


def corpus_score_heatmap(stats,counts,labels,metric):
    n=len(labels)
    ticks=['<br>'.join(html.escape(line) for line in textwrap.wrap(textwrap.shorten(label,width=32,placeholder='…'),16)) for label in labels]
    custom=[[[html.escape(labels[i]),html.escape(labels[j]),number(stats['mean'][i,j]),
              number(stats['min'][i,j]),number(stats['max'][i,j]),int(counts[i,j]),int(counts[j,i])]
             for j in range(n)] for i in range(n)]
    text=[['' if i==j else (f'{stats["mean"][i,j]:.3g}' if np.isfinite(stats['mean'][i,j]) else 'undefined')
           for j in range(n)] for i in range(n)]
    fig=go.Figure(go.Heatmap(z=stats['mean'],x=list(range(n)),y=list(range(n)),customdata=custom,
                            colorscale='Viridis',text=text,texttemplate='%{text}' if n<=6 else None,
                            hoverongaps=False,colorbar=dict(title=metric.upper(),thickness=12),
                            hovertemplate='%{customdata[0]} ↔ %{customdata[1]}<br>'+metric.upper()+': %{customdata[2]}'
                            '<br>Sample min–max: %{customdata[3]}–%{customdata[4]}'
                            '<br>Usages per sample: %{customdata[5]} / %{customdata[6]}<extra></extra>'))
    fig.update_layout(title=metric.upper(),height=max(370,32*n+170),margin=dict(l=95,r=50,t=55,b=100),
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
    indices=tuple(tuple(data.indices_for_corpus(label)) for label in labels)
    with st.spinner('Comparing all corpus pairs…'):
        matrices,counts=cached_corpus_scores(embeddings.detach().cpu().numpy(),indices,tuple(labels),distance,mode,
                                             st.session_state.sampling_repetitions,st.session_state.sampling_seed)
    for metrics in [('apd','prt'),('amd','samd')]:
        cols=st.columns(2)
        for col,metric in zip(cols,metrics):
            with col:
                plot_chart(corpus_score_heatmap(matrices[metric],counts,labels,metric),width='stretch',key=f'all_pairs_{metric}')
                st.caption(METRIC_HELP[metric])
    st.caption('Each heatmap has its own colour scale. The diagonal is omitted: within-corpus comparisons are not change scores. Sample sizes can differ between corpus pairs.')
    if not np.isfinite(matrices['prt']['mean'][~np.eye(len(labels),dtype=bool)]).all():
        st.caption('Blank PRT cells outside the diagonal are undefined: a corpus centroid has no direction under cosine distance.')


def render_full_space():
    st.header('Explore all corpora')
    eu=st.session_state.embedded_usages
    if eu is None:
        st.info('Load data and embed usages in Data & Model first.')
        return
    data=st.session_state.target_word_data
    st.caption(f"Representation: {eu.metadata.get('representation','unspecified')} · {eu.dim} dimensions · all {len(data.corpus_labels)} corpora")
    render_plot(data,eu.embeddings,key='full')
    render_corpus_scores(data,eu.embeddings)
    render_comparison(data,eu.embeddings,key='full')
