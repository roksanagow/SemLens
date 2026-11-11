"""Shared rendering and cached numerical work; cache keys include actual inputs."""
from __future__ import annotations
import html
from pathlib import Path
import numpy as np
import streamlit as st
import torch
from semlens.embeddings import load_model, resolve_model_name
from semlens.metrics import compute_sampled_metrics,SampledComparison
from semlens.reduction import reduce_to_2d
from semlens.utils import get_palette

ROOT=Path(__file__).resolve().parents[1]
STORAGE=ROOT/'.semlens'
SAMPLING_HELP=('Balanced: use an equal number of usages from each compared corpus, based on the smaller corpus. '
               'Repeated balanced: repeat without replacement within each sample. A usage may take part in several '
               'repetitions but appears once in the plot. The score range shows sampling variability, not a confidence interval.')
METRIC_HELP={
    'apd':'Average distance over all cross-corpus pairs. Also reflects variation within the corpora; identical distributions need not score zero.',
    'prt':'Distance between the two mean embeddings. Undefined when a centroid has no direction under cosine distance.',
    'amd':'Mean distance to each usage’s nearest counterpart, averaged over both directions.',
    'samd':'Greedy one-to-one matching distance. With unequal sizes, surplus usages in the larger corpus are left unmatched.',
}


def get_model(model_id, revision=None):
    model_id=resolve_model_name(model_id)
    model=st.session_state.get('_loaded_model')
    if model is not None and model.name==model_id and (not revision or getattr(model.model.config,'_commit_hash',None)==revision):
        return model
    try:
        model=load_model(model_id,revision=revision,local_files_only=True)
    except OSError:
        model=load_model(model_id,revision=revision)
    st.session_state['_loaded_model']=model
    return model


def show_usage(usage, *, prefix=''):
    sentence=usage.sentence
    text=html.escape(sentence[:usage.char_start])+'<mark>'+html.escape(usage.matched_form)+'</mark>'+html.escape(sentence[usage.char_end:])
    st.markdown(f'<div><small>{html.escape(prefix)}{html.escape(usage.corpus)}</small><br>{text}</div>',unsafe_allow_html=True)


def corpus_colours(labels, palette_name):
    # Extend short palettes without cycling a colour for different corpora.
    palette=list(get_palette(palette_name))
    for colour in get_palette('Tab10 (matplotlib)'):
        if colour not in palette:
            palette.append(colour)
    return dict(zip(labels,palette[:len(labels)]))


@st.cache_data(show_spinner=False,max_entries=24)
def cached_projection(array,method):
    return reduce_to_2d(array,method=method)


@st.cache_data(show_spinner=False,max_entries=24)
def cached_comparison(a,b,labels,distance,mode,repetitions,seed):
    # Fix the calculation order for an unordered corpus pair, including sampling
    # and greedy-matching tie breaks; reversing selectors must keep its scores.
    reverse=labels[0]>labels[1]
    aa,bb,ordered=(b,a,tuple(reversed(labels))) if reverse else (a,b,labels)
    result=compute_sampled_metrics(torch.from_numpy(aa),torch.from_numpy(bb),ordered,distance=distance,
                                   mode=mode,repetitions=repetitions,seed=seed)
    return SampledComparison(result.scores,[(j,i) for i,j in result.draws]) if reverse else result


@st.cache_data(show_spinner=False,max_entries=8)
def cached_corpus_scores(array,indices,labels,distance,mode,repetitions,seed):
    """All unordered pairs, with the exact settings used by pair inspection."""
    n=len(labels)
    result={metric:{stat:np.full((n,n),np.nan) for stat in ('mean','min','max')}
            for metric in ('apd','prt','amd','samd')}
    counts=np.zeros((n,n),dtype=int)
    for i in range(n):
        for j in range(i+1,n):
            comparison=cached_comparison(array[list(indices[i])],array[list(indices[j])],(labels[i],labels[j]),
                                         distance,mode,repetitions,seed)
            summary=comparison.summary
            for metric,stats in result.items():
                for stat,matrix in stats.items():
                    matrix[i,j]=matrix[j,i]=summary.loc[metric,stat]
            counts[i,j],counts[j,i]=map(len,comparison.draws[0])
    return result,counts


def number(value):
    if not np.isfinite(value):
        return 'undefined'
    if value != 0 and abs(value)<.0001:
        return f'{value:.3e}'
    return f'{value:.4f}'


def ordered_labels(data):
    order=st.session_state.get('corpus_order',data.corpus_labels)
    return [c for c in order if c in data.corpus_labels]+[c for c in data.corpus_labels if c not in order]
