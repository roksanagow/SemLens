"""Shared rendering and cached numerical work; cache keys include actual inputs."""
from __future__ import annotations
import html
import re
from pathlib import Path
import numpy as np
import streamlit as st
import torch
from semlens.embeddings import load_model, resolve_model_name
from semlens.metrics import compute_sampled_metrics,SampledComparison
from semlens.reduction import reduce_to_2d
from semlens.utils import palette_for

ROOT=Path(__file__).resolve().parents[1]
STORAGE=ROOT/'.semlens'
SAMPLING_HELP=('Balanced: use an equal number of usages from each compared corpus, based on the smaller corpus. '
               'Repeated balanced: repeat without replacement within each sample. A usage may take part in several '
               'repetitions but appears once in the plot. The score range shows sampling variability, not a confidence interval.')
METRIC_NAMES={'apd':'Average pairwise distance','prt':'Prototype distance',
              'amd':'Average minimum distance','samd':'Symmetric average minimum distance'}
# A and B are the usages of the two corpora; d is the distance chosen in the sidebar.
METRIC_FORMULAS={
    'apd':r'$\mathrm{APD}=\frac{1}{|A||B|}\sum_{a\in A}\sum_{b\in B}d(a,b)$',
    'prt':r'$\mathrm{PRT}=d(\bar a,\bar b)$, where $\bar a$ and $\bar b$ are the mean vectors of $A$ and $B$',
    'amd':r'$\mathrm{AMD}=\frac{1}{2}\left(\frac{1}{|A|}\sum_{a\in A}\min_{b\in B}d(a,b)+\frac{1}{|B|}\sum_{b\in B}\min_{a\in A}d(a,b)\right)$',
    'samd':r'$\mathrm{SAMD}=\frac{1}{|M|}\sum_{(a,b)\in M}d(a,b)$, where $M$ pairs each usage with at most one usage of the other corpus, '
           r'taking the closest remaining pair first',
}
METRIC_NOTES={
    'apd':'The mean distance over every cross-corpus pair of usages. It also reflects variation within each corpus, so two samples of the same usages need not score zero.',
    'prt':'The distance between the two corpus averages (“prototypes”). Undefined under cosine distance when an average has no direction.',
    'amd':'For each usage, the distance to its closest counterpart in the other corpus; averaged within each direction, then across the two directions.',
    'samd':'A one-to-one matching: the closest unmatched pair is matched first, and so on. With unequal sizes, the surplus usages of the larger corpus stay unmatched.',
}
METRIC_HELP={metric:f'**{METRIC_NAMES[metric]} ({metric.upper()})**\n\n{METRIC_FORMULAS[metric]}\n\n{METRIC_NOTES[metric]}' for metric in METRIC_NAMES}
DIRECTIONAL_AMD_FORMULA=r'$\mathrm{AMD}_{A\to B}=\frac{1}{|A|}\sum_{a\in A}\min_{b\in B}d(a,b)$'
PROJECTION_HELP=(
    'Every usage is a point in a space with hundreds of dimensions; each view below flattens that space to two. '
    'Scores are always computed in the full space, never from these pictures.\n\n'
    '**PCA** (principal component analysis) finds the direction along which the usages vary most (PC1), then the direction of greatest remaining '
    'variation at right angles to it (PC2), and plots each usage by its position along the two. It is a plain rotation of the data, so distances '
    'are comparable across the plot, but it shows only part of the variation: the percentage on each axis says how much.\n\n'
    '**UMAP** and **t-SNE** place usages so that close neighbours stay close. They show clusters well, but the size of a cluster and the gaps '
    'between clusters carry little meaning, and the layout changes with the data.\n\n'
    '**LDA** (linear discriminant analysis) looks for the directions that best separate the corpora, using the corpus labels. '
    'It will always show some separation, so treat it as a way to look for differences, not as evidence of them.')


GLOSSARY=(
    '**Usage** · one occurrence of the target word, read in its sentence.\n\n'
    '**Corpus** · one collection of text to compare with the others: a period, a genre, a variety.\n\n'
    '**Embedding** · the list of numbers a language model assigns to a usage. Usages the model treats as similar in meaning get similar numbers. '
    'It is the model’s judgement, not a dictionary’s.\n\n'
    '**Target word / whole sentence** · whether a usage is represented by the model’s numbers for the word itself in its context, or by the average over the whole sentence, '
    'which also reflects the topic.\n\n'
    '**Distance** · how different two usages are for the model: 0 for identical, larger for less alike. Cosine distance compares direction and is the usual choice.\n\n'
    '**Projection** · a two-dimensional picture of the embeddings (PCA, UMAP, t-SNE, LDA). Every picture loses information; scores never use it.\n\n'
    '**Definition space** · each usage described by its distance to each definition you supply. A way to ask which of your candidate senses a usage is nearest.\n\n'
    '**Change scores (APD, PRT, AMD, SAMD)** · single numbers for how far apart the usages of two corpora are. The (?) beside each gives its formula.\n\n'
    '**Nearest counterpart** · for a usage in one corpus, the most similar usage in another. A usage with no close counterpart is a candidate new or lost use.\n\n'
    '**Sense class** · a label you define and assign to usages when annotating.')


def natural_sort_key(text):
    """Sort names so that numbers compare as numbers: 1999 before 2004, corpus_2 before corpus_10."""
    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r'(\d+)',str(text))]


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
    # Short palettes are extended, never cycled; the sequential one follows corpus order.
    return dict(zip(labels,palette_for(palette_name,len(labels))))


@st.cache_data(show_spinner=False,max_entries=24)
def cached_projection(array,method,**settings):
    """UMAP or t-SNE coordinates; ``settings`` are the reducer's own parameters."""
    return reduce_to_2d(array,method=method,**settings)


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
