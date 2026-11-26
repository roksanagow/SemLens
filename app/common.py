"""Shared rendering and cached numerical work; cache keys include actual inputs."""
from __future__ import annotations
import html
import re
from pathlib import Path
import numpy as np
import streamlit as st
import torch
from semlens.embeddings import load_model, resolve_model_name
from semlens.metrics import compute_sampled_metrics,SampledComparison,within_amd_baseline
from semlens.reduction import reduce_to_2d
from semlens.utils import palette_for

from app.paths import ROOT,STORAGE,storage_folder
SAMPLING_HELP=('Balanced: use an equal number of usages from each compared corpus, based on the smaller corpus. '
               'Repeated balanced: repeat without replacement within each sample. A usage may take part in several '
               'repetitions but appears once in the plot. The score range shows sampling variability, not a confidence interval.')
METRIC_NAMES={'apd':'Average pairwise distance','prt':'Prototype distance',
              'amd':'Average minimum distance','samd':'Symmetric average minimum distance'}
# A and B are the usages of the two corpora; d is the vector distance chosen under Score settings.
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
# A short animation of each score on ten usages, five in each of two corpora (blue and orange),
# drawn by hand: made by scripts/build_score_animations.py, shown at the top of the score's (?) help.
METRIC_PICTURES={
    'apd':'Every blue usage is compared with every orange one: APD is the average length of all these lines.',
    'prt':'The crosses mark each corpus’s average usage, its prototype: PRT is the distance between the two crosses.',
    'amd':'Green: each blue usage to its closest orange one. Red: each orange usage to its closest blue one. '
          'AMD averages the lengths in each direction, then the two averages.',
    'samd':'Each usage is paired with at most one usage of the other corpus, the closest pairs first, '
           'so outlying usages are left with distant partners: SAMD is the average length of the pairs.',
}


def _picture(metric):
    import base64
    path=Path(__file__).resolve().parent/'data'/'score_help'/f'{metric}.gif'
    if not path.is_file():
        return ''
    data=base64.b64encode(path.read_bytes()).decode()
    return (f'![{METRIC_NAMES[metric]}, drawn on ten usages](data:image/gif;base64,{data})\n\n'
            f'*In the drawing, blue and orange circles are the usages of two corpora.* {METRIC_PICTURES[metric]}\n\n')


METRIC_HELP={metric:f'**{METRIC_NAMES[metric]} ({metric.upper()})**\n\n{_picture(metric)}{METRIC_FORMULAS[metric]}\n\n{METRIC_NOTES[metric]}'
             for metric in METRIC_NAMES}
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
    '**Usage** · one occurrence of a word together with its context: the sentence, line or paragraph it appears in.\n\n'
    '**Corpus** · a collection of texts to compare with others, such as a period, a genre or a variety.\n\n'
    '**Embedding** · the list of numbers a language model gives a usage. Usages the model treats as similar in meaning get similar embeddings: '
    'the model’s judgement, not a dictionary’s.\n\n'
    '**Target word / whole sentence** · a usage represented by the embedding of the word itself, or by the average over the whole sentence, '
    'which also reflects the topic.\n\n'
    '**Distance** · how different two embeddings are: 0 for the same, larger for less alike. Cosine distance is the usual choice.\n\n'
    '**Map (projection)** · a two-dimensional picture of the embeddings: main directions of variation (PCA), neighbourhoods (UMAP, t-SNE), '
    'or what separates the corpora (LDA). Every picture loses information; scores are always computed from the embeddings themselves.\n\n'
    '**Usage map (full embeddings)** · the usages placed by everything the model gives them.\n\n'
    '**Definitions map (definition space)** · each usage described by its distance to each definition you write: which of your senses is it nearest?\n\n'
    '**Change scores (APD, PRT, AMD, SAMD)** · single numbers for how far apart the usages of two corpora are. The (?) beside each gives its formula.\n\n'
    '**Nearest counterpart** · for a usage in one corpus, the most similar usage in another. A usage with no close counterpart may be a new or lost sense.\n\n'
    '**Sense** · a label you name and give to usages when annotating: a sense of the word, or any other category you want to count.\n\n'
    '**Graded annotation** · an Annotation option: each sense of a usage also gets a rating, from 1 (barely) to the top of the scale (clearly). '
    'Another option lets one usage have several senses.\n\n'
    '**Different words** · with several forms, Data & Model asks whether they are forms of one word or different words to compare. '
    'Different words get their own colours, and a switch at the top of each workspace compares corpora or words.\n\n'
    '**Time** · a year, date or other number for each usage, from a table column or from corpus names such as 1999–2003. '
    'It adds the **Over time** view and colouring by time.\n\n'
    '**Discover related words** · the other section (in the sidebar): ranks the words of your corpora by how close their usages are '
    'to seed words or seed usages.')


ADVANCED_HELP=('Shows the settings most analyses can leave at their defaults: the language model, its layer, context window and batch size; '
               'how scores sample usages and measure distance; UMAP and t-SNE settings; corpus order; pre-computed and saved word embeddings; '
               'and Discover’s word filters, model, layer and scoring options. Display settings stay in the sidebar either way.')


def model_name(model_id):
    """A model's short name: the presets' own (XL-LEXEME), else the last part of its ID."""
    from app.ui_data import MODEL_PRESETS
    for label,preset in MODEL_PRESETS.items():
        if preset and preset==model_id:
            return label.split(' · ')[0]
    return (model_id or 'unnamed model').rstrip('/').split('/')[-1]


def layer_name(layer):
    return 'final layer' if int(layer)==-1 else f'layer {int(layer)}'


def advanced():
    """Whether the Advanced options switch is on. Off, the hidden settings keep their remembered or default values."""
    return bool(st.session_state.get('advanced_options'))


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
        # A first use downloads the model: never when it would fill the disk.
        from app.costs import check_download,model_profile
        check_download(model_id)
        model=load_model(model_id,revision=revision)
        model_profile.clear()
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


@st.cache_data(show_spinner=False,max_entries=48)
def cached_within_amd(a,b,labels,distance,mode,repetitions,seed):
    """AMD within each of two corpora, on the very usages that the pair's scores sample: (within a, within b, their mean)."""
    comparison=cached_comparison(a,b,labels,distance,mode,repetitions,seed)
    return within_amd_baseline(torch.from_numpy(a),torch.from_numpy(b),comparison.draws,distance)


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
