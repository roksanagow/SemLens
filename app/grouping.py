"""Corpora or words: which groups the plots, scores and frequencies compare.

With "Different words" chosen in Data & Model, every workspace can colour and
compare usages by corpus or by word, and show only some words (or corpora).
The choice is global: one setting in session state, with a linked switch at
the top of Explore, Definitions and Annotation.

Workspaces draw a *view*: the usages shown, each under its group as its
corpus. Annotations stay keyed by position in the loaded data, so a view
maps its positions back with ``indices``.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
import streamlit as st
from semlens.words import MAX_WORDS, regroup, usage_words


@dataclass
class View:
    data: object                       # the usages shown, grouped
    indices: list[int] | None          # position in the loaded data of each usage shown; None: all, in order
    group_by: str = 'corpus'           # 'corpus' or 'word'
    corpora: list[str] = field(default_factory=list)   # the original corpus of each usage shown
    words: list[str] | None = None     # the word of each usage shown, when words are compared

    @property
    def identity(self):
        return self.indices is None

    @property
    def noun(self):
        """('corpus', 'corpora') or ('word', 'words'): what the groups are called."""
        return ('word','words') if self.group_by=='word' else ('corpus','corpora')

    def embeddings(self,tensor):
        return tensor if tensor is None or self.identity else tensor[self.indices]

    def original(self,position):
        return position if self.identity else self.indices[position]

    def to_original(self,positions):
        return [self.original(int(p)) for p in positions]

    def annotations(self,annotations):
        """Annotations keyed by position in the view."""
        if self.identity:
            return annotations
        return {k:annotations[i] for k,i in enumerate(self.indices) if i in annotations}

    def hover_notes(self,hide_corpus=False):
        """A line for each usage's hover: its corpus when grouped by word, its word when grouped by corpus."""
        if self.words is None:
            return None
        if self.group_by=='word':
            return None if hide_corpus else [f'Corpus: {c}' for c in self.corpora]
        return [f'Word: {w}' for w in self.words]

    def download_columns(self):
        """Columns for a plot's CSV: the dimension that is not the group."""
        if self.words is None:
            return {}
        return {'corpus':self.corpora} if self.group_by=='word' else {'word':self.words}


def words_mode():
    return st.session_state.get('words_mode','forms')


def word_mapping():
    return st.session_state.get('word_mapping') or {}


def loaded_words(data):
    """(the word of every loaded usage, the words in order), cached per data and mapping."""
    key=(id(data),len(data),json.dumps(word_mapping(),sort_keys=True))
    cached=st.session_state.get('_usage_words')
    if not cached or cached[0]!=key:
        cached=(key,usage_words(data,word_mapping()))
        st.session_state['_usage_words']=cached
    return cached[1]


def words_active(data):
    """Whether words are compared: "Different words" chosen, and 2–10 words among the loaded usages."""
    if data is None or words_mode()!='words' or len(data.forms)<2:
        return False
    return 2<=len(loaded_words(data)[1])<=MAX_WORDS


def group_by(data):
    return st.session_state.get('group_by','corpus') if words_active(data) else 'corpus'


def _shown(options,chosen):
    """The chosen names that are among the options, or all when none is."""
    kept=[o for o in options if chosen and o in chosen]
    return kept or list(options)


def current_view(data):
    """The usages to draw and compare, grouped as chosen. Without words, the loaded data itself."""
    if not words_active(data):
        return View(data,None,'corpus',data.corpus_list if data is not None else [])
    words,word_order=loaded_words(data)
    from app.common import ordered_labels
    corpus_order=ordered_labels(data)
    by=group_by(data)
    if by=='word':
        shown=set(_shown(corpus_order,st.session_state.get('corpora_shown')))
        keep=[i for i,u in enumerate(data.usages) if u.corpus in shown]
        labels,order=words,word_order
    else:
        shown=set(_shown(word_order,st.session_state.get('words_shown')))
        keep=[i for i,w in enumerate(words) if w in shown]
        labels,order=data.corpus_list,corpus_order
    key=(id(data),len(data),by,tuple(order),tuple(keep),json.dumps(word_mapping(),sort_keys=True))
    cached=st.session_state.get('_group_view')
    if cached and cached[0]==key:
        return cached[1]
    everything=len(keep)==len(data) and by=='corpus'
    view=View(data if everything else regroup(data,labels,order,keep),None if everything else keep,by,
              [data.usages[i].corpus for i in keep],[words[i] for i in keep])
    st.session_state['_group_view']=(key,view)
    return view


def _sync(name,widget_key,all_options=None):
    value=st.session_state[widget_key]
    if all_options is not None:
        # Every option chosen (or none) means "all", so words added later are shown too.
        value=None if not value or set(value)==set(all_options) else list(value)
    st.session_state[name]=value


def render_grouping_controls(data,key):
    """The switch between corpora and words, and the filter of the other one, at the top of a workspace."""
    if not words_active(data):
        return
    words,word_order=loaded_words(data)
    from app.common import ordered_labels
    corpus_order=ordered_labels(data)
    with st.container(border=True,key=f'grouping_{key}'):
        cols=st.columns([2,3])
        widget=f'_group_by_{key}'
        st.session_state[widget]=st.session_state.get('group_by','corpus')
        by=cols[0].radio('Colour and compare by',['corpus','word'],format_func={'corpus':'Corpus','word':'Word'}.get,horizontal=True,
                         key=widget,on_change=_sync,args=('group_by',widget),
                         help='**Corpus**: colours, legend, scores and frequencies compare the corpora. Show one word below to see how it is used across them.\n\n'
                              '**Word**: each word is treated like a corpus, so they compare the words. Show one corpus below to compare the words within it.\n\n'
                              'The choice applies in every workspace.')
        if by=='word':
            name,options,label='corpora_shown',corpus_order,'Corpora shown'
            hint='Only usages from these corpora are drawn and compared.'
        else:
            name,options,label='words_shown',word_order,'Words shown'
            hint='Only usages of these words are drawn and compared.'
        widget=f'_{name}_{key}'
        st.session_state[widget]=_shown(options,st.session_state.get(name))
        cols[1].multiselect(label,options,key=widget,on_change=_sync,args=(name,widget,options),help=hint+' Annotations are kept for every usage.')
