import json
import html
import re
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from streamlit.testing.v1 import AppTest
from semlens.data_loading import load_from_sentences
from semlens.embeddings import embedded_usages_from_precomputed
from semlens.spaces import project_to_definition_space


def seeded_app(corpora=3,definitions=3):
    at=AppTest.from_file('app/app.py',default_timeout=30).run()
    data=load_from_sentences([f'The bank had example {i}.' for i in range(corpora*4)],'bank',
                            [f'corpus_{i//4}' for i in range(corpora*4)])
    torch.manual_seed(3)
    vectors=torch.randn(len(data),8)
    defs=torch.randn(definitions,8)
    for k,v in {'target_word_data':data,'target_word':'bank','embedded_usages':embedded_usages_from_precomputed(data,vectors),
                'corpus_order':data.corpus_labels,'corpus_labels':data.corpus_labels,
                'definitions_raw':[f'sense {i}' for i in range(definitions)],
                'definitions_formatted':[f'bank: sense {i}' for i in range(definitions)],
                'definition_embeddings':defs,'def_space_all':project_to_definition_space(vectors,defs),'data_revision':1}.items():
        at.session_state[k]=v
    return at,data


def widget(at,kind,label):
    return next(w for w in getattr(at,kind) if w.label==label)


def plotted_points(at):
    fig=json.loads(at.get('plotly_chart')[0].proto.spec)
    return sum(len(trace.get('x',[])) for trace in fig['data'] if trace.get('name','').startswith('corpus_'))


def pair_tables(at):
    return [el.proto.body for el in at.get('html') if '<div class="semlens-pair-table semlens-compact">' in el.proto.body]


@pytest.mark.parametrize('count',[1,3,10])
def test_all_corpora_share_plot_and_single_corpus_has_no_change_scores(count):
    at,data=seeded_app(count)
    widget(at,'radio','Workspace').set_value('Explore').run()
    assert not at.exception
    assert plotted_points(at)==len(data)
    assert len(at.metric)==(0 if count==1 else 6)
    assert len(at.get('plotly_chart'))==(1 if count==1 else 5)
    if count>2:
        widget(at,'selectbox','From corpus').set_value('corpus_2').run()
        assert not at.exception
        assert plotted_points(at)==len(data)
        widget(at,'selectbox','Plot view').set_value('LDA').run()
        assert not at.exception
        assert plotted_points(at)==len(data)


def test_definition_pc_axis_distribution_and_scalar_scores():
    at,data=seeded_app(3,definitions=1)
    widget(at,'radio','Workspace').set_value('Definitions').run()
    assert not at.exception
    assert any(float(m.value)>0 for m in at.metric if m.label=='AMD')
    widget(at,'selectbox','Plot view').set_value('Definition axes').run()
    assert not at.exception
    assert widget(at,'selectbox','Vertical axis').value=='PC1'
    assert plotted_points(at)==len(data)
    at.session_state['definition_distribution']=True
    at.run()
    assert not at.exception
    assert len(at.get('plotly_chart'))==2
    at.session_state['defs_perdef']=True
    at.run()
    assert not at.exception
    values=at.dataframe[-1].value
    assert values['AMD'].max()>0
    # Scores come straight after the capped definition column, under readable names.
    assert list(values.columns[:5])==['Definition','APD','PRT','AMD','SAMD']
    assert values['PRT'].iloc[0]==pytest.approx(abs(values.iloc[0,-1]))


def test_distribution_lines_share_bins_and_preserve_counts_and_proportions():
    at,data=seeded_app(3)
    keep=[0,1,4,5,6,8,9,10,11]
    data=data.subset(keep)
    at.session_state['target_word_data']=data
    at.session_state['embedded_usages']=embedded_usages_from_precomputed(data,at.session_state['embedded_usages'].embeddings[keep])
    # Unequal corpus sizes and both distance endpoints exercise normalisation
    # and the inclusive rightmost bin when replacing Plotly's histogramming.
    values=torch.tensor([0.,2.,.1,.1,.1,1.,1.,1.,1.])
    at.session_state['def_space_all']=torch.stack([values,2-values,values],dim=1)
    at.session_state['definition_distribution']=True
    widget(at,'radio','Workspace').set_value('Definitions').run()
    assert not at.exception
    fig=json.loads(at.get('plotly_chart')[1].proto.spec)
    traces=fig['data']
    assert len(traces)==3
    assert all(t['type']=='scatter' and t['mode']=='lines+markers' and t['fill']=='none' for t in traces)
    # The smallest corpus has two usages, so the minimum of eight shared bins applies.
    assert all(t['x']==traces[0]['x'] and len(t['x'])==8 for t in traces)
    assert fig['layout']['xaxis']['range']==pytest.approx([-.08,2.08])
    for trace,size in zip(traces,[2,3,4]):
        assert sum(trace['y'])==pytest.approx(1)
        assert sum(item[2] for item in trace['customdata'])==size
    assert traces[0]['y'][0]==pytest.approx(.5) and traces[0]['y'][-1]==pytest.approx(.5)
    assert fig['layout']['hovermode']=='x unified'
    at.session_state['definition_distribution']=True
    widget(at,'checkbox','Normalise within each corpus').uncheck().run()
    assert not at.exception
    counts=json.loads(at.get('plotly_chart')[1].proto.spec)['data']
    for trace,size in zip(counts,[2,3,4]):
        assert sum(trace['y'])==size
    assert list(at.dataframe[0].value['Occurrences'])==[2,3,4]


def test_distribution_bins_follow_the_observed_distances():
    at,data=seeded_app(2)
    # Real usage-to-definition distances sit in a narrow band, far from the full 0–2 range.
    values=torch.linspace(.02,.28,len(data))
    at.session_state['def_space_all']=torch.stack([values,values.flip(0),values],dim=1)
    at.session_state['definition_distribution']=True
    widget(at,'radio','Workspace').set_value('Definitions').run()
    assert not at.exception
    fig=json.loads(at.get('plotly_chart')[1].proto.spec)
    for trace in fig['data']:
        edges=[item[:2] for item in trace['customdata']]
        assert edges[0][0]==pytest.approx(.02) and edges[-1][1]==pytest.approx(.28)
        assert sum(item[2] for item in trace['customdata'])==4
    assert fig['layout']['xaxis']['range'][1]<.3
    # A definition every usage is equally far from still draws.
    at.session_state['def_space_all']=torch.full((len(data),3),.4)
    at.session_state['definition_distribution']=True
    at.run()
    assert not at.exception


def test_definition_axes_default_and_selected_coordinates_with_anchors():
    at,data=seeded_app(3,definitions=3)
    widget(at,'radio','Workspace').set_value('Definitions').run()
    assert not at.exception
    assert widget(at,'selectbox','Plot view').value=='Definition axes'
    assert widget(at,'selectbox','Horizontal axis').value=='Definition 1: sense 0'
    assert widget(at,'selectbox','Vertical axis').value=='Definition 2: sense 1'
    space=at.session_state['def_space_all'].numpy()

    def coordinates():
        fig=json.loads(at.get('plotly_chart')[0].proto.spec)
        traces=[t for t in fig['data'] if t.get('name','').startswith('corpus_')]
        return fig,np.concatenate([np.column_stack([t['x'],t['y']]) for t in traces])

    fig,actual=coordinates()
    np.testing.assert_allclose(actual,space[:,[0,1]])
    widget(at,'selectbox','Horizontal axis').set_value('Definition 3: sense 2')
    widget(at,'selectbox','Vertical axis').set_value('Definition 1: sense 0')
    widget(at,'checkbox','Show definition anchors').check().run()
    assert not at.exception
    fig,actual=coordinates()
    np.testing.assert_allclose(actual,space[:,[2,0]])
    assert 'sense 2' in fig['layout']['xaxis']['title']['text']
    assert 'sense 0' in fig['layout']['yaxis']['title']['text']
    anchors=next(t for t in fig['data'] if t.get('name','').startswith('Definitions'))
    # Short labels in the theme's own text colour; full definitions in the hover.
    assert anchors['text']==['D1','D2','D3'] and 'color' not in anchors['textfont']
    assert anchors['hovertext'][1]=='Definition 2: sense 1'
    defs=at.session_state['definition_embeddings']
    expected=project_to_definition_space(defs,defs).numpy()[:,[2,0]]
    np.testing.assert_allclose(np.column_stack([anchors['x'],anchors['y']]),expected)
    widget(at,'selectbox','Vertical axis').set_value('PC1').run()
    fig,actual=coordinates()
    from app.ui_explore import _pca_axes
    pcs,_=_pca_axes(space)
    np.testing.assert_allclose(actual[:,0],space[:,2])
    np.testing.assert_allclose(actual[:,1],pcs[:,0])
    assert plotted_points(at)==len(data)


def test_existing_choose_axes_selection_migrates_to_definition_axes():
    at,_=seeded_app(2)
    at.session_state['defs_view']='Choose axes'
    widget(at,'radio','Workspace').set_value('Definitions').run()
    assert not at.exception
    assert widget(at,'selectbox','Plot view').value=='Definition axes'


def test_fetched_definitions_can_be_edited_applied_and_kept_across_navigation(monkeypatch):
    import app.ui_definitions as ui
    monkeypatch.setattr(ui,'fetch_wiktionary_entries',lambda *a,**kw:[('Noun','A financial institution'),('Noun','The edge of a river')])
    monkeypatch.setattr(ui,'get_model',lambda *a:SimpleNamespace(name='test/encoder'))
    embedded=[]
    def embed(model,word,definitions,**kwargs):
        embedded.append(list(definitions))
        return torch.eye(8)[:len(definitions)]
    monkeypatch.setattr(ui,'embed_definitions',embed)
    at,_=seeded_app(2)
    at.session_state['embedded_usages'].metadata['model_id']='test/encoder'
    widget(at,'radio','Workspace').set_value('Definitions').run()
    widget(at,'selectbox','Definition source').set_value('Wiktionary').run()
    widget(at,'button','Fetch definitions').click().run()
    editor=widget(at,'text_area','Edit definitions (one per line)')
    assert editor.value=='A financial institution\nThe edge of a river'
    draft='  A place that holds money  \n\nA bank of switches\n'
    editor.set_value(draft).run()
    assert at.session_state['definitions_raw']==['sense 0','sense 1','sense 2']
    widget(at,'button','Create definition space').click().run()
    assert not at.exception and not at.error
    assert embedded==[['bank: A place that holds money','bank: A bank of switches']]
    assert at.session_state['definitions_raw']==['A place that holds money','A bank of switches']
    assert at.session_state['def_space_all'].shape==(8,2)
    at.run()  # Creating a space increments the data revision.
    assert widget(at,'text_area','Edit definitions (one per line)').value==draft
    widget(at,'radio','Workspace').set_value('Explore').run()
    widget(at,'radio','Workspace').set_value('Definitions').run()
    widget(at,'selectbox','Definition source').set_value('Wiktionary').run()
    assert widget(at,'text_area','Edit definitions (one per line)').value==draft
    # A failed refresh must preserve edits; a successful explicit refresh replaces them.
    def fail(*a,**kw):
        raise ValueError('Dictionary unavailable')
    monkeypatch.setattr(ui,'fetch_wiktionary_entries',fail)
    widget(at,'button','Fetch definitions').click().run()
    assert widget(at,'text_area','Edit definitions (one per line)').value==draft
    assert at.error[0].value=='Dictionary unavailable'
    monkeypatch.setattr(ui,'fetch_wiktionary_entries',lambda *a,**kw:[('Noun','Refetched definition')])
    widget(at,'button','Fetch definitions').click().run()
    assert not at.exception
    assert widget(at,'text_area','Edit definitions (one per line)').value=='Refetched definition'


def test_definition_drafts_are_separate_for_dictionary_languages(monkeypatch):
    import app.ui_definitions as ui
    monkeypatch.setattr(ui,'fetch_wiktionary_entries',lambda *a,**kw:[])
    at,_=seeded_app(2)
    widget(at,'radio','Workspace').set_value('Definitions').run()
    widget(at,'selectbox','Definition source').set_value('Wiktionary').run()
    widget(at,'button','Fetch definitions').click().run()
    widget(at,'text_area','Edit definitions (one per line)').set_value('My own definition').run()
    widget(at,'text_input','Language section heading').set_value('German').run()
    assert not at.text_area
    widget(at,'button','Fetch definitions').click().run()
    assert widget(at,'text_area','Edit definitions (one per line)').value==''
    widget(at,'text_input','Language section heading').set_value('English').run()
    assert widget(at,'text_area','Edit definitions (one per line)').value=='My own definition'
    assert not at.exception


def test_amd_evidence_and_repeated_sampling_are_optional():
    at,data=seeded_app(3)
    widget(at,'radio','Workspace').set_value('Explore').run()
    at.session_state['full_pairs']=True
    at.run()
    assert not at.exception
    assert len(pair_tables(at))==2
    widget(at,'selectbox','Score sampling').set_value('Repeated equal-size samples')
    # AppTest does not yet serialise stateful expander frontend state.
    at.session_state['full_pairs']=True
    at.run()
    assert not at.exception
    assert len(at.dataframe)==1
    assert len(pair_tables(at))==2
    assert plotted_points(at)==len(data)


def test_ranked_sentence_search_and_independently_scrollable_cells():
    at,_=seeded_app(2)
    widget(at,'radio','Workspace').set_value('Explore').run()
    at.session_state['full_pairs']=True
    at.run()
    assert not at.exception
    widget(at,'text_input','Find a sentence').set_value('example 0.')
    at.session_state['full_pairs']=True
    at.run()
    assert not at.exception
    assert pair_tables(at)
    for table in pair_tables(at):
        rows=[]
        for row in re.findall(r'<tr>(.*?)</tr>',table):
            cells=re.findall(r'<td>(.*?)</td>',row)
            if cells:
                rows.append([html.unescape(re.sub('<[^>]+>','',cell)) for cell in cells])
        assert rows
        assert all('example 0.' in row[1] or 'example 0.' in row[2] for row in rows)
        distances=[float(row[0]) for row in rows]
        assert distances==sorted(distances,reverse=True)
        assert table.count('class="semlens-sentence-cell"')==2*len(rows)
        assert table.count('tabindex="0"')==2*len(rows)


def test_theme_and_destination_navigation():
    at,_=seeded_app(2)
    assert widget(at,'selectbox','Data source').value=='Upload raw corpora (.txt)'
    assert widget(at,'radio','Theme').value=='Dark'
    widget(at,'radio','Theme').set_value('Dark').run()
    widget(at,'button','Continue to Explore').click().run()
    assert not at.exception
    assert widget(at,'radio','Workspace').value=='Explore'
    fig=json.loads(at.get('plotly_chart')[0].proto.spec)
    assert fig['layout']['paper_bgcolor']=='#0e1117'
    widget(at,'radio','Theme').set_value('Light').run()
    fig=json.loads(at.get('plotly_chart')[0].proto.spec)
    assert fig['layout']['paper_bgcolor']=='#ffffff'
    assert fig['layout']['plot_bgcolor']=='#ffffff'
    for label,destination in [('Continue to Definitions','Definitions'),('Continue to Annotation','Annotation'),('Return to Data & Model','Data & Model')]:
        widget(at,'button',label).click().run()
        assert not at.exception
        assert widget(at,'radio','Workspace').value==destination


def test_annotation_sense_buttons_and_default_click_selection():
    at,_=seeded_app(2)
    widget(at,'radio','Workspace').set_value('Annotation').run()
    for sense in ['finance','river']:
        widget(at,'text_input','New sense class').set_value(sense)
        widget(at,'button','Add class').click().run()
        assert not at.exception
        assert widget(at,'button',f'✓ {sense}')
    widget(at,'button','finance').click().run()
    assert widget(at,'button','✓ finance')
    assert not any(w.label in {'Active sense class','Active annotation class'} for w in at.selectbox)
    assert not any(w.label=='Assign selected' for w in at.button)
    assert widget(at,'radio','Selection').value=='Click'
    chart=at.get('plotly_chart')[0]
    layout=json.loads(chart.proto.spec)['layout']
    assert layout['clickmode']=='event+select'
    assert list(chart.proto.selection_mode)==[0]  # PlotlyChart.POINTS
    widget(at,'radio','Selection').set_value('Lasso').run()
    assert json.loads(at.get('plotly_chart')[0].proto.spec)['layout']['dragmode']=='lasso'


@pytest.mark.parametrize('representation,label',[('word','Target word'),('sentence','Whole input')])
def test_definition_readout_follows_usage_representation(representation,label):
    at,_=seeded_app(2)
    at.session_state['embedded_usages'].metadata['representation']=representation
    widget(at,'radio','Workspace').set_value('Definitions').run()
    assert widget(at,'selectbox','Representation to extract').value==label
    at.session_state['definition_heatmaps']=True
    at.run()
    assert not at.exception
    assert len(at.get('plotly_chart'))==3
    heatmap=json.loads(at.get('plotly_chart')[1].proto.spec)
    assert heatmap['data'][0]['x']==['D1','D2','D3']
    assert heatmap['data'][0]['customdata'][0][0]==['sense 0','sense 0']


@pytest.mark.parametrize('representation',['word','sentence'])
def test_save_only_word_embeddings_and_reopen_without_encoder(tmp_path,monkeypatch,representation):
    import app.ui_data as ui
    from semlens.projects import load_project
    monkeypatch.setattr(ui,'STORAGE',tmp_path)
    at,data=seeded_app(2)
    embedded=at.session_state['embedded_usages']
    embedded.metadata.update(model_id='test/encoder',layer=3,representation=representation)
    at.session_state['source_texts']={'large_corpus':'This unselected corpus text should not be copied.'}
    at.run()
    assert widget(at,'radio','Embedding save scope').value=='Whole corpus'
    widget(at,'radio','Embedding save scope').set_value('Current word only').run()
    widget(at,'button','Prepare word embedding file').click().run()
    assert not at.exception
    assert any('File size before saving' in message.value for message in at.info)
    widget(at,'button','Save word embeddings locally').click().run()
    assert not at.exception
    paths=list((tmp_path/'words').glob('*.semlens'))
    assert len(paths)==1
    saved=load_project(paths[0])
    assert saved['source_texts']=={}
    assert saved['data'].sentences==data.sentences
    assert torch.equal(saved['embedded'].embeddings,embedded.embeddings)
    assert saved['embedded'].metadata==embedded.metadata
    fresh=AppTest.from_file('app/app.py',default_timeout=30).run()
    widget(fresh,'selectbox','Data source').set_value('Saved word embeddings').run()
    widget(fresh,'button','Open saved word embeddings').click().run()
    assert not fresh.exception
    assert fresh.session_state['embedded_usages'].metadata['representation']==representation
    assert fresh.session_state['target_word_data'].sentences==data.sentences


def test_auto_annotation_consumes_events_once_and_undo_survives_rerun(monkeypatch):
    import app.ui_annotations as ui
    import app.annotation_store as store
    fake=SimpleNamespace(session_state={})
    monkeypatch.setattr(ui,'st',fake)
    monkeypatch.setattr(store,'st',fake)
    result={'selection_event':('chart',1),'clicked_points':[2,4]}
    assert ui.assign_new_selection(result,'finance')
    assert store.get_annotations()=={2:'finance',4:'finance'}
    assert not ui.assign_new_selection(result,'river')
    store.undo_annotations()
    assert store.get_annotations()=={}
    assert not ui.assign_new_selection(result,'finance')
    result['selection_event']=('chart',2)
    assert ui.assign_new_selection(result,'river')
    assert store.get_annotations()=={2:'river',4:'river'}


def test_input_loads_forms_from_all_three_pasted_corpora():
    at=AppTest.from_file('app/app.py',default_timeout=30).run()
    widget(at,'selectbox','Data source').set_value('Paste sentences').run()
    widget(at,'number_input','Number of corpora').set_value(3).run()
    widget(at,'text_input','Target word').set_value('run')
    widget(at,'text_input','Other forms (optional)').set_value('runs, running, runner')
    for i,area in enumerate(at.text_area):
        area.set_value(['They run.','He runs.','A runner is running.'][i])
    widget(at,'button','Load usages').click().run()
    assert not at.exception
    data=at.session_state['target_word_data']
    assert len(data)==4
    assert len(data.corpus_labels)==3


def test_annotation_invalidation_preserves_labels_on_reembedding(monkeypatch):
    import app.state as state
    import app.annotation_store as store
    class State(dict):
        def __getattr__(self,name):return self[name]
        def __setattr__(self,name,value):self[name]=value
    session=State()
    fake=SimpleNamespace(session_state=session)
    monkeypatch.setattr(state,'st',fake)
    monkeypatch.setattr(store,'st',fake)
    state.init_state()
    data=load_from_sentences(['The bank is open.','The bank is closed.'],'bank',['old','new'])
    state.set_data(data)
    store.add_sense_class('finance')
    store.set_annotation(0,'finance')
    session['_annot_coords_cache']={'old':np.ones((2,2))}
    old_revision=session.data_revision
    state.reset_downstream_of('embeddings')
    assert store.get_annotations()=={0:'finance'}
    assert session.data_revision>old_revision
    assert '_annot_coords_cache' not in session
    new=load_from_sentences(['The river bank is steep.','The bank is closed.'],'bank',['old','new'])
    state.set_data(new)
    assert store.get_annotations()=={}


def test_annotation_overlay_selection_does_not_label_unrelated_point(monkeypatch):
    import app.components.scatter as scatter
    event={'selection':{'points':[{'customdata':8,'point_index':0,'curve_number':3}],
                        'point_indices':[0],'lasso':[],'box':[]}}
    monkeypatch.setattr(scatter.st,'plotly_chart',lambda *a,**k:event)
    result=scatter.render_scatter(np.zeros((10,2)),['old']*5+['new']*5,['The bank is open.']*10,'bank',
        enable_selection=True,annotations={8:'sense'},sense_classes=['sense'])
    assert result['clicked_points']==[8]


def test_hover_wrapping_keeps_exact_occurrence_and_escapes_corpus_text():
    from semlens.utils import format_hover_text
    import html
    import re
    sentence='bank <img src=x onerror=alert(1)> '+('context '*14)+'the bank is open & welcoming.'
    start=sentence.rfind('bank')
    hover=format_hover_text(sentence,'bank','<old>',max_line_length=35,
                            nearest_definition='A financial institution '+('definition '*15),
                            char_span=(start,start+4))
    assert hover.count('<b>bank</b>')==1
    assert '<img' not in hover and '&lt;img' in hover
    assert '[&lt;old&gt;]' in hover
    assert 'Nearest definition:' in hover
    assert ' '.join(html.unescape(re.sub('<[^>]+>',' ',hover)).split()).find('the bank is open & welcoming.')>=0
    for line in hover.split('<br>'):
        assert len(html.unescape(re.sub('<[^>]+>','',line)))<=37  # corpus brackets


def test_hover_handles_unbroken_tokens_and_targets_crossing_wrap_boundaries():
    from semlens.utils import wrap_hover_html
    assert wrap_hover_html('x'*20,8)==('x'*8+'<br>'+'x'*8+'<br>'+'x'*4)
    assert wrap_hover_html('abcdefghij',5,[(3,8)])=='abc<b>de</b><br><b>fgh</b>ij'
    assert wrap_hover_html('a\n\nb',8)=='a<br><br>b'


@pytest.mark.parametrize('mode',['balanced','unequal','repeated'])
def test_all_pair_heatmap_scores_follow_comparisons_and_corpus_reordering(mode):
    from app.common import cached_corpus_scores,cached_comparison
    array=np.random.default_rng(3).normal(size=(9,5)).astype('float32')
    indices=((0,1),(2,3,4),(5,6,7,8))
    labels=('old','new','middle')
    matrices,counts=cached_corpus_scores(array,indices,labels,'cosine',mode,5,8)
    for i,j in [(0,1),(0,2),(1,2)]:
        pair=cached_comparison(array[list(indices[i])],array[list(indices[j])],(labels[i],labels[j]),'cosine',mode,5,8)
        reverse=cached_comparison(array[list(indices[j])],array[list(indices[i])],(labels[j],labels[i]),'cosine',mode,5,8)
        for metric in ('apd','prt','amd','samd'):
            expected=pair.summary.loc[metric,'mean']
            assert matrices[metric]['mean'][i,j]==pytest.approx(expected)
            assert matrices[metric]['mean'][j,i]==pytest.approx(expected)
            assert reverse.summary.loc[metric,'mean']==pytest.approx(expected)
        assert counts[i,j]==len(pair.draws[0][0])
        assert counts[j,i]==len(pair.draws[0][1])
    permutation=[2,0,1]
    reordered,_=cached_corpus_scores(array,tuple(indices[i] for i in permutation),tuple(labels[i] for i in permutation),'cosine',mode,5,8)
    for metric in matrices:
        assert np.isnan(np.diag(matrices[metric]['mean'])).all()
        np.testing.assert_allclose(reordered[metric]['mean'],matrices[metric]['mean'][np.ix_(permutation,permutation)],equal_nan=True)


def test_definition_heatmaps_adapt_to_observed_values():
    from app.ui_definitions import definition_heatmap
    matrix=np.array([[1,.87],[.87,1]])
    chart=definition_heatmap(matrix,['a','b'],'similarity').data[0]
    assert chart.zauto is True
    assert chart.zmin is None and chart.zmax is None
    np.testing.assert_array_equal(chart.z,matrix)


def test_annotation_review_excludes_unannotated_by_default():
    at,data=seeded_app(2)
    at.session_state['_annotation_store']={'annotations':{0:'finance'},'sense_classes':['finance'],
                                          'active_sense_class':'finance','history':[]}
    widget(at,'radio','Workspace').set_value('Annotation').run()
    assert not at.exception
    assert widget(at,'checkbox','Include unannotated occurrences').value is False
    assert len(at.dataframe[-1].value)==1
    widget(at,'checkbox','Include unannotated occurrences').check().run()
    assert len(at.dataframe[-1].value)==len(data)


def test_sentence_pair_cells_keep_full_text_without_html_execution():
    import pandas as pd
    from app.presentation import sentence_pair_html
    sentence='<img src=x onerror=alert(1)> '+('long text '*200)+'THE END'
    markup=sentence_pair_html(pd.DataFrame([[.2,sentence,'Another sentence']],columns=['Distance','Usage','Closest counterpart']))
    assert '<img' not in markup
    assert html.escape(sentence) in markup
    assert '<col style="width:90px"><col><col>' in markup
    assert markup.count('class="semlens-sentence-cell"')==2


@pytest.mark.parametrize('mode,scope,open_label',[
    ('Saved analysis','projects','Open saved analysis'),
    ('Saved word embeddings','words','Open saved word embeddings')])
def test_manage_saved_files_rename_collision_cancel_and_delete(tmp_path,monkeypatch,mode,scope,open_label):
    import app.ui_data as ui
    from semlens.projects import project_bytes,save_project
    monkeypatch.setattr(ui,'STORAGE',tmp_path)
    at,data=seeded_app(2)
    embedded=at.session_state['embedded_usages']
    content=project_bytes(data,embedded,annotations={0:'finance'},sense_classes=['finance'])
    original=tmp_path/scope/'original.semlens'
    other=tmp_path/scope/'other.semlens'
    save_project(original,content)
    save_project(other,content)
    widget(at,'selectbox','Data source').set_value(mode).run()
    widget(at,'selectbox','Saved locally').set_value(original).run()
    widget(at,'button',open_label).click().run()
    assert not at.exception
    widget(at,'text_input','New name').set_value('other')
    widget(at,'button','Rename').click().run()
    assert at.error and original.read_bytes()==content and other.read_bytes()==content
    widget(at,'text_input','New name').set_value('Renamed bank study')
    widget(at,'button','Rename').click().run()
    renamed=original.with_name('Renamed bank study.semlens')
    assert not at.exception
    assert renamed.read_bytes()==content and not original.exists()
    assert widget(at,'selectbox','Saved locally').value==renamed
    if scope=='projects':
        assert widget(at,'text_input','Analysis name').value=='Renamed bank study'
    widget(at,'button','Delete saved file…').click().run()
    assert renamed.exists()
    widget(at,'button','Cancel').click().run()
    assert renamed.exists() and not any(b.label=='Confirm deletion' for b in at.button)
    widget(at,'button','Delete saved file…').click().run()
    # Selecting another file must cancel the pending deletion.
    widget(at,'selectbox','Saved locally').set_value(other).run()
    assert not any(b.label=='Confirm deletion' for b in at.button)
    widget(at,'selectbox','Saved locally').set_value(renamed).run()
    widget(at,'button','Delete saved file…').click().run()
    widget(at,'button','Confirm deletion').click().run()
    assert not at.exception
    assert not renamed.exists() and other.read_bytes()==content
    assert at.session_state['target_word_data'].sentences==data.sentences
    assert at.session_state['_annotation_store']['annotations']=={0:'finance'}
    torch.testing.assert_close(at.session_state['embedded_usages'].embeddings,embedded.embeddings)
    widget(at,'button','Delete saved file…').click().run()
    widget(at,'button','Confirm deletion').click().run()
    assert not at.exception and not list((tmp_path/scope).glob('*.semlens'))
    assert any('saved locally yet' in message.value for message in at.info)


def test_open_edit_and_save_analysis_uses_current_name_and_protects_other_files(tmp_path,monkeypatch):
    import app.ui_data as ui
    from semlens.projects import project_bytes,save_project,load_project
    monkeypatch.setattr(ui,'STORAGE',tmp_path)
    at,data=seeded_app(2)
    content=project_bytes(data,at.session_state['embedded_usages'])
    original=tmp_path/'projects'/'Original study.semlens'
    other=tmp_path/'projects'/'Other study.semlens'
    save_project(original,content)
    save_project(other,content)
    widget(at,'selectbox','Data source').set_value('Saved analysis').run()
    widget(at,'selectbox','Saved locally').set_value(original).run()
    widget(at,'button','Open saved analysis').click().run()
    assert widget(at,'text_input','Analysis name').value=='Original study'
    at.session_state['_annotation_store']={'annotations':{0:'finance'},'sense_classes':['finance'],
                                          'active_sense_class':'finance','history':[]}
    widget(at,'button','Prepare analysis file').click().run()
    widget(at,'button','Save analysis locally').click().run()
    assert not at.exception
    assert load_project(original)['annotations']=={0:'finance'}
    # A changed name after preparation must be honoured, with no second encode/prepare.
    widget(at,'text_input','Analysis name').set_value('Other study').run()
    assert widget(at,'button','Save analysis locally').disabled
    assert other.read_bytes()==content
    widget(at,'text_input','Analysis name').set_value('Separate copy').run()
    widget(at,'button','Save analysis locally').click().run()
    assert not at.exception
    copy=tmp_path/'projects'/'Separate copy.semlens'
    assert load_project(copy)['annotations']=={0:'finance'}
    assert original.exists() and other.read_bytes()==content
    # A fresh session can reopen the renamed copy with its edited annotations.
    fresh=AppTest.from_file('app/app.py',default_timeout=30).run()
    widget(fresh,'selectbox','Data source').set_value('Saved analysis').run()
    widget(fresh,'selectbox','Saved locally').set_value(copy).run()
    widget(fresh,'button','Open saved analysis').click().run()
    assert not fresh.exception
    assert fresh.session_state['_annotation_store']['annotations']=={0:'finance'}


def scatter_figure(at):
    return json.loads(at.get('plotly_chart')[0].proto.spec)


def corpus_visibility(fig):
    return {t['name']:t.get('visible',True) for t in fig['data'] if (t.get('meta') or {}).get('semlens_corpus')}


def test_corpora_switched_off_in_the_legend_stay_off_across_views():
    at,data=seeded_app(3)
    at.session_state['hidden_corpora']=['corpus_1','not loaded']
    widget(at,'radio','Workspace').set_value('Explore').run()
    assert not at.exception
    expected={'corpus_0':True,'corpus_1':'legendonly','corpus_2':True}
    assert corpus_visibility(scatter_figure(at))==expected
    for view in ['t-SNE','LDA','PCA']:
        widget(at,'selectbox','Plot view').set_value(view).run()
        assert not at.exception
        assert corpus_visibility(scatter_figure(at))==expected
    # Still every corpus in the figure, so the legend can switch it back on.
    assert plotted_points(at)==len(data)
    widget(at,'radio','Workspace').set_value('Definitions').run()
    widget(at,'selectbox','Vertical axis').set_value('Definition 3: sense 2').run()
    assert corpus_visibility(scatter_figure(at))==expected
    widget(at,'selectbox','Highlight nearest definition').set_value('sense 0').run()
    fig=scatter_figure(at)
    hidden_xy={(x,y) for t in fig['data'] if t['name']=='corpus_1' for x,y in zip(t['x'],t['y'])}
    highlighted=next((t for t in fig['data'] if t['name']=='Highlighted'),None)
    assert highlighted is None or not hidden_xy&set(zip(highlighted['x'],highlighted['y']))
    widget(at,'radio','Workspace').set_value('Annotation').run()
    assert corpus_visibility(scatter_figure(at))==expected


def test_annotation_plot_has_selectable_corpus_traces_and_skips_hidden_corpora(monkeypatch):
    import app.components.scatter as scatter
    captured={}
    # A lasso drawn over everything, reported without explicit points.
    event={'selection':{'points':[],'point_indices':[],'box':[],
                        'lasso':[{'x':[-1,3,3,-1],'y':[-1,-1,3,3]}]}}
    def chart(fig,**kwargs):
        captured['fig']=fig
        return event
    monkeypatch.setattr(scatter.st,'plotly_chart',chart)
    coords=np.column_stack([np.linspace(0,2,6),np.linspace(0,2,6)])
    corpora=['old','old','old','new','new','new']
    arguments=dict(enable_selection=True,interaction_mode='click',annotations={1:'sense',4:'sense'},sense_classes=['sense'])
    scatter.render_scatter(coords,corpora,['The bank is open.']*6,'bank',**arguments)
    traces={t.name:t for t in captured['fig'].data}
    assert list(traces['old'].customdata)==[0,1,2] and list(traces['new'].customdata)==[3,4,5]
    assert list(traces['_hit_area'].customdata)==[0,1,2,3,4,5]
    assert list(traces['⬡ sense'].customdata)==[1,4]
    event['selection']={'points':[{'customdata':4,'point_index':1,'curve_number':2}],'point_indices':[1],'lasso':[],'box':[]}
    assert scatter.render_scatter(coords,corpora,['The bank is open.']*6,'bank',**arguments)['clicked_points']==[4]
    # With a corpus switched off, its points, click targets and sense markers go,
    # and no selection (including the geometric fallback) can reach them.
    event['selection']={'points':[],'point_indices':[],'box':[],'lasso':{'x':[-1,3,3,-1],'y':[-1,-1,3,3]}}
    result=scatter.render_scatter(coords,corpora,['The bank is open.']*6,'bank',hidden_corpora=['new'],**arguments)
    traces={t.name:t for t in captured['fig'].data}
    assert traces['new'].visible=='legendonly' and traces['old'].visible is True
    assert list(traces['_hit_area'].customdata)==[0,1,2]
    assert list(traces['⬡ sense'].customdata)==[1]
    assert result['clicked_points']==[0,1,2]


def test_blind_annotation_hides_corpus_colour_and_hover_but_keeps_legend():
    at,data=seeded_app(3)
    at.session_state['hidden_corpora']=['corpus_2']
    widget(at,'radio','Workspace').set_value('Annotation').run()
    assert 'LDA' in widget(at,'selectbox','Plot view').options
    def corpus_traces():
        return [t for t in scatter_figure(at)['data'] if (t.get('meta') or {}).get('semlens_corpus')]
    assert len({t['marker']['color'] for t in corpus_traces()})==3
    assert all(t['text'][0].startswith('<b>[corpus_') for t in corpus_traces())
    widget(at,'selectbox','Plot view').set_value('LDA').run()
    widget(at,'toggle','Blind annotation').set_value(True).run()
    assert not at.exception
    traces=corpus_traces()
    assert [t['name'] for t in traces]==['corpus_0','corpus_1','corpus_2']
    assert len({t['marker']['color'] for t in traces})==1
    assert not any('corpus_' in text for t in scatter_figure(at)['data'] for text in t.get('text') or [])
    assert traces[2]['visible']=='legendonly'
    # LDA arranges usages by corpus, so it is withdrawn while blind.
    view=widget(at,'selectbox','Plot view')
    assert 'LDA' not in view.options and view.value=='PCA'
    assert sum(len(t['x']) for t in traces)==len(data)


def test_inspector_chooses_pair_and_finds_closest_usages_in_other_corpora():
    at,data=seeded_app(3)
    widget(at,'radio','Workspace').set_value('Explore').run()
    at.session_state['full_pairs']=True
    at.run()
    assert not at.exception
    assert widget(at,'selectbox','Inspect usages from').value=='corpus_0'
    assert widget(at,'selectbox','Closest counterparts in').value=='corpus_1'
    # Choosing the pair inside the panel moves the comparison above with it.
    widget(at,'selectbox','Closest counterparts in').set_value('corpus_2')
    at.session_state['full_pairs']=True
    at.run()
    assert not at.exception
    assert widget(at,'selectbox','To corpus').value=='corpus_2'
    assert any(m.label=='AMD: corpus_0 → corpus_2' for m in at.metric)
    # Picking the other side's corpus swaps the pair instead of failing.
    widget(at,'selectbox','Inspect usages from').set_value('corpus_2')
    at.session_state['full_pairs']=True
    at.run()
    assert not at.exception
    assert (widget(at,'selectbox','From corpus').value,widget(at,'selectbox','To corpus').value)==('corpus_2','corpus_0')
    assert len(pair_tables(at))==2
    widget(at,'radio','Show').set_value('Closest usages to a chosen sentence')
    at.session_state['full_pairs']=True
    at.run()
    assert not at.exception
    widget(at,'text_input','Find the sentence to look up').set_value('example 5.')
    widget(at,'number_input','Closest per corpus').set_value(2)
    at.session_state['full_pairs']=True
    at.run()
    assert not at.exception
    table=next(el.proto.body for el in at.get('html') if 'semlens-nearest-table' in el.proto.body)
    rows=[[html.unescape(re.sub('<[^>]+>','',cell)) for cell in re.findall(r'<td>(.*?)</td>',row)]
          for row in re.findall(r'<tr>(.*?)</tr>',table) if '<td>' in row]
    # Usage 5 belongs to corpus_1: two nearest usages from each of the other two corpora.
    assert [row[0] for row in rows]==['corpus_0','corpus_0','corpus_2','corpus_2']
    vectors=at.session_state['embedded_usages'].embeddings
    cosine=1-torch.nn.functional.cosine_similarity(vectors[5:6],vectors)
    for corpus in ('corpus_0','corpus_2'):
        expected=sorted(float(cosine[i]) for i in data.indices_for_corpus(corpus))[:2]
        assert [float(row[1]) for row in rows if row[0]==corpus]==pytest.approx(expected,abs=1e-5)


def test_occurrence_cap_accepts_all_or_a_positive_number():
    from app.ui_data import _parse_cap
    assert [_parse_cap(text) for text in ['all',' All ','','0','500','1,000']]==[0,0,0,0,500,1000]
    for text in ['-5','many','2.5']:
        with pytest.raises(ValueError,match='“all” or a positive whole number'):
            _parse_cap(text)
    at=AppTest.from_file('app/app.py',default_timeout=30).run()
    widget(at,'selectbox','Data source').set_value('Paste sentences').run()
    assert widget(at,'text_input','Maximum occurrences per corpus').value=='all'
    widget(at,'text_input','Target word').set_value('bank')
    at.text_area[0].set_value('\n'.join(f'The bank had example {i}.' for i in range(6)))
    at.text_area[1].set_value('\n'.join(f'A bank closed in case {i}.' for i in range(4)))
    widget(at,'button','Load usages').click().run()
    assert len(at.session_state['target_word_data'])==10
    widget(at,'text_input','Maximum occurrences per corpus').set_value('3')
    widget(at,'button','Load usages').click().run()
    data=at.session_state['target_word_data']
    assert [len(data.indices_for_corpus(c)) for c in data.corpus_labels]==[3,3]
    widget(at,'text_input','Maximum occurrences per corpus').set_value('lots')
    widget(at,'button','Load usages').click().run()
    assert any('positive whole number' in e.value for e in at.error)
    assert len(at.session_state['target_word_data'])==6


def test_help_icon_styling_leaves_buttons_with_help_text_alone():
    at,_=seeded_app(2)
    widget(at,'radio','Workspace').set_value('Annotation').run()
    css=''.join(el.proto.body for el in at.get('html'))
    assert '[data-testid="stTooltipIcon"] button:not([data-testid])' in css
    assert '[data-testid="stTooltipIcon"] button {' not in css


@pytest.mark.parametrize('count',[3,7,10])
def test_score_heatmaps_print_a_number_in_every_cell(count):
    at,_=seeded_app(count)
    widget(at,'radio','Workspace').set_value('Explore').run()
    assert not at.exception
    for chart in at.get('plotly_chart')[1:5]:
        heatmap=json.loads(chart.proto.spec)['data'][0]
        assert heatmap['texttemplate']=='%{text}'
        assert all(heatmap['text'][i][j] for i in range(count) for j in range(count) if i!=j)
        assert all(heatmap['text'][i][i]=='' for i in range(count))


def test_sidebar_shows_new_data_immediately_and_keeps_the_load_message():
    at=AppTest.from_file('app/app.py',default_timeout=30).run()
    widget(at,'selectbox','Data source').set_value('Paste sentences').run()
    widget(at,'text_input','Target word').set_value('bank')
    at.text_area[0].set_value('The bank is open.\nA river bank.')
    at.text_area[1].set_value('No target here.')
    widget(at,'button','Load usages').click().run()
    assert not at.exception
    assert any('**bank** · 2 occurrences' in m.value for m in at.sidebar.markdown)
    assert any(m.label=='Corpus order' for m in at.sidebar.multiselect)
    assert any('Loaded 2 occurrences from 1 corpora' in m.value for m in at.success)
    assert any('No selected forms found in: corpus_2' in m.value for m in at.warning)
    at.run()
    assert not at.success and not at.warning


def test_manual_definition_draft_and_source_survive_navigation():
    at,_=seeded_app(2,definitions=0)
    at.session_state['definitions_raw']=[]
    at.session_state['definitions_formatted']=[]
    at.session_state['definition_embeddings']=None
    at.session_state['def_space_all']=None
    widget(at,'radio','Workspace').set_value('Definitions').run()
    widget(at,'text_area','One definition per line').set_value('A financial institution\nThe side of a river').run()
    widget(at,'radio','Workspace').set_value('Explore').run()
    widget(at,'radio','Workspace').set_value('Definitions').run()
    assert not at.exception
    assert widget(at,'text_area','One definition per line').value=='A financial institution\nThe side of a river'
    widget(at,'selectbox','Definition source').set_value('Wiktionary').run()
    widget(at,'radio','Workspace').set_value('Annotation').run()
    widget(at,'radio','Workspace').set_value('Definitions').run()
    assert widget(at,'selectbox','Definition source').value=='Wiktionary'


def test_fetched_definitions_can_be_narrowed_by_part_of_speech(monkeypatch):
    import app.ui_definitions as ui
    monkeypatch.setattr(ui,'fetch_wiktionary_entries',lambda *a,**kw:[('Noun','A financial institution'),('Verb','To tilt an aircraft'),('Noun','The edge of a river')])
    at,_=seeded_app(2)
    widget(at,'radio','Workspace').set_value('Definitions').run()
    widget(at,'selectbox','Definition source').set_value('Wiktionary').run()
    widget(at,'button','Fetch definitions').click().run()
    assert not at.exception
    assert widget(at,'text_area','Edit definitions (one per line)').value=='A financial institution\nTo tilt an aircraft\nThe edge of a river'
    assert any('2 noun, 1 verb' in c.value for c in at.caption)
    widget(at,'multiselect','Parts of speech to include').set_value(['Noun']).run()
    assert not at.exception
    assert widget(at,'text_area','Edit definitions (one per line)').value=='A financial institution\nThe edge of a river'
    # Edits made after narrowing are kept on later reruns.
    widget(at,'text_area','Edit definitions (one per line)').set_value('A financial institution').run()
    at.run()
    assert widget(at,'text_area','Edit definitions (one per line)').value=='A financial institution'


def test_sense_frequencies_by_corpus_follow_annotations_and_hide_while_blind():
    at,data=seeded_app(3)
    # corpus_0: two finance, one river; corpus_1: one river; corpus_2: nothing annotated.
    at.session_state['_annotation_store']={'annotations':{0:'finance',1:'finance',2:'river',5:'river'},
                                          'sense_classes':['finance','river'],'active_sense_class':'finance','history':[]}
    widget(at,'radio','Workspace').set_value('Annotation').run()
    assert not at.exception
    table=at.dataframe[0].value.set_index('Sense')
    assert list(table.index)==['finance','river','Annotated usages','Loaded usages']
    assert list(table['corpus_0'])==['66.7%','33.3%','3','4']
    assert list(table['corpus_1'])==['0.0%','100.0%','1','4']
    assert list(table['corpus_2'])==['—','—','0','4']
    assert any('corpus_2 (0 of 4)' in c.value for c in at.caption)
    bars=[t for t in json.loads(at.get('plotly_chart')[-1].proto.spec)['data'] if t['type']=='bar']
    assert [b['name'] for b in bars]==['finance','river']
    assert bars[0]['y'][:2]==pytest.approx([2/3,0]) and bars[0]['x']==['corpus_0','corpus_1','corpus_2']
    widget(at,'checkbox','Show percentages of each corpus’s annotated usages').uncheck().run()
    counts=at.dataframe[0].value.set_index('Sense')
    assert list(counts['corpus_0'])==[2,1,3,4] and list(counts['Total'])==[2,2,4,12]
    widget(at,'toggle','Blind annotation').set_value(True).run()
    assert not at.exception
    assert not any('Sense' in frame.value.columns for frame in at.dataframe)
    assert any('Hidden while blind annotation is on' in c.value for c in at.caption)


def test_annotation_export_keeps_extra_columns_from_the_usage_table():
    import io
    from semlens.data_loading import load_from_csv
    from semlens.projects import project_bytes,load_project
    table=('sentence,corpus,start,end,date,title,sense,url\n'
           '"The bank is open.",old,4,8,1999-05-10,Opening hours,money,https://example.org/a\n'
           '"A river bank.",new,8,12,2021-02-01,,,\n')
    data=load_from_csv(io.StringIO(table),'bank')
    assert data.usages[0].metadata=={'date':'1999-05-10','title':'Opening hours','sense':'money','url':'https://example.org/a'}
    assert data.usages[1].metadata=={'date':'2021-02-01'}
    # Extra columns do not change a usage's identity, and survive a saved analysis.
    plain=load_from_csv(io.StringIO('sentence,corpus,start,end\n"The bank is open.",old,4,8\n'),'bank')
    assert plain.usages[0].usage_id==data.usages[0].usage_id
    embedded=embedded_usages_from_precomputed(data,torch.eye(2))
    assert load_project(project_bytes(data,embedded))['data'].usages[0].metadata==data.usages[0].metadata
    at=AppTest.from_file('app/app.py',default_timeout=30).run()
    for k,v in {'target_word_data':data,'target_word':'bank','embedded_usages':embedded,'corpus_order':data.corpus_labels,
                'corpus_labels':data.corpus_labels,'data_revision':1}.items():
        at.session_state[k]=v
    at.session_state['_annotation_store']={'annotations':{0:'finance'},'sense_classes':['finance'],'active_sense_class':'finance','history':[]}
    widget(at,'radio','Workspace').set_value('Annotation').run()
    assert not at.exception
    from app.ui_annotations import annotation_table
    exported=annotation_table(data,{0:'finance'})
    assert exported.loc[0,['sense','date','title','source_sense','url']].tolist()==['finance','1999-05-10','Opening hours','money','https://example.org/a']
    assert exported.loc[1,['sense','date','title']].tolist()==['','2021-02-01','']
    assert any('date, title, source_sense, url' in c.value for c in at.caption)


def test_search_rings_matching_usages_in_the_annotation_plot():
    from semlens.utils import find_usages
    sentences=['An AI bubble.','She said it again.','The dot-com bubble burst.','Speculative bubbles; pure speculation.','Dot com era']
    assert find_usages(sentences,'AI')==[0]
    assert find_usages(sentences,'ai, dot-com')==[0,2]
    assert find_usages(sentences,'specul*')==[3]
    assert find_usages(sentences,'bubble')==[0,2] and find_usages(sentences,'bubble*')==[0,2,3]
    assert find_usages(sentences,'dot com')==[4]
    assert find_usages(sentences,' , * ')==[]
    at,data=seeded_app(2)
    at.session_state['hidden_corpora']=['corpus_1']
    widget(at,'radio','Workspace').set_value('Annotation').run()
    widget(at,'text_input','Highlight usages containing').set_value('example 2, example 6').run()
    assert not at.exception
    assert any('2 of 8 usages contain' in c.value and 'corpus_0: 1, corpus_1: 1' in c.value for c in at.caption)
    ring=next(t for t in scatter_figure(at)['data'] if t['name'].startswith('Contains'))
    # Only the match in a corpus that is switched on is ringed, and it stays clickable.
    assert ring['customdata']==[2]
    widget(at,'toggle','Blind annotation').set_value(True).run()
    assert any('2 of 8 usages contain' in c.value and 'corpus_0' not in c.value for c in at.caption)


def test_target_is_highlighted_in_evidence_tables():
    import pandas as pd
    from app.presentation import sentence_pair_html,nearest_usage_html
    usage='A <b>bank</b> by the river bank.'
    frame=pd.DataFrame([[.2,usage,'The bank closed.',(usage.rfind('bank'),usage.rfind('bank')+4),(4,8)]],
                       columns=['Distance','Usage','Closest counterpart','Usage span','Counterpart span'])
    markup=sentence_pair_html(frame,compact=True)
    assert 'A &lt;b&gt;bank&lt;/b&gt; by the river <mark>bank</mark>.' in markup
    assert 'The <mark>bank</mark> closed.' in markup and markup.count('<mark>')==2
    nearest=nearest_usage_html(pd.DataFrame([['old',.1,'The bank closed.',(4,8)]],columns=['Corpus','Distance','Closest usage','Span']))
    assert 'The <mark>bank</mark> closed.' in nearest
    at,_=seeded_app(2)
    widget(at,'radio','Workspace').set_value('Explore').run()
    at.session_state['full_pairs']=True
    at.run()
    assert all(table.count('<mark>bank</mark>')==2*table.count('<tr><td>') for table in pair_tables(at))


def test_change_over_time_and_scores_follow_chronology_and_legend():
    at,_=seeded_app(5)
    widget(at,'radio','Workspace').set_value('Explore').run()
    assert len(at.get('plotly_chart'))==5
    at.session_state['chronological']=True
    at.run()
    assert not at.exception
    charts=[json.loads(c.proto.spec) for c in at.get('plotly_chart')]
    assert len(charts)==6
    trend=charts[5]['data']
    assert len(trend)==8 and trend[0]['x']==['corpus_1','corpus_2','corpus_3','corpus_4']
    from app.common import cached_comparison
    vectors=at.session_state['embedded_usages'].embeddings.numpy()
    def apd(a,b):
        return cached_comparison(vectors[a*4:a*4+4],vectors[b*4:b*4+4],(f'corpus_{a}',f'corpus_{b}'),'cosine','balanced',20,42).summary.loc['apd','mean']
    assert trend[0]['y']==pytest.approx([apd(0,1),apd(1,2),apd(2,3),apd(3,4)])
    assert trend[1]['y']==pytest.approx([apd(0,1),apd(0,2),apd(0,3),apd(0,4)])
    # Corpora switched off in the legend drop out of the heatmaps and the trend.
    at.session_state['hidden_corpora']=['corpus_1','corpus_3']
    at.run()
    charts=[json.loads(c.proto.spec) for c in at.get('plotly_chart')]
    assert charts[1]['layout']['xaxis']['ticktext']==['corpus_0','corpus_2','corpus_4']
    assert charts[1]['layout']['title']['text']=='Average pairwise distance (APD)'
    assert charts[5]['data'][0]['y']==pytest.approx([apd(0,2),apd(2,4)])
    assert any('Showing 3 of 5 corpora' in c.value for c in at.caption)
    at.session_state['hidden_corpora']=['corpus_0','corpus_1','corpus_2','corpus_3']
    at.run()
    assert not at.exception and len(at.get('plotly_chart'))==1
    assert any('at least two corpora' in m.value for m in at.info)


def test_pca_axes_report_their_share_of_the_variation_and_help_explains_views():
    at,_=seeded_app(2)
    widget(at,'radio','Workspace').set_value('Explore').run()
    layout=scatter_figure(at)['layout']
    from sklearn.decomposition import PCA
    ratios=PCA(2).fit(at.session_state['embedded_usages'].embeddings.numpy()).explained_variance_ratio_
    assert layout['xaxis']['title']['text']==f'PC1 · {ratios[0]:.0%} of the variation'
    assert layout['yaxis']['title']['text']==f'PC2 · {ratios[1]:.0%} of the variation'
    help_text=widget(at,'selectbox','Plot view').help
    assert all(term in help_text for term in ['principal component analysis','UMAP','t-SNE','linear discriminant analysis'])
    amd=next(m for m in at.metric if m.label=='AMD')
    assert 'Average minimum distance (AMD)' in amd.help and r'\min_{b\in B}d(a,b)' in amd.help


def test_first_screen_guides_a_new_user_and_offers_an_example():
    at=AppTest.from_file('app/app.py',default_timeout=60).run()
    guide=next(e for e in at.expander if e.label=='New here? How SemLens works')
    assert guide.proto.expanded
    assert any(e.label=='Glossary' for e in at.sidebar.expander)
    source=widget(at,'selectbox','Data source')
    assert source.value=='Upload raw corpora (.txt)' and any(option.startswith('Example') for option in source.options)
    source.set_value(next(option for option in source.options if option.startswith('Example'))).run()
    widget(at,'button','Load the example').click().run()
    assert not at.exception
    data=at.session_state['target_word_data']
    assert (data.word,len(data),len(data.corpus_labels))==('bubble',350,7)
    assert at.session_state['chronological'] is True
    assert data.usages[0].metadata['date'].startswith('1999')
    # The guide steps aside once data is loaded, and the chosen source is remembered.
    assert not next(e for e in at.expander if e.label=='New here? How SemLens works').proto.expanded
    widget(at,'radio','Workspace').set_value('Annotation').run()
    widget(at,'radio','Workspace').set_value('Data & Model').run()
    assert widget(at,'selectbox','Data source').value.startswith('Example')
    assert widget(at,'selectbox','Language model').value.startswith('XL-LEXEME')


def test_review_shows_frequencies_forms_and_a_sortable_concordance():
    at=AppTest.from_file('app/app.py',default_timeout=30).run()
    widget(at,'selectbox','Data source').set_value('Paste sentences').run()
    widget(at,'text_input','Target word').set_value('bank')
    widget(at,'text_input','Other forms (optional)').set_value('bank*')
    at.text_area[0].set_value('The bank was shut.\nTwo banks merged.\nNo money here at all.')
    at.text_area[1].set_value('A river bank.\nBanking on rain.')
    widget(at,'button','Load usages').click().run()
    assert not at.exception
    assert [u.matched_form for u in at.session_state['target_word_data'].usages]==['bank','banks','bank','Banking']
    at.session_state['review_occurrences']=True
    at.run()
    assert not at.exception
    counts=at.dataframe[0].value.set_index('Corpus')
    assert list(counts['Found in the corpus'])==[2,2] and list(counts['Corpus size (words)'])==[12,6]
    assert counts.loc['corpus_2','Per million words']==pytest.approx(2/6*1e6,abs=.1)
    forms=at.dataframe[1].value.set_index('Form')
    assert forms.loc['bank','Total']==2 and forms.loc['banking','corpus_2']==1
    def lines():
        table=next(el.proto.body for el in at.get('html') if 'semlens-kwic' in el.proto.body)
        return re.findall(r'<mark>(.*?)</mark></td><td class="semlens-kwic-right">(.*?)</td>',table)
    assert [target for target,_ in lines()]==['bank','banks','bank','Banking']
    widget(at,'selectbox','Sort lines by').set_value('Word after the target')
    at.session_state['review_occurrences']=True
    at.run()
    assert [right.strip() for _,right in lines()]==['.','merged.','on rain.','was shut.']
    widget(at,'text_input','Lines containing').set_value('river, merg*')
    at.session_state['review_occurrences']=True
    at.run()
    assert sorted(target for target,_ in lines())==['bank','banks']
    # The pasted texts are offered again after visiting another workspace.
    widget(at,'radio','Workspace').set_value('Explore').run()
    widget(at,'radio','Workspace').set_value('Data & Model').run()
    assert widget(at,'selectbox','Data source').value=='Paste sentences'
    assert at.text_area[1].value=='A river bank.\nBanking on rain.'
    widget(at,'text_input','Target word').set_value('river')
    widget(at,'checkbox','Match upper and lower case exactly').check()
    widget(at,'button','Load usages').click().run()
    assert not at.exception and at.session_state['target_word_data'].word=='river'


def test_plot_options_set_components_and_projection_settings(monkeypatch):
    at,_=seeded_app(3)
    widget(at,'radio','Workspace').set_value('Explore').run()
    widget(at,'number_input','Horizontal component').set_value(2)
    widget(at,'number_input','Vertical component').set_value(3).run()
    assert not at.exception
    fig=scatter_figure(at)
    assert fig['layout']['xaxis']['title']['text'].startswith('PC2 · ') and fig['layout']['yaxis']['title']['text'].startswith('PC3 · ')
    from sklearn.decomposition import PCA
    expected=PCA(3).fit_transform(at.session_state['embedded_usages'].embeddings.numpy())[:,[1,2]]
    traces=[t for t in fig['data'] if t.get('name','').startswith('corpus_')]
    actual=np.concatenate([np.column_stack([t['x'],t['y']]) for t in traces])
    np.testing.assert_allclose(np.abs(actual),np.abs(expected),atol=1e-4)
    import app.common as common
    calls=[]
    def fake(array,method='pca',**settings):
        calls.append((method,settings))
        return np.zeros((len(array),2))
    monkeypatch.setattr(common,'reduce_to_2d',fake)
    common.cached_projection.clear()
    widget(at,'selectbox','Plot view').set_value('UMAP').run()
    assert calls[-1]==('umap',{'n_neighbors':11,'min_dist':.1,'random_state':42,'metric':'cosine'})
    widget(at,'selectbox','Vector distance').set_value('euclidean')
    widget(at,'number_input','Neighbours').set_value(5)
    widget(at,'number_input','Seed').set_value(7).run()
    assert calls[-1]==('umap',{'n_neighbors':5,'min_dist':.1,'random_state':7,'metric':'euclidean'})
    widget(at,'selectbox','Plot view').set_value('t-SNE').run()
    assert calls[-1][0]=='tsne' and calls[-1][1]['perplexity']==3 and calls[-1][1]['metric']=='euclidean'
    assert not at.exception
    widget(at,'selectbox','Plot view').set_value('PCA')
    widget(at,'text_input','Highlight usages containing').set_value('example 3').run()
    ring=next(t for t in scatter_figure(at)['data'] if t['name'].startswith('Contains'))
    assert len(ring['x'])==1
    assert any(b.label=='Download plot data (CSV)' for b in at.get('download_button'))
    assert any(b.label=='Download scores for all corpus pairs (CSV)' for b in at.get('download_button'))


def test_display_settings_change_points_palette_and_order():
    from semlens.utils import palette_for,SEQUENTIAL_PALETTE
    at,_=seeded_app(3)
    at.session_state['corpus_order']=['corpus_2','corpus_0','corpus_1']
    widget(at,'radio','Workspace').set_value('Explore').run()
    widget(at,'slider','Point size').set_value(14)
    widget(at,'slider','Point opacity').set_value(.5)
    widget(at,'selectbox','Colour palette').set_value(SEQUENTIAL_PALETTE).run()
    assert not at.exception
    traces=[t for t in scatter_figure(at)['data'] if (t.get('meta') or {}).get('semlens_corpus')]
    assert all(t['marker']['size']==14 and t['marker']['opacity']==.5 for t in traces)
    assert [t['marker']['color'] for t in traces]==palette_for(SEQUENTIAL_PALETTE,3)
    widget(at,'radio','Save figures as').set_value('svg').run()
    assert json.loads(at.get('plotly_chart')[0].proto.config)['toImageButtonOptions']['format']=='svg'
    widget(at,'button','Order by name').click().run()
    assert at.session_state['corpus_order']==['corpus_0','corpus_1','corpus_2']
    from app.common import natural_sort_key
    assert sorted(['corpus_10','corpus_2','1999-2003','2004'],key=natural_sort_key)==['1999-2003','2004','corpus_2','corpus_10']


def test_annotation_tools_rename_import_and_show_remaining(monkeypatch):
    import pandas as pd
    import app.ui_annotations as ui
    import app.annotation_store as store
    at,data=seeded_app(2)
    at.session_state['_annotation_store']={'annotations':{0:'finance',1:'money',5:'river'},'sense_classes':['finance','money','river'],
                                          'active_sense_class':'money','history':[]}
    widget(at,'radio','Workspace').set_value('Annotation').run()
    assert not at.exception
    # Renaming onto an existing class merges the two, and can be undone.
    widget(at,'selectbox','Class to rename').set_value('money')
    widget(at,'text_input','New name').set_value('finance')
    widget(at,'button','Rename').click().run()
    assert not at.exception
    saved=at.session_state['_annotation_store']
    assert saved['sense_classes']==['finance','river'] and saved['annotations']=={0:'finance',1:'finance',5:'river'}
    assert saved['active_sense_class']=='finance'
    # Only what is left to annotate is drawn, and nothing hidden can be selected.
    widget(at,'toggle','Show only usages without a sense').set_value(True).run()
    fig=scatter_figure(at)
    assert sum(len(t['x']) for t in fig['data'] if (t.get('meta') or {}).get('semlens_corpus'))==len(data)-3
    assert not any(t['name'].startswith('⬡') for t in fig['data'])
    assert sorted(next(t for t in fig['data'] if t['name']=='_hit_area')['customdata'])==[2,3,4,6,7]
    # Importing matches by usage_id or by sentence position, and adds new classes.
    fake=SimpleNamespace(session_state={'_annotation_store':{'annotations':{0:'finance'},'sense_classes':['finance'],'active_sense_class':'finance','history':[]}})
    monkeypatch.setattr(store,'st',fake)
    table=pd.DataFrame([{'usage_id':data.usages[2].usage_id,'sentence':'','corpus':'','start':'','end':'','sense':'river'},
                        {'usage_id':'','sentence':data.usages[6].sentence,'corpus':'corpus_1','start':'4','end':'8','sense':'finance'},
                        {'usage_id':'unknown','sentence':'Not loaded.','corpus':'corpus_1','start':'0','end':'3','sense':'finance'},
                        {'usage_id':data.usages[3].usage_id,'sentence':'','corpus':'','start':'','end':'','sense':''}])
    assert ui.import_annotations(data,table)==(2,1,['river'])
    assert store.get_annotations()=={0:'finance',2:'river',6:'finance'} and store.get_sense_classes()==['finance','river']
    store.undo_annotations()
    assert store.get_annotations()=={0:'finance'}
    with pytest.raises(ValueError,match='column named sense'):
        ui.import_annotations(data,pd.DataFrame([{'usage_id':'x'}]))
    with pytest.raises(ValueError,match='No row with a sense matches'):
        ui.import_annotations(data,pd.DataFrame([{'usage_id':'x','sense':'finance'}]))


def test_annotation_list_shows_concordance_lines_with_their_senses():
    at,data=seeded_app(2)
    at.session_state['_annotation_store']={'annotations':{1:'finance'},'sense_classes':['finance','river'],'active_sense_class':'finance','history':[]}
    widget(at,'radio','Workspace').set_value('Annotation').run()
    at.session_state['annotation_list']=True
    at.run()
    assert not at.exception
    table=at.get('arrow_data_frame')[0].value if at.get('arrow_data_frame') else None
    editor=next(frame.value for frame in at.dataframe if 'Target' in frame.value.columns) if table is None else table
    assert list(editor['Sense'][:3])==['—','finance','—'] and set(editor['Target'])=={'bank'}
    assert list(editor.columns)==['Sense','Before','Target','After','Corpus']
    widget(at,'checkbox','Only usages without a sense').check()
    at.session_state['annotation_list']=True
    at.run()
    editor=next(frame.value for frame in at.dataframe if 'Target' in frame.value.columns)
    assert len(editor)==len(data)-1
    widget(at,'toggle','Blind annotation').set_value(True)
    at.session_state['annotation_list']=True
    at.run()
    assert not at.exception
    assert 'Corpus' not in next(frame.value for frame in at.dataframe if 'Target' in frame.value.columns).columns
