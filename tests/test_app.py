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
    return [el.proto.body for el in at.get('html') if '<div class="semlens-pair-table">' in el.proto.body]


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
    assert values.amd.max()>0


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
    assert all(t['x']==traces[0]['x'] and len(t['x'])==30 for t in traces)
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
    anchors=next(t for t in fig['data'] if t.get('name')=='Definitions')
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
    monkeypatch.setattr(ui,'fetch_wiktionary_definitions',lambda *a,**kw:['A financial institution','The edge of a river'])
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
    monkeypatch.setattr(ui,'fetch_wiktionary_definitions',fail)
    widget(at,'button','Fetch definitions').click().run()
    assert widget(at,'text_area','Edit definitions (one per line)').value==draft
    assert at.error[0].value=='Dictionary unavailable'
    monkeypatch.setattr(ui,'fetch_wiktionary_definitions',lambda *a,**kw:['Refetched definition'])
    widget(at,'button','Fetch definitions').click().run()
    assert not at.exception
    assert widget(at,'text_area','Edit definitions (one per line)').value=='Refetched definition'


def test_definition_drafts_are_separate_for_dictionary_languages(monkeypatch):
    import app.ui_definitions as ui
    monkeypatch.setattr(ui,'fetch_wiktionary_definitions',lambda *a,**kw:[])
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
    assert len(at.dataframe[0].value)==1
    widget(at,'checkbox','Include unannotated occurrences').check().run()
    assert len(at.dataframe[0].value)==len(data)


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
