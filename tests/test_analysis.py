import io
from pathlib import Path
import numpy as np
import pytest
import torch
from semlens.data_loading import load_from_csv, load_from_sentences, load_from_raw_text, corpus_sentences, select_usages
from semlens.metrics import compute_per_definition_metrics, sample_indices, compute_sampled_metrics, nearest_counterparts
from semlens.lda import lda_projection, lda_transform_new_points


def test_scalar_scores_do_not_collapse_under_cosine_setting():
    a, b = torch.tensor([[.1], [.2]]), torch.tensor([[.8], [.9]])
    row = compute_per_definition_metrics(a, b, ['sense'], distance='cosine').iloc[0]
    assert row.amd == pytest.approx(.65)
    assert row.samd == pytest.approx(.70)
    assert row.mean_distance_difference == pytest.approx(.70)


def test_offsets_are_authoritative_and_keep_second_occurrence():
    data = load_from_csv(io.StringIO('sentence,corpus,start,end\nbank then bank,old,10,14\n'), 'bank')
    assert data.char_spans == [(10, 14)]
    assert data.usages[0].matched_form == 'bank'
    with pytest.raises(ValueError, match='Invalid target span'):
        load_from_csv(io.StringIO('sentence,corpus,start,end\nbank,old,10,14\n'), 'bank')
    with pytest.raises(ValueError, match='both start and end'):
        load_from_csv(io.StringIO('sentence,corpus,start,end\nbank,old,0,\n'), 'bank')


def test_forms_repeated_occurrences_and_no_unrelated_fallback():
    data = load_from_raw_text({'earlier': 'Run, runner! Running and runs.', 'later': 'A runner runs.'}, 'run', forms=['runner','running','runs'])
    assert len(data) == 6
    assert data.corpus_labels == ['earlier', 'later']
    assert [u.matched_form for u in data.usages] == ['Run','runner','Running','runs','runner','runs']
    with pytest.raises(ValueError, match='no target form'):
        load_from_sentences(['The cat sat on the mat.'], 'bank', ['old'])


def test_precomputed_metadata_cannot_expand_ambiguously():
    with pytest.raises(ValueError, match='multiple occurrences'):
        load_from_csv(io.StringIO('sentence,corpus\nbank then bank,old\n'), 'bank', all_occurrences=False)


def test_one_to_ten_corpora_and_order():
    data = load_from_sentences(['bank']*10, 'bank', [f'c{i}' for i in reversed(range(10))])
    assert data.corpus_labels == [f'c{i}' for i in reversed(range(10))]
    with pytest.raises(ValueError, match='10 corpora'):
        load_from_sentences(['bank']*11, 'bank', [f'c{i}' for i in range(11)])


def test_balanced_sampling_is_reproducible_without_duplicate_draws():
    draws = sample_indices(4, 12, mode='repeated', repetitions=8, seed=5)
    again = sample_indices(4, 12, mode='repeated', repetitions=8, seed=5)
    for (a,b), (aa,bb) in zip(draws,again):
        assert len(set(a)) == len(set(b)) == 4
        np.testing.assert_array_equal(a,aa)
        np.testing.assert_array_equal(b,bb)
    assert len({tuple(b) for a,b in draws}) > 1
    a,b = sample_indices(4,12,mode='unequal')[0]
    assert (len(a), len(b)) == (4,12)


def test_counterparts_reconstruct_directional_score():
    a, b = torch.tensor([[0.],[2.],[9.]]), torch.tensor([[1.],[3.]])
    result = compute_sampled_metrics(a,b,distance='euclidean',mode='unequal')
    ab,ba = nearest_counterparts(a,b,'euclidean')
    assert ab.iloc[0].source_index == 2
    assert ab.iloc[0].counterpart_index == 1
    assert ab.distance.mean() == pytest.approx(result.scores.iloc[0].amd_corpus_1_to_corpus_2)
    assert ba.distance.mean() == pytest.approx(result.scores.iloc[0].amd_corpus_2_to_corpus_1)


def test_multicorpus_lda_projects_all_usages_and_new_points_consistently():
    torch.manual_seed(11)
    x = torch.randn(24,5)
    labels = ['old']*8 + ['middle']*8 + ['new']*8
    result = lda_projection(x, labels)
    assert result.coords_2d.shape == (24,2)
    assert result.axis_labels == ('LD1','LD2')
    np.testing.assert_allclose(lda_transform_new_points(result,x),result.coords_2d,atol=1e-5)
    with pytest.raises(ValueError, match='two corpora'):
        lda_projection(x, ['only']*24)


def test_user_testing_texts_match_forms_and_source_sentences():
    root=Path(__file__).resolve().parents[1]/'testing_texts'
    paths=list(root.glob('*.txt'))
    if not paths:
        pytest.skip('Local user-provided testing_texts are unavailable')
    records=corpus_sentences({p.stem:p.read_text(encoding='utf-8-sig') for p in paths})
    data=select_usages(records,'love',['loves','loved','loving'])
    assert len(data.corpus_labels)==len(paths)
    assert len(data)>100
    assert all(u.matched_form.casefold() in {'love','loves','loved','loving'} for u in data.usages)
    assert len({u.usage_id for u in data.usages})==len(data)
