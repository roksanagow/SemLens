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
    # APD and PRT are reported too, in the same absolute distance.
    assert row.apd == pytest.approx(.70)
    assert row.prt == pytest.approx(.70)


def test_closest_usages_are_nearest_first_in_the_chosen_distance():
    from semlens.metrics import closest_usages
    query = torch.tensor([1., 0.])
    candidates = torch.tensor([[0., 1.], [2., 0.], [1., 1.], [-1., 0.]])
    positions, distances = closest_usages(query, candidates, 3)
    assert positions.tolist() == [1, 2, 0]
    assert distances.tolist() == pytest.approx([0, 1 - 2 ** -.5, 1], abs=1e-6)
    positions, distances = closest_usages(query, candidates, 2, 'euclidean')
    assert positions.tolist() == [1, 2] and distances.tolist() == pytest.approx([1, 1])
    assert len(closest_usages(query, candidates, 10)[0]) == 4


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


def test_lda_does_not_separate_shuffled_labels_in_high_dimensions():
    rng = np.random.default_rng(0)
    # More dimensions than usages: fitted directly, LDA separates any labelling.
    x = torch.from_numpy(rng.normal(size=(120, 300)).astype('float32'))
    labels = [f'corpus_{i % 3}' for i in range(120)]
    result = lda_projection(x, labels)
    assert result.reducer is not None and result.reducer.n_components_ == 5 and result.n_features == 300
    y = np.array([int(label[-1]) for label in labels])
    centroids = np.stack([result.coords_2d[y == i].mean(0) for i in range(3)])
    accuracy = (np.linalg.norm(result.coords_2d[:, None] - centroids[None], axis=2).argmin(1) == y).mean()
    assert accuracy < .6
    # New points go through the same reduction, and LD1 weights are reported on the original features.
    np.testing.assert_allclose(lda_transform_new_points(result, x), result.coords_2d, atol=1e-4)
    from semlens.lda import lda_definition_weights
    weights = lda_definition_weights(result, [f'f{i}' for i in range(300)]).set_index('definition').weight
    ld1 = (x.numpy() - x.numpy().mean(0)) @ weights[[f'f{i}' for i in range(300)]].to_numpy()
    assert abs(np.corrcoef(ld1, result.coords_2d[:, 0])[0, 1]) > .999
    # A few definition dimensions with plenty of usages are used as they are.
    assert lda_projection(x[:, :4], labels).reducer is None


def test_forms_can_use_wildcards_and_exact_case():
    from semlens.data_loading import find_occurrences
    sentence = 'Bubbles bubbled in May; it may bubble, unlike a bubblegum-free hubbub.'
    spans = lambda *a, **k: [sentence[s:e] for s, e in find_occurrences(sentence, *a, **k)]
    assert spans('bubble') == ['bubble']
    assert spans('bubble', ['bubbl*']) == ['Bubbles', 'bubbled', 'bubble', 'bubblegum']
    assert spans('may') == ['May', 'may'] and spans('may', match_case=True) == ['may']
    assert spans('May', match_case=True) == ['May']
    with pytest.raises(ValueError, match='target word in full'):
        find_occurrences(sentence, 'bubbl*')
    data = load_from_raw_text({'old': 'In May we may go.'}, 'may')
    assert len(data) == 2
    assert len(select_usages(corpus_sentences({'old': 'In May we may go.'}), 'may', match_case=True)) == 1


def test_text_is_divided_at_punctuation_blank_lines_lines_or_paragraphs():
    text = 'A HEADING\n\nDr. Smith met Mr. Jones at No. 10. They talked,\nat length! See Fig. 2 e.g. the chart.\n\nNext paragraph'
    auto = [r.sentence for r in corpus_sentences({'c': text})]
    assert auto == ['A HEADING', 'Dr. Smith met Mr. Jones at No. 10.', 'They talked, at length!', 'See Fig. 2 e.g. the chart.', 'Next paragraph']
    assert [r.sentence for r in corpus_sentences({'c': text}, split='paragraphs')] == [
        'A HEADING', 'Dr. Smith met Mr. Jones at No. 10. They talked, at length! See Fig. 2 e.g. the chart.', 'Next paragraph']
    by_line = [r.sentence for r in corpus_sentences({'c': text}, split='lines')]
    assert by_line == [r.sentence for r in corpus_sentences({'c': text}, lines=True)] and len(by_line) == 4
    with pytest.raises(ValueError, match='auto, lines or paragraphs'):
        corpus_sentences({'c': text}, split='words')


def test_uploaded_text_is_decoded_and_counted():
    from semlens.data_loading import decode_text, count_words
    assert decode_text('naïve café'.encode('utf-8')) == ('naïve café', 'UTF-8')
    assert decode_text(b'\xef\xbb\xbfbank') == ('bank', 'UTF-8')
    assert decode_text('naïve café'.encode('cp1252')) == ('naïve café', 'Windows-1252')
    assert decode_text('naïve'.encode('utf-16')) == ('naïve', 'UTF-16')
    assert count_words("The dot-com bubble didn't burst in 2001.") == 7


def test_concordance_lines_and_palettes():
    from semlens.utils import keyword_in_context, palette_for, SEQUENTIAL_PALETTE
    sentence = 'It was said that the great housing bubble would burst\nsooner or later, whatever anyone did about it.'
    start = sentence.index('bubble')
    left, target, right = keyword_in_context(sentence, (start, start + 6), 20)
    assert (left, target, right) == ('… the great housing ', 'bubble', ' would burst sooner…')
    assert keyword_in_context('bubble.', (0, 6)) == ('', 'bubble', '.')
    seven = palette_for(SEQUENTIAL_PALETTE, 7)
    assert len(set(seven)) == 7 and seven[0] == '#472d7b' and seven[-1] == '#fde725'
    assert palette_for(SEQUENTIAL_PALETTE, 1) == ['#28ae80']
    assert len(set(palette_for('IBM Design', 10))) == 10
