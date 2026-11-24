"""Discover: counting and sampling, contextual vectors on disk, kNN scoring, and the static first pass."""
import json
from collections import Counter

import numpy as np
import pytest

from semlens.discover.knn import (Neighbours, StoreCandidates, add_seeds, calibrate_threshold, nearest_seeds, rows_for_words,
                                  score_words)
from semlens.discover.static import Space, align, ppmi_space, recommended_method, static_scores, tokenised
from semlens.discover.vocabulary import Context, Sample, contexts_from_texts, sample_occurrences
from semlens.embeddings import load_model


def test_restricting_texts_keeps_only_contexts_with_the_given_words():
    texts = {'old': 'A soap bubble rose. The market fell. An echo chamber formed.', 'new': 'Nothing here. A bubble again.'}
    kept = contexts_from_texts(texts, restrict='bubble*, echo chamber')
    assert [c.text for c in kept] == ['A soap bubble rose.', 'An echo chamber formed.', 'A bubble again.']
    assert len(contexts_from_texts(texts)) == 5


def test_counting_filters_stop_words_and_rare_words_but_keeps_seeds():
    contexts = [Context('a', 'The bubble and the bubble burst.'), Context('a', 'A bubble, a silo.'), Context('b', 'The silo.')]
    sample = sample_occurrences(contexts, cap=10, min_count=2, stop_words={'the', 'and', 'a'}, keep_words={'silo'})
    assert sample.counts == {'a': {'bubble': 3, 'silo': 1}, 'b': {'silo': 1}}
    assert 'burst' not in sample.words and len(sample) == 5
    row = int(np.flatnonzero(sample.corpus == 1)[0])
    assert sample.word_of(row) == 'silo' and sample.text_of(row)[slice(*sample.span_of(row))] == 'silo'


def test_the_cap_samples_evenly_through_the_corpus():
    # 2,000 occurrences of one word, one per context; a cap of 100 should come from every part of the corpus.
    contexts = [Context('c', f'word{"" if i % 2 else ""} filler{i % 7} bubble.') for i in range(2000)]
    sample = sample_occurrences(contexts, cap=100, min_count=1, seed=1)
    rows = np.flatnonzero(np.array(sample.words)[sample.word] == 'bubble')
    assert len(rows) == 100 and sample.counts['c']['bubble'] == 2000
    positions = sample.context[rows]
    # About a tenth in each tenth of the corpus (uniform sampling, not the first 100).
    per_tenth = np.bincount(positions // 200, minlength=10)
    assert per_tenth.min() >= 3 and per_tenth.max() <= 20 and positions.max() > 1800
    # The same seed gives the same sample.
    assert np.array_equal(sample_occurrences(contexts, cap=100, min_count=1, seed=1).context, sample.context)
    summary = sample.summary()
    assert int(summary.loc[0, 'Occurrences left out by the cap']) == sum(sample.counts['c'].values()) - len(sample)


class ArrayCandidates:
    def __init__(self, vectors):
        self.vectors, self.rows = vectors, len(vectors)

    def __call__(self, rows):
        return self.vectors[rows]


def unit(rng, n, d):
    x = rng.normal(size=(n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_nearest_seeds_match_a_direct_computation_and_skip_themselves():
    rng = np.random.default_rng(0)
    candidates, seeds = unit(rng, 300, 16), unit(rng, 25, 16)
    found = nearest_seeds(ArrayCandidates(candidates), seeds, k=5, chunk=64, device='cpu')
    direct = np.sort(1 - candidates @ seeds.T, axis=1)[:, :5]
    assert np.allclose(found.distances, direct, atol=1e-5)
    # A candidate that is itself seed 3 does not count itself.
    own = np.full(300, -1); own[7] = 3
    candidates[7] = seeds[3]
    skipped = nearest_seeds(ArrayCandidates(candidates), seeds, k=5, own_seed=own, device='cpu')
    assert 3 not in skipped.seeds[7] and skipped.distances[7, 0] > 1e-4


def test_adding_seeds_later_gives_the_same_neighbours_as_starting_with_them():
    rng = np.random.default_rng(1)
    candidates, seeds = unit(rng, 200, 12), unit(rng, 30, 12)
    full = nearest_seeds(ArrayCandidates(candidates), seeds, k=6, device='cpu')
    first = nearest_seeds(ArrayCandidates(candidates), seeds[:18], k=6, device='cpu')
    later = add_seeds(first, ArrayCandidates(candidates), seeds[18:], 18, device='cpu')
    assert np.allclose(later.distances, full.distances, atol=1e-5) and np.array_equal(np.sort(later.seeds, 1), np.sort(full.seeds, 1))


def test_threshold_and_scores_reward_words_whose_usages_come_close_to_the_seeds():
    rng = np.random.default_rng(2)
    concept = unit(rng, 1, 32)[0]
    seeds = concept + 0.3 * unit(rng, 40, 32)
    seeds /= np.linalg.norm(seeds, axis=1, keepdims=True)
    threshold = calibrate_threshold(seeds, k=5, percentile=90)
    # Three words: "near" (all usages like the seeds), "half" (half its usages), "far" (none).
    near = concept + 0.3 * unit(rng, 20, 32)
    half = np.vstack([concept + 0.3 * unit(rng, 10, 32), unit(rng, 10, 32)])
    far = unit(rng, 20, 32)
    vectors = np.vstack([near, half, far])
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    sample = Sample([Context('c', 'x')], ['c'], ['far', 'half', 'near'], np.zeros(60, np.int16), np.zeros(60, np.int32),
                    np.zeros(60, np.int32), np.ones(60, np.int32), np.array([2] * 20 + [1] * 20 + [0] * 20, np.int32),
                    {'c': {'near': 20, 'half': 20, 'far': 50}})
    scores = score_words(sample, nearest_seeds(ArrayCandidates(vectors), seeds, k=5, device='cpu'), threshold).set_index('word')
    assert list(scores.sort_values('knn_amd').index) == ['near', 'half', 'far']
    assert scores.loc['near', 'share_near'] > 0.8 and 0.3 < scores.loc['half', 'share_near'] < 0.7 and scores.loc['far', 'share_near'] < 0.1
    assert scores.loc['far', 'occurrences'] == 50 and scores.loc['far', 'sampled'] == 20
    assert 'near' not in score_words(sample, Neighbours(np.zeros((60, 1), np.float32), np.zeros((60, 1), np.int32)), None,
                                     hide_words={'NEAR'})['word'].tolist()


def test_seed_words_may_use_wildcards():
    contexts = [Context('c', 'Bubbles and a bubble, a balloon and silos.')]
    sample = sample_occurrences(contexts, cap=5, min_count=1)
    rows = rows_for_words(sample, ['bubbl*', 'silo'])
    assert sorted(sample.word_of(r) for r in rows) == ['bubble', 'bubbles']
    assert len(rows_for_words(sample, [' '])) == 0


@pytest.fixture(scope='module')
def encoder():
    try:
        return load_model('google/bert_uncased_L-2_H-128_A-2', device='cpu', local_files_only=True)
    except Exception:
        pytest.skip('Local test encoder unavailable')


def test_sampled_occurrences_are_embedded_to_disk_and_centred(encoder, tmp_path):
    from semlens.discover.contextual import VectorStore, embed_seed_usages, encode_sample
    contexts = contexts_from_texts({'old': 'The bank lent money. The river bank flooded. A bank closed.',
                                    'new': 'The bank app failed. Banks lend money. The bank of the river.'})
    sample = sample_occurrences(contexts, cap=10, min_count=1)
    store = encode_sample(encoder, sample, tmp_path / 'run', layer=1, batch_size=2)
    assert store.vectors.dtype == np.float16 and store.vectors.shape == (len(sample), 128) and store.valid.all()
    reopened = VectorStore.load(tmp_path / 'run')
    assert reopened.sample.words == sample.words and np.allclose(reopened.mean, store.mean)
    unit_vectors = reopened.normalised(slice(0, len(sample)))
    assert np.allclose(np.linalg.norm(unit_vectors, axis=1), 1, atol=1e-4)
    # Centring: the mean of the centred vectors is about zero.
    assert np.abs((np.asarray(reopened.vectors, np.float32) - reopened.mean).mean(axis=0)).max() < 1e-2
    # A seed usage given as a sentence lands in the same space as the same usage from the corpus.
    row = int(rows_for_words(sample, ['bank'])[0])
    seed = embed_seed_usages(encoder, [sample.text_of(row)], [sample.span_of(row)], reopened)
    assert float(seed[0] @ unit_vectors[row]) > 0.99
    found = nearest_seeds(StoreCandidates(reopened), seed, k=1, device='cpu')
    assert np.argmin(found.distances[:, 0]) == row


def test_ppmi_spaces_align_and_score_words_near_the_seeds():
    rng = np.random.default_rng(3)
    # Two corpora where "silo" and "enclave" share contexts with "bubble"; "tea" does not.
    def corpus(n):
        sentences = []
        for _ in range(n):
            sentences.append(list(rng.choice(['bubble', 'silo', 'enclave'], 1)) + ['closed', 'group', 'insiders', 'isolated'])
            sentences.append(['tea', 'cup', 'milk', 'sugar', 'hot'])
            sentences.append(['market', 'price', 'rise', 'fall', 'trade'])
        return sentences
    spaces = {'old': ppmi_space(corpus(150), dimensions=8, min_count=2), 'new': ppmi_space(corpus(150), dimensions=8, min_count=2)}
    aligned = align(spaces)
    old, new = aligned['old'], aligned['new']
    # After alignment, a shared word points the same way in both corpora.
    assert float(old.vectors[old.index()['tea']] @ new.vectors[new.index()['tea']]) > 0.9
    scores = static_scores(aligned, ['bubble'], k=2)
    ranked = scores[scores.corpus == 'new'].sort_values('knn_amd').word.tolist()
    assert 'bubble' not in ranked and ranked.index('silo') < ranked.index('tea') and ranked.index('enclave') < ranked.index('milk')
    assert recommended_method({'a': 2_000_000, 'b': 500_000}) == 'ppmi' and recommended_method({'a': 2_000_000}) == 'word2vec'


def test_word2vec_space_is_reproducible():
    pytest.importorskip('gensim')
    from semlens.discover.static import word2vec_space
    sentences = [['the', 'bubble', 'burst'], ['the', 'market', 'fell'], ['a', 'soap', 'bubble']] * 30
    first = word2vec_space(sentences, dimensions=10, min_count=2, epochs=3)
    second = word2vec_space(sentences, dimensions=10, min_count=2, epochs=3)
    assert first.words == second.words and np.allclose(first.vectors, second.vectors)
    assert tokenised([Context('a', "A don't-stop bubble.")], 'a') == [['a', "don't-stop", 'bubble']]


def test_slot_vectors_ignore_the_word_itself(encoder, tmp_path):
    from semlens.discover.contextual import VectorStore, embed_seed_usages, encode_sample, masked_text
    assert masked_text('A soap bubble rose.', (7, 13), '[MASK]') == 'A soap [MASK] rose.'
    contexts = contexts_from_texts({'c': 'She lives in a bubble of friends. She lives in a cocoon of friends. The bank lent money.'})
    sample = sample_occurrences(contexts, cap=5, min_count=1)
    store = encode_sample(encoder, sample, tmp_path / 'slots', layer=1, batch_size=2, representation='slot')
    assert store.metadata['discover_representation'] == 'slot' and store.valid.all()
    vectors = store.normalised(slice(0, len(sample)))
    bubble, cocoon = (int(rows_for_words(sample, [w])[0]) for w in ('bubble', 'cocoon'))
    # The same slot with a different word in it gives the same vector.
    assert float(vectors[bubble] @ vectors[cocoon]) > 0.999
    seed = embed_seed_usages(encoder, [sample.text_of(bubble)], [sample.span_of(bubble)], VectorStore.load(tmp_path / 'slots'))
    assert float(seed[0] @ vectors[bubble]) > 0.999


def test_background_correction_discounts_words_close_to_everything():
    from semlens.discover.knn import background_rows
    rng = np.random.default_rng(4)
    concept, everywhere = unit(rng, 1, 24)[0], unit(rng, 1, 24)[0]
    seeds = np.vstack([concept + 0.4 * unit(rng, 1, 24)[0] for _ in range(20)] + [everywhere + 0.4 * unit(rng, 1, 24)[0] for _ in range(20)])
    seeds /= np.linalg.norm(seeds, axis=1, keepdims=True)
    # "hub" sits near the common direction shared by every usage; "specific" near the concept only.
    background = np.vstack([everywhere + 0.4 * unit(rng, 1, 24)[0] for _ in range(200)] + [unit(rng, 1, 24)[0] for _ in range(200)])
    hub = everywhere + 0.3 * unit(rng, 10, 24)
    specific = concept + 0.3 * unit(rng, 10, 24)
    vectors = np.vstack([hub, specific, background])
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    n = len(vectors)
    sample = Sample([Context('c', 'x')], ['c'], ['background', 'hub', 'specific'], np.zeros(n, np.int16), np.zeros(n, np.int32),
                    np.zeros(n, np.int32), np.ones(n, np.int32), np.array([1] * 10 + [2] * 10 + [0] * 400, np.int32), {'c': {}})
    rows = background_rows(sample, np.ones(n, bool), size=300, exclude_rows=range(20))
    assert len(rows) == 300 and rows.min() >= 20
    plain = score_words(sample, nearest_seeds(ArrayCandidates(vectors), seeds, k=5, device='cpu'), None).set_index('word')
    corrected = score_words(sample, nearest_seeds(ArrayCandidates(vectors), seeds, k=5, device='cpu'), None,
                            background=nearest_seeds(ArrayCandidates(vectors), vectors[rows], k=5, device='cpu')).set_index('word')
    assert corrected.loc['specific', 'knn_amd'] < corrected.loc['hub', 'knn_amd']
    assert corrected.loc['specific', 'share_near'] > corrected.loc['hub', 'share_near']
    assert plain.loc['hub', 'knn_amd'] < plain.loc['background', 'knn_amd']


# --- Counting, seeds, saving and hand-off ---------------------------------------------------------

def test_cap_estimates_match_the_sample_and_the_vocabulary_can_be_limited():
    from semlens.discover.vocabulary import cap_estimates, count_occurrences
    contexts = [Context('a', ' '.join(['bubble'] * 30 + ['silo'] * 4 + ['the'] * 9)), Context('b', 'bubble cocoon cocoon cocoon')]
    counts = count_occurrences(contexts)
    assert counts[('a', 'bubble')] == 30 and counts[('a', 'the')] == 9
    table = cap_estimates(counts, [10, 0], min_count=3, stop_words={'the'}).set_index('cap')
    for cap in (10, 0):
        sample = sample_occurrences(contexts, cap=cap, min_count=3, stop_words={'the'})
        assert table.loc[cap, 'occurrences'] == len(sample)
    assert table.loc[10, 'capped'] == 1 and table.loc[0, 'share'] == 1.0
    only = sample_occurrences(contexts, cap=0, min_count=1, only_words={'cocoon'}, keep_words={'silo'})
    assert set(only.words) == {'cocoon', 'silo'} and only.settings['cap'] == 0


def test_seed_usages_from_tables_and_annotations():
    from semlens.data_loading import load_from_sentences
    from semlens.discover.seeds import parse_entries, seed_usages_from_annotations, seed_usages_from_table
    assert parse_entries('bubble, Cocoon*\nsilo, bubble') == ['bubble', 'Cocoon*', 'silo']
    with pytest.raises(ValueError):
        parse_entries('bubble, *')
    spans = seed_usages_from_table('sentence,start,end,corpus\n"In a bubble.",5,11,new\n')
    assert [(u.word, u.corpus) for u in spans] == [('bubble', 'new')]
    found = seed_usages_from_table('sentence\nBubbles and a bubble.\nNo match.\n', forms=['bubbl*'])
    assert [u.word for u in found] == ['Bubbles', 'bubble']
    with pytest.raises(ValueError):
        seed_usages_from_table('sentence\nA bubble.\n')
    data = load_from_sentences(['A social bubble.', 'A soap bubble.', 'A market bubble.'], 'bubble', ['x'] * 3)
    annotations = {0: {'social': 4}, 1: {'physical': None}, 2: {'social': 1, 'economic': 3}}
    assert len(seed_usages_from_annotations(data, annotations, ['social'])) == 2
    assert [u.sentence for u in seed_usages_from_annotations(data, annotations, ['social'], min_rating=2)] == ['A social bubble.']


def test_seed_keys_keep_their_order_and_scoring_adds_new_seeds_incrementally():
    from semlens.discover.knn import own_seed_positions, update_scoring
    from semlens.discover.seeds import SeedUsage, seed_keys
    contexts = [Context('a', 'bubble silo cocoon market'), Context('b', 'bubble silo tea cup')]
    sample = sample_occurrences(contexts, cap=5, min_count=1)
    usage = SeedUsage('In a bubble.', 5, 11)
    marked = int(rows_for_words(sample, ['tea'])[0])
    keys = seed_keys(sample, words=['bubble'], corpora=['a'], usages=[usage], marked_rows=[marked], accepted_words=['bubble'])
    bubble_a = int(rows_for_words(sample, ['bubble'])[0])
    bubble_b = int(rows_for_words(sample, ['bubble'])[1])
    assert keys == [('row', bubble_a), ('usage', usage.key), ('row', marked), ('row', bubble_b)]
    assert own_seed_positions(len(sample), keys)[marked] == 2
    rng = np.random.default_rng(2)
    vectors = unit(rng, len(sample), 16)
    candidates = ArrayCandidates(vectors)
    seeds = unit(rng, 6, 16)
    first = update_scoring(candidates, [('usage', str(i)) for i in range(4)], seeds[:4], 3, device='cpu')
    later = update_scoring(candidates, [('usage', str(i)) for i in range(6)], seeds, 3, first, device='cpu')
    full = update_scoring(candidates, [('usage', str(i)) for i in range(6)], seeds, 3, device='cpu')
    assert np.allclose(later.neighbours.distances, full.neighbours.distances, atol=1e-6)
    assert later.threshold == pytest.approx(full.threshold)


def test_spelling_overlaps_flag_shared_word_pieces():
    from semlens.discover.seeds import spelling_overlaps

    class Pieces:
        split = {'bubble': ['▁bu', 'bble'], 'bust': ['▁bu', 'st'], 'buick': ['▁bu', 'ick'], 'cocoon': ['▁cocoon'], 'dribble': ['▁dri', 'bble'],
                 'bus': ['▁bus'], 'b': ['▁b']}

        def tokenize(self, text):
            return self.split[text.strip()]

    notes = spelling_overlaps(Pieces(), ['bust', 'buick', 'cocoon', 'dribble', 'bus', 'bubble'], ['bubble'])
    assert set(notes) == {'bust', 'buick', 'dribble'} and 'bu' in notes['bust'] and 'bble' in notes['dribble']


def test_a_discovery_saves_reopens_and_hands_off_to_semlens(encoder, tmp_path):
    from semlens.discover.contextual import encode_sample
    from semlens.discover.project import (Discovery, handoff_sentences, handoff_with_vectors, list_discoveries, load_seed_vectors,
                                          save_seed_vectors, source_texts)
    from semlens.discover.seeds import SeedUsage
    from semlens.words import usage_words
    contexts = contexts_from_texts({'old': 'A bubble of friends. A cocoon of friends. Tea in a cup.',
                                    'new': 'A silo of friends. The bubble burst. Tea again.'})
    sample = sample_occurrences(contexts, cap=1, min_count=1)
    store = encode_sample(encoder, sample, tmp_path / 'd', layer=1)
    disc = Discovery(tmp_path / 'd', 'Bubbles', 'test', ['bubble'], None, [SeedUsage('In a bubble.', 5, 11, 'x', 'table')],
                     {3: True, 4: False}, {'cocoon': 'accepted'}, True)
    disc.save()
    again = Discovery.load(tmp_path / 'd')
    assert (again.marks, again.marked_rows, again.accepted_words, again.add_marked) == ({3: True, 4: False}, [3], ['cocoon'], True)
    assert again.seed_usages[0].word == 'bubble'
    assert [d['name'] for d in list_discoveries(tmp_path)] == ['Bubbles']
    save_seed_vectors(tmp_path / 'd', {'k': np.ones(4, np.float32)})
    assert np.allclose(load_seed_vectors(tmp_path / 'd')['k'], 1)
    data, embedded = handoff_with_vectors(store, ['bubble', 'cocoon', 'silo'])
    assert data.word == 'bubble' and data.forms == ['bubble', 'cocoon', 'silo'] and len(data) == len(embedded.embeddings)
    labels, order = usage_words(data)
    assert set(labels) == {'bubble', 'cocoon', 'silo'} and all(u.matched_form.casefold() == w for u, w in zip(data.usages, labels))
    assert np.allclose(embedded.embeddings.numpy(), np.asarray(store.vectors[rows_for_words(sample, ['bubble', 'cocoon', 'silo'])], np.float32))
    full = handoff_sentences(sample, ['bubble', 'cocoon'], corpora=['old'])
    assert full.corpus_labels == ['old'] and len(full) == 2
    assert source_texts(sample)['new'].split('\n\n')[1] == 'The bubble burst.'


def test_saved_corpus_vectors_are_reused_without_a_model(encoder, tmp_path):
    from semlens.corpus_store import CorpusCache, create_corpus_cache
    from semlens.discover.contextual import contexts_from_cache, encode_from_cache, encode_sample
    texts = {'old': 'The bank lent money. The river bank flooded.', 'new': 'The bank app failed. A river ran.'}
    path = create_corpus_cache(tmp_path / 'corpora', 'test', texts, encoder, layer=1)
    cache = CorpusCache(path)
    contexts = contexts_from_cache(cache, restrict='bank')
    assert len(contexts) == 3
    sample = sample_occurrences(contexts, cap=5, min_count=1)
    reused = encode_from_cache(cache, sample, tmp_path / 'reused')
    direct = encode_sample(encoder, sample, tmp_path / 'direct', layer=1)
    assert reused.valid.all() and reused.metadata['from_saved_corpora'] == 'test'
    assert np.allclose(np.asarray(reused.vectors, np.float32), np.asarray(direct.vectors, np.float32), atol=1e-2)


def test_vectorised_ppmi_counts_match_counting_pairs_one_by_one():
    from semlens.discover.static import cooccurrence_counts
    sentences = [['a', 'b', 'c', 'a'], ['b', 'c'], ['a', 'c', 'd', 'b', 'a', 'rare']]
    words = ['a', 'b', 'c', 'd']
    index = {w: i for i, w in enumerate(words)}
    expected = np.zeros((4, 4))
    # The earlier loop: each word against the words before it in the window, both ways round.
    for s in sentences:
        ids = [index.get(w, -1) for w in s]
        for i, a in enumerate(ids):
            for b in ids[max(0, i - 2):i]:
                if a >= 0 and b >= 0:
                    expected[a, b] += 1
                    expected[b, a] += 1
    assert np.array_equal(cooccurrence_counts(sentences, words, 2).toarray(), expected)
    assert len(ppmi_space(sentences, dimensions=2, window=2, min_count=1).words) == 5


def test_discover_runs_from_corpora_to_a_semlens_analysis(encoder, tmp_path, monkeypatch):
    """The whole loop in the app: read, seed, sample, embed, rank, accept a word, update, hand off."""
    import app.discover as discover
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(discover, 'DISCOVER_ROOT', tmp_path / 'discover')
    rng = np.random.default_rng(0)
    def corpus(words):
        lines = []
        for i in range(40):
            lines.append(f'They live in a {words[i % len(words)]} of {["friends", "allies", "fans"][i % 3]} and rarely meet anyone else number {i}.')
            lines.append(f'The {["bank", "market", "fund"][i % 3]} raised interest rates again in month {i}.')
        return ' '.join(lines)
    at = AppTest.from_file('app/app.py', default_timeout=120)
    at.session_state['source_texts'] = {'old': corpus(['bubble', 'cocoon']), 'new': corpus(['bubble', 'silo'])}
    # The test model, already loaded on the CPU: the app reuses a loaded model of the same name.
    at.session_state['_loaded_model'] = encoder
    at.run()
    at.sidebar.radio(key='section').set_value('Discover related words').run()
    click = lambda label: at.button[[b.label for b in at.button].index(label)].click().run()
    at.selectbox(key='dv_source_choice').set_value('Texts loaded in Analyse a word').run()
    click('Read the corpora')
    assert len(at.session_state['dv_contexts']) == 160
    at.button_group(key='dv_step').set_value('Seeds').run()
    at.text_input(key='dv_seed_words_input').set_value('bubble').run()
    assert any('one word' in w.value for w in at.warning)
    at.button_group(key='dv_step').set_value('Quick first pass').run()
    assert at.radio(key='dv_static_method').value == 'word2vec'
    at.radio(key='dv_static_method').set_value('ppmi').run()
    click('Run the quick first pass')
    assert not at.exception and not at.error and {'cocoon', 'silo'} & set(at.dataframe[0].value.Word)
    # The map of words: the shown words and the seed word, in the aligned static embeddings.
    at.session_state['dv_static_map'] = True
    at.run()
    words_map = json.loads(at.get('plotly_chart')[-1].proto.spec)
    labels = {w for trace in words_map['data'] for w in (trace.get('text') or [])}
    assert not at.exception and 'bubble' in labels and labels & {'cocoon', 'silo'}
    at.session_state['dv_static_map'] = False
    at.button_group(key='dv_step').set_value('Sample & embed').run()
    at.number_input(key='dv_min_count').set_value(2).run()
    at.selectbox(key='dv_model_preset').set_value('Another model…').run()
    at.text_input(key='dv_model_other').set_value('google/bert_uncased_L-2_H-128_A-2').run()
    click('Prepare the sample')
    click('Embed and start exploring')
    assert not at.exception and at.session_state['dv_step'] == 'Results'
    assert (tmp_path / 'discover').is_dir() and len(discover.list_discoveries(tmp_path / 'discover')) == 1
    click('Find related words')
    ranked = at.dataframe[0].value.Word.tolist()
    assert 'bubble' not in ranked and {'cocoon', 'silo'} <= set(ranked)
    # The map of usages, with the seeds ringed; then UMAP; then words as the average of their usages.
    at.session_state['dv_results_map'] = True
    at.run()
    usage_map = json.loads(at.get('plotly_chart')[-1].proto.spec)
    names = [trace.get('name') for trace in usage_map['data']]
    assert not at.exception and 'Seeds' in names and 'bubble' in names
    # AppTest does not keep a panel's open state between runs: open it again before each.
    at.radio(key='dv_usage_map_view').set_value('UMAP')
    at.session_state['dv_results_map'] = True
    at.run()
    assert not at.exception and json.loads(at.get('plotly_chart')[-1].proto.spec)['layout']['xaxis']['title']['text'] == 'UMAP 1'
    at.radio(key='dv_map_points').set_value('words')
    at.session_state['dv_results_map'] = True
    at.run()
    words_map = json.loads(at.get('plotly_chart')[-1].proto.spec)
    assert not at.exception and 'bubble' in {w for trace in words_map['data'] for w in (trace.get('text') or [])}
    at.session_state['dv_results_map'] = False
    at.run()
    at.radio(key='dv_add_marked').set_value('add').run()
    at.selectbox(key='dv_inspect_word').set_value('cocoon').run()
    at.button_group(key='dv_status_cocoon').set_value('accepted').run()
    update = next(b.label for b in at.button if b.label.startswith('Update the results with'))
    click(update)
    assert not at.exception and 'cocoon' not in at.dataframe[0].value.Word.tolist()
    # The next discovery is offered a name of its own.
    assert at.session_state['dv_name'].endswith(' 2')
    reopened = discover.Discovery.load(discover.list_discoveries(tmp_path / 'discover')[0]['folder'])
    assert reopened.word_status == {'cocoon': 'accepted'} and reopened.add_marked is True and reopened.seed_words == ['bubble']
    assert at.multiselect(key='dv_handoff_words').value == ['bubble', 'cocoon']
    click('Open in Analyse a word')
    assert not at.exception
    assert at.session_state['section'] == 'Analyse a word' and at.session_state['workspace'] == 'Explore'
    data = at.session_state['target_word_data']
    assert data.forms == ['bubble', 'cocoon'] and at.session_state['words_mode'] == 'words'
    assert at.session_state['embedded_usages'] is not None and len(at.session_state['embedded_usages'].embeddings) == len(data)


def test_the_threshold_compares_seeds_only_with_other_seed_words():
    from semlens.discover.knn import calibrate_threshold
    rng = np.random.default_rng(5)
    centre = unit(rng, 1, 32)[0]
    # Uses of one word sit very close together; the two words are further apart.
    a = centre + 0.05 * unit(rng, 10, 32) + 0.6 * unit(rng, 1, 32)[0]
    b = centre + 0.05 * unit(rng, 10, 32) + 0.6 * unit(rng, 1, 32)[0]
    seeds = np.vstack([a, b])
    seeds /= np.linalg.norm(seeds, axis=1, keepdims=True)
    within = calibrate_threshold(seeds, k=3)
    across = calibrate_threshold(seeds, k=3, groups=['a'] * 10 + ['b'] * 10)
    assert across > 5 * within
    assert calibrate_threshold(seeds, k=3, groups=['a'] * 20) is None


def test_corpora_with_different_vocabularies_get_aligned_spaces_of_one_size():
    from semlens.discover.static import build_spaces
    shared = 'tea milk sugar cake bread jam friends allies market bank fund price'.split()
    small = ' '.join(f'The bubble of {shared[i % 12]} and {shared[(i + 3) % 12]}.' for i in range(60))
    large = ' '.join(f'The silo of {shared[i % 12]} with {shared[(i + 5) % 12]} and {["rain", "snow", "wind", "sun"][i % 4]} {["north", "south"][i % 2]}.'
                     for i in range(120))
    contexts = contexts_from_texts({'small': small, 'large': large})
    spaces = build_spaces(contexts, ['small', 'large'], 'ppmi', dimensions=50, min_count=2)
    assert spaces['small'].vectors.shape[1] == spaces['large'].vectors.shape[1] < 50


def test_adding_seeds_after_starting_with_fewer_than_k_keeps_k_neighbours():
    rng = np.random.default_rng(7)
    vectors = unit(rng, 40, 16)
    seeds = unit(rng, 25, 16)
    candidates = ArrayCandidates(vectors)
    own = np.full(40, -1)
    own[0] = 1  # the first candidate is also the second seed
    few = nearest_seeds(candidates, seeds[:3], 10, own_seed=own, device='cpu')
    later = add_seeds(few, candidates, seeds[3:], 3, own_seed=own, device='cpu')
    full = nearest_seeds(candidates, seeds, 10, own_seed=own, device='cpu')
    assert later.distances.shape == full.distances.shape == (40, 10)
    assert np.allclose(later.knn_distance, full.knn_distance, atol=1e-6)
    assert np.isfinite(few.knn_distance).all() and np.isfinite(few.distances[1:, :3]).all()


def test_settings_survive_a_visit_to_the_other_section(tmp_path, monkeypatch):
    import app.discover as discover
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(discover, 'DISCOVER_ROOT', tmp_path / 'discover')
    at = AppTest.from_file('app/app.py', default_timeout=60).run()
    at.sidebar.slider(key='point_size').set_value(15).run()
    at.button_group(key='workspace').set_value('Explore').run()
    at.sidebar.radio(key='section').set_value('Discover related words').run()
    at.button_group(key='dv_step').set_value('Seeds').run()
    at.text_input(key='dv_seed_words_input').set_value('bubble, silo').run()
    at.sidebar.radio(key='section').set_value('Analyse a word').run()
    assert at.session_state['point_size'] == 15 and at.session_state['workspace'] == 'Explore'
    at.sidebar.radio(key='section').set_value('Discover related words').run()
    assert at.session_state['dv_seed_words_input'] == 'bubble, silo' and at.session_state['dv_step'] == 'Seeds'
    assert not at.exception


def test_a_discovery_name_without_letters_is_refused_not_a_crash(tmp_path, monkeypatch):
    import app.discover as discover
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(discover, 'DISCOVER_ROOT', tmp_path / 'discover')
    at = AppTest.from_file('app/app.py', default_timeout=60)
    at.session_state['source_texts'] = {'a': 'The bubble of friends. ' * 6, 'b': 'The silo of friends. ' * 6}
    at.run()
    at.sidebar.radio(key='section').set_value('Discover related words').run()
    at.selectbox(key='dv_source_choice').set_value('Texts loaded in Analyse a word').run()
    at.button[[b.label for b in at.button].index('Read the corpora')].click().run()
    at.button_group(key='dv_step').set_value('Sample & embed').run()
    at.selectbox(key='dv_model_preset').set_value('Another model…').run()
    at.text_input(key='dv_model_other').set_value('google/bert_uncased_L-2_H-128_A-2').run()
    at.text_input(key='dv_name').set_value('***').run()
    at.button[[b.label for b in at.button].index('Prepare the sample')].click().run()
    assert not at.exception and any('letter or digit' in e.value for e in at.error)
    embed = next(b for b in at.button if b.label == 'Embed and start exploring')
    assert embed.disabled


def test_spelling_notes_ignore_the_capitals_of_seed_words():
    from semlens.discover.seeds import spelling_overlaps

    class Cased:
        def tokenize(self, text):
            return {'bubble': ['▁bu', 'bble'], 'Bubble': ['▁Bub', 'ble'], 'bust': ['▁bu', 'st']}[text.strip()]

    assert spelling_overlaps(Cased(), ['bust'], ['Bubble']) == spelling_overlaps(Cased(), ['bust'], ['bubble']) != {}


def test_handing_off_words_absent_from_the_chosen_corpora_says_so():
    from semlens.discover.project import handoff_sentences
    sample = sample_occurrences(contexts_from_texts({'a': 'A bubble here.', 'b': 'A silo there.'}), cap=5, min_count=1)
    with pytest.raises(ValueError, match='No sentences containing'):
        handoff_sentences(sample, ['silo'], corpora=['a'])
