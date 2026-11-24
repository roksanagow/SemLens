"""Cost estimates and the checks that hold back heavy work."""
import pytest
import torch
from streamlit.testing.v1 import AppTest

from semlens.data_loading import load_from_sentences
from semlens.embeddings import embedded_usages_from_precomputed
from semlens.resources import (Cost, Resources, assess, comparison_cost, count_tokens, embedding_cost, format_bytes,
                               format_duration, rough_tokens_per_second)

GB = 1e9
LAPTOP = Resources(total_memory=16 * GB, available_memory=8 * GB, free_disk=100 * GB)


def test_verdicts_follow_the_agreed_thresholds():
    assert assess(Cost('small', disk_bytes=1 * GB, memory_bytes=2 * GB, seconds=60), LAPTOP).level == 'ok'
    # Confirm above 2 GB of disk, 40% of memory, or ten minutes.
    assert assess(Cost('disk', disk_bytes=3 * GB), LAPTOP).level == 'confirm'
    assert assess(Cost('memory', memory_bytes=7 * GB), LAPTOP).level == 'confirm'
    assert assess(Cost('time', seconds=11 * 60), LAPTOP).level == 'confirm'
    # Block when free disk would fall below 5 GB, or memory above 75% of the RAM.
    blocked = assess(Cost('disk', download_bytes=2 * GB, disk_bytes=94 * GB), LAPTOP)
    assert blocked.level == 'block' and 'disk space' in blocked.reasons[0]
    assert assess(Cost('memory', memory_bytes=13 * GB), LAPTOP).level == 'block'
    assert not assess(Cost('memory', memory_bytes=13 * GB), LAPTOP).allowed


def test_estimates_for_embedding_and_comparing():
    texts = ['one two three four five'] * 100
    assert count_tokens(texts) == 700  # about 1.4 tokens per word without a tokenizer
    cost = embedding_cost('Embedding', texts=texts, parameters=279e6, hidden=768, layers=12, device='cpu', vectors=100,
                          batch_size=16, max_length=256)
    assert cost.rough_time and cost.memory_bytes > 279e6 * 4 and cost.seconds == pytest.approx(700 / rough_tokens_per_second(279e6, 'cpu'))
    measured = embedding_cost('Embedding', texts=texts, parameters=279e6, hidden=768, layers=12, device='cpu', vectors=100,
                              tokens_per_second=350)
    assert not measured.rough_time and measured.seconds == pytest.approx(2.0)
    # SAMD sorts every pair: 30,000 × 30,000 usages is about 14 GB.
    big = comparison_cost([(100, 100), (30_000, 30_000)])
    assert big.memory_bytes == pytest.approx(30_000 ** 2 * 16) and assess(big, LAPTOP).level == 'block'
    assert format_bytes(2_500_000_000) == '2.5 GB' and format_duration(30) == 'under a minute' and format_duration(5400) == 'about 1.5 hours'


def seeded(monkeypatch, resources, usages=8):
    import app.costs as costs
    monkeypatch.setattr(costs, 'system_resources', lambda path=None: resources)
    monkeypatch.setattr(costs, 'model_profile', lambda model_id: {'parameters': 560e6, 'hidden': 1024, 'layers': 24, 'downloaded': True,
                                                                  'download_bytes': 0.0, 'extra_disk_bytes': 0.0})
    at = AppTest.from_file('app/app.py', default_timeout=30).run()
    data = load_from_sentences([f'The bank had example {i}.' for i in range(usages)], 'bank', ['a', 'b'] * (usages // 2))
    at.session_state['target_word_data'] = data
    at.session_state['corpus_order'] = data.corpus_labels
    return at


def embed_button(at):
    return next(b for b in at.button if b.label == 'Embed loaded usages')


def test_embedding_shows_its_cost_and_starts_freely_when_small(monkeypatch):
    at = seeded(monkeypatch, LAPTOP)
    at.run()
    assert not at.exception
    assert any(c.value.startswith('Embedding the loaded usages:') for c in at.caption)
    assert not embed_button(at).disabled


def test_embedding_needs_a_tick_when_large_and_is_blocked_when_it_would_freeze_the_computer(monkeypatch):
    small_computer = Resources(total_memory=4 * GB, available_memory=2 * GB, free_disk=100 * GB)
    at = seeded(monkeypatch, small_computer)
    at.run()
    # The model alone takes 2.2 GB: over 40% of 4 GB, so a tick is needed first.
    assert any('Embedding the loaded usages' in w.value for w in at.warning)
    assert embed_button(at).disabled
    next(c for c in at.checkbox if c.label == 'I understand, continue').check().run()
    assert not embed_button(at).disabled
    tiny = Resources(total_memory=2 * GB, available_memory=1 * GB, free_disk=100 * GB)
    at = seeded(monkeypatch, tiny)
    at.run()
    assert any('Not started' in e.value for e in at.error) and embed_button(at).disabled
    assert not any(c.label == 'I understand, continue' for c in at.checkbox)


def test_comparisons_too_large_for_memory_are_not_computed(monkeypatch):
    import app.costs as costs
    import app.ui_explore as ui
    monkeypatch.setattr(costs, 'system_resources', lambda path=None: Resources(total_memory=4 * GB, available_memory=2 * GB, free_disk=50 * GB))
    # Pretend each pair of usages takes far more memory, so a small test corpus is "too large".
    monkeypatch.setattr(ui, 'comparison_cost', lambda sizes, action: Cost(action, memory_bytes=10 * GB))
    at = AppTest.from_file('app/app.py', default_timeout=30).run()
    data = load_from_sentences([f'The bank had example {i}.' for i in range(9)], 'bank', ['a', 'b', 'c'] * 3)
    for k, v in {'target_word_data': data, 'embedded_usages': embedded_usages_from_precomputed(data, torch.randn(9, 6)),
                 'corpus_order': data.corpus_labels, 'corpus_labels': data.corpus_labels, 'data_revision': 1}.items():
        at.session_state[k] = v
    at.session_state['workspace'] = 'Explore'
    at.run()
    assert not at.exception
    assert any('Comparing every pair of corpora' in e.value and 'Not started' in e.value for e in at.error)
    # The pair comparison keeps its selectors, so another pair can be chosen, but computes nothing.
    assert any(e.value.startswith('**Not started') and 'Comparing a and b' in e.value for e in at.error)
    assert not at.metric
