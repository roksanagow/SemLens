"""AMD as the headline score, and AMD compared with the AMD within each corpus."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from test_app import seeded_app, widget, open_tab  # noqa: E402
from semlens.metrics import within_amd, within_amd_baseline, sample_indices, compute_all_metrics  # noqa: E402


def test_within_amd_leaves_each_usage_out():
    x = torch.tensor([[0.0], [1.0], [3.0]])
    # Nearest other usage: 1, 1, 2 → mean 4/3.
    assert within_amd(x, 'euclidean') == pytest.approx(4 / 3)
    assert math.isnan(within_amd(x[:1], 'euclidean'))
    # Two copies of a usage are each other's nearest neighbour, at distance 0.
    assert within_amd(torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), 'cosine') == pytest.approx(1 / 3)


def test_the_baseline_uses_the_sampled_usages_of_each_draw():
    torch.manual_seed(0)
    a, b = torch.randn(10, 4), torch.randn(6, 4)
    draws = sample_indices(10, 6, mode='repeated', repetitions=3, seed=1)
    in_a, in_b, baseline = within_amd_baseline(a, b, draws, 'cosine')
    assert in_a == pytest.approx(np.mean([within_amd(a[i]) for i, _ in draws]))
    assert in_b == pytest.approx(within_amd(b))  # the smaller group is used whole in every draw
    assert baseline == pytest.approx((in_a + in_b) / 2)


def test_amd_leads_and_is_compared_with_amd_within_each_corpus():
    at, data = seeded_app(3)
    widget(at, 'button_group', 'Workspace').set_value('Explore').run()
    open_tab(at, 'Compare two corpora')
    labels = [m.label for m in at.metric]
    assert labels[:3] == ['AMD', 'AMD: corpus_0 → corpus_1', 'AMD: corpus_1 → corpus_0']
    assert any(e.label == 'More scores: APD, PRT and SAMD' for e in at.expander)
    at.session_state['full_within'] = True
    at.run()
    assert not at.exception
    vectors = at.session_state['embedded_usages'].embeddings
    a, b = vectors[:4], vectors[4:8]
    amd = compute_all_metrics(a, b, ('corpus_0', 'corpus_1'), 'cosine')['amd']
    ratio = amd / ((within_amd(a) + within_amd(b)) / 2)
    assert any(f'AMD is **{ratio:.2f}×** the AMD within each corpus.' == m.value for m in at.markdown)
    # The same, for every pair, in the Scores tab.
    open_tab(at, 'Scores')
    at.session_state['within_scores'] = True
    at.run()
    heatmap = next(json.loads(c.proto.spec) for c in at.get('plotly_chart') if 'within' in json.loads(c.proto.spec)['layout']['title']['text'])
    assert heatmap['data'][0]['text'][0][1] == f'{ratio:.2f}×'


def test_each_score_help_shows_its_animation():
    from PIL import Image, ImageSequence
    from app.common import METRIC_HELP
    folder = Path(__file__).resolve().parents[1] / 'app' / 'data' / 'score_help'
    frames = {'amd': 3, 'samd': 2, 'prt': 3, 'apd': 2}
    for metric, count in frames.items():
        with Image.open(folder / f'{metric}.gif') as gif:
            assert sum(1 for _ in ImageSequence.Iterator(gif)) == count and gif.info.get('loop') == 0
        assert (folder / f'{metric}.gif').stat().st_size < 20_000
        assert METRIC_HELP[metric].count('](data:image/gif;base64,') == 1
