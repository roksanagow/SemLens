"""Tests for the SemLens core library."""

import numpy as np
import pandas as pd
import pytest
import torch

from semlens.utils import (
    edit_distance,
    find_word_position,
    bold_word_in_sentence,
    format_hover_text,
    get_palette,
    palette_names,
)
from semlens.data_loading import (
    UsageInstance,
    TargetWordData,
    load_from_sentences,
    load_from_sentences_with_positions,
    load_from_raw_text,
)
from semlens.metrics import (
    calculate_apd,
    calculate_prt,
    calculate_amd,
    calculate_directional_amd,
    calculate_samd,
    compute_all_metrics,
    compute_per_definition_metrics,
)
from semlens.spaces import project_to_definition_space, split_by_corpus
from semlens.reduction import reduce_to_2d, available_methods
from semlens.lda import lda_projection, lda_definition_weights
from semlens.definitions import format_definitions, _clean_wikitext


# ===== utils =====

class TestEditDistance:
    def test_identical(self):
        assert edit_distance("hello", "hello") == 0

    def test_single_substitution(self):
        assert edit_distance("hello", "hallo") == 1

    def test_empty(self):
        assert edit_distance("", "abc") == 3
        assert edit_distance("abc", "") == 3

    def test_deletion(self):
        assert edit_distance("abcde", "abce") == 1


class TestFindWordPosition:
    def test_exact_match(self):
        s, e = find_word_position("The bank is closed", "bank")
        assert s == 4 and e == 8

    def test_case_insensitive(self):
        s, e = find_word_position("The Bank is closed", "bank")
        assert s == 4 and e == 8

    def test_edit_distance_fallback(self):
        # "banks" is the closest to "bank"
        s, e = find_word_position("The banks are closed", "bank")
        assert "banks" == "The banks are closed"[s:e]


class TestTextFormatting:
    def test_bold_word(self):
        result = bold_word_in_sentence("the bank is open", "bank")
        assert "<b>bank</b>" in result

    def test_hover_text(self):
        result = format_hover_text("the bank is open", "bank", "1800s")
        assert "[1800s]" in result
        assert "<b>bank</b>" in result


class TestPalettes:
    def test_default_palette(self):
        p = get_palette()
        assert len(p) >= 5

    def test_named_palette(self):
        for name in palette_names():
            p = get_palette(name)
            assert len(p) >= 4

    def test_unknown_falls_back(self):
        p = get_palette("nonexistent")
        assert p == get_palette()


# ===== data_loading =====

class TestDataLoading:
    def test_load_from_sentences(self):
        sents = ["The bank is closed", "Walking along the river bank"]
        data = load_from_sentences(sents, "bank", ["modern", "old"])
        assert len(data) == 2
        assert data.word == "bank"
        assert set(data.corpus_labels) == {"modern", "old"}
        # Check positions were found
        for u in data.usages:
            assert u.sentence[u.char_start : u.char_end].lower() == "bank"

    def test_load_with_positions(self):
        data = load_from_sentences_with_positions(
            ["The bank is closed"], "bank", [4], [8], ["modern"]
        )
        assert data.usages[0].char_start == 4
        assert data.usages[0].char_end == 8

    def test_corpus_helpers(self):
        data = load_from_sentences(
            ["sent A1", "sent A2", "sent B1"],
            "sent",
            ["A", "A", "B"],
        )
        assert data.indices_for_corpus("A") == [0, 1]
        assert data.indices_for_corpus("B") == [2]
        assert len(data.usages_for_corpus("A")) == 2

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            load_from_sentences(["a", "b"], "a", ["x"])

    def test_load_from_raw_text(self):
        old = "The bank of the river was steep. The bank was muddy. Birds flew overhead."
        new = "I went to the bank for a loan. The bank approved my mortgage. It was sunny."
        data = load_from_raw_text({"old": old, "new": new}, "bank")
        assert len(data) == 4  # 2 from old (river bank, muddy bank), 2 from new
        assert set(data.corpus_labels) == {"old", "new"}
        for u in data.usages:
            assert "bank" in u.sentence.lower()

    def test_load_from_raw_text_no_matches(self):
        text = "The sky is blue. It was a fine day."
        with pytest.raises(ValueError, match="No sentences"):
            load_from_raw_text({"c1": text, "c2": text}, "bank")


# ===== metrics =====

class TestMetrics:
    """Test metrics on synthetic embeddings with known properties."""

    def _identical_corpora(self):
        """Two corpora with identical embeddings → zero change."""
        embs = torch.randn(10, 64)
        return embs, embs.clone()

    def _shifted_corpora(self):
        """Two corpora with a clear shift."""
        torch.manual_seed(42)
        c1 = torch.randn(20, 64)
        c2 = torch.randn(20, 64) + 3.0  # large shift
        return c1, c2

    def test_apd_identical(self):
        # APD between identical corpus copies equals the within-corpus
        # average pairwise distance (not zero, since points differ from
        # each other — only self-distance is zero).
        c1, c2 = self._identical_corpora()
        score = calculate_apd(c1, c2)
        # Should be the same as APD(c1, c1) since c2 is a clone
        score_self = calculate_apd(c1, c1)
        assert abs(score - score_self) < 1e-5

    def test_prt_identical(self):
        c1, c2 = self._identical_corpora()
        score = calculate_prt(c1, c2)
        # PRT returns 1/(cos+eps), so for identical centroids → near 1
        # Actually PRT = 1/(cos_sim + eps).  cos_sim=1 → PRT ≈ 1.
        # But the paper convention is cosine *distance*. Let me check.
        # Looking at the original: (1.0 / (cos + eps))
        # For identical: cos=1 → 1/1 = 1.  This is not cosine distance.
        # This matches the paper repo exactly.  It's inverted cosine sim.
        assert score < 1.1  # near 1 for identical

    def test_amd_identical(self):
        c1, c2 = self._identical_corpora()
        score = calculate_amd(c1, c2)
        assert abs(score) < 1e-5

    def test_samd_identical(self):
        c1, c2 = self._identical_corpora()
        score = calculate_samd(c1, c2)
        assert abs(score) < 1e-5

    def test_shifted_scores_positive(self):
        c1, c2 = self._shifted_corpora()
        assert calculate_apd(c1, c2) > 0.1
        assert calculate_amd(c1, c2) > 0.01
        assert calculate_samd(c1, c2) > 0.01

    def test_directional_amd(self):
        c1, c2 = self._shifted_corpora()
        d_12 = calculate_directional_amd(c1, c2)
        d_21 = calculate_directional_amd(c2, c1)
        # Both should be positive for shifted corpora
        assert d_12 > 0
        assert d_21 > 0
        # Symmetric AMD should be their average
        assert abs(calculate_amd(c1, c2) - (d_12 + d_21) / 2) < 1e-6

    def test_compute_all_metrics(self):
        c1, c2 = self._shifted_corpora()
        result = compute_all_metrics(c1, c2, ("early", "late"))
        assert "apd" in result
        assert "prt" in result
        assert "amd" in result
        assert "samd" in result
        assert "amd_early_to_late" in result
        assert "amd_late_to_early" in result

    def test_per_definition_metrics(self):
        c1 = torch.randn(15, 5)  # 15 usages, 5 definitions
        c2 = torch.randn(15, 5)
        labels = [f"def_{i}" for i in range(5)]
        df = compute_per_definition_metrics(c1, c2, labels)
        assert len(df) == 5
        assert "definition" in df.columns
        assert "amd" in df.columns


# ===== spaces =====

class TestSpaces:
    def test_definition_space_shape(self):
        usages = torch.randn(20, 768)
        defs = torch.randn(5, 768)
        projected = project_to_definition_space(usages, defs)
        assert projected.shape == (20, 5)

    def test_definition_space_values(self):
        """Cosine distance should be in [0, 2]."""
        usages = torch.randn(10, 64)
        defs = torch.randn(3, 64)
        projected = project_to_definition_space(usages, defs)
        assert projected.min() >= -0.01  # cosine dist ≥ 0
        assert projected.max() <= 2.01   # cosine dist ≤ 2

    def test_split_by_corpus(self):
        embs = torch.randn(10, 64)
        indices = {"A": [0, 1, 2, 3], "B": [4, 5, 6, 7, 8, 9]}
        a, b = split_by_corpus(embs, indices, "A", "B")
        assert a.shape[0] == 4
        assert b.shape[0] == 6


# ===== reduction =====

class TestReduction:
    def test_pca_2d(self):
        X = torch.randn(30, 64)
        coords = reduce_to_2d(X, method="pca")
        assert coords.shape == (30, 2)

    def test_tsne_2d(self):
        X = torch.randn(30, 64)
        coords = reduce_to_2d(X, method="tsne")
        assert coords.shape == (30, 2)

    def test_small_n_tsne(self):
        """t-SNE should adapt perplexity for small N."""
        X = torch.randn(5, 64)
        coords = reduce_to_2d(X, method="tsne")
        assert coords.shape == (5, 2)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            reduce_to_2d(torch.randn(10, 64), method="magic")

    def test_available_methods(self):
        methods = available_methods()
        assert "pca" in methods


# ===== lda =====

class TestLDA:
    def test_lda_2d_output(self):
        torch.manual_seed(0)
        # Two clearly separated clusters
        c1 = torch.randn(20, 10)
        c2 = torch.randn(20, 10) + 2.0
        embs = torch.cat([c1, c2])
        labels = ["old"] * 20 + ["new"] * 20

        result = lda_projection(embs, labels)
        assert result.coords_2d.shape == (40, 2)
        assert result.n_lda_components == 1

    def test_lda_weights(self):
        torch.manual_seed(0)
        c1 = torch.randn(20, 5)
        c2 = torch.randn(20, 5) + torch.tensor([0, 0, 0, 0, 3.0])
        embs = torch.cat([c1, c2])
        labels = ["old"] * 20 + ["new"] * 20
        defs = [f"def_{i}" for i in range(5)]

        result = lda_projection(embs, labels, feature_names=defs)
        weights_df = lda_definition_weights(result, defs)

        assert len(weights_df) == 5
        assert "definition" in weights_df.columns
        assert "weight" in weights_df.columns
        # The 5th dimension has the biggest shift, so should have highest weight
        top_def = weights_df.iloc[0]["definition"]
        assert top_def == "def_4"


# ===== definitions =====

class TestDefinitions:
    def test_format_definitions(self):
        defs = format_definitions("bank", [
            "a financial institution",
            "bank: the side of a river",
            "  a financial institution  ",  # duplicate after strip
        ])
        assert all(d.startswith("bank:") for d in defs)
        # Should deduplicate
        assert len(defs) == 2

    def test_clean_wikitext(self):
        raw = "A [[financial]] {{institution|something|bank}} that ''holds'' money"
        clean = _clean_wikitext(raw)
        assert "[[" not in clean
        assert "{{" not in clean
        assert "''" not in clean
