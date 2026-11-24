"""Discover related words: words used like a set of seed words or seed usages, across corpora.

- :mod:`.vocabulary`: contexts, restricting texts, counting words, even sampling under a cap.
- :mod:`.contextual`: XLM-RoBERTa vectors for the sampled occurrences, kept on disk (word vectors, or slot vectors with the word masked).
- :mod:`.knn`: kNN-AMD and the share of usages near the concept, with incremental seeds and an optional background correction.
- :mod:`.static`: a cheap first pass with aligned static vectors (word2vec or PPMI + SVD).
- :mod:`.seeds`: seed words and seed usages, their order, and notes on words spelled like a seed.
- :mod:`.project`: a discovery saved on disk, and the hand-off to an ordinary SemLens analysis.
"""
