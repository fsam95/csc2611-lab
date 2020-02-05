from meaning_construction import lower_cased_brown_words
from read_analogy_file import _is_semantic, _collect_word_pairs, _include_analogy_sample
from synchronic import _compute_accuracy

from numpy.random import rand
from numpy import array
import numpy as np
import pandas as pd
from nltk import bigrams, ngrams
from nltk.corpus import brown
from nltk.probability import FreqDist

def test_all_words_lower():
    assert all([(not word.isalpha()) or word.islower() for word in lower_cased_brown_words()])

# def test_pmi_matrix_implementation():

# def test_ppmi_matrix_implementation():

def test_newaxis_operator():
    a = rand(3,3)
    x = np.array([0.25, 2, 10])
    row_wise_multiplication = a * x[:, None]  # TODO: add this to notes
    for i in range(a.shape[0]):
        assert all(row_wise_multiplication[i] == x[i] * a[i])

def test_row_vector_multiplication():
    a = rand(3,3)
    x = np.array([0.25, 2, 10])
    row_wise_broadcast_multiplication = a * x # TODO: add this to notes
    for i in range(a.shape[0]):
        assert all(row_wise_broadcast_multiplication[i] == x * a[i])

def test_zero_axis_sum_is_sum_across_rows():
    a = rand(3,3)
    sum_axis_zero = a.sum(axis=0)
    for i in range(a.shape[0]):
        assert sum_axis_zero[i] == sum(a[:,i])

def test_one_axis_sum_is_sum_across_columns():
    a = rand(3,3)
    sum_axis_zero = a.sum(axis=1)
    for i in range(a.shape[0]):
        assert sum_axis_zero[i] == sum(a[i])

def test_is_semantic():
    semantic = ': capital-common-countries'
    syntactic = ': gram1-adjective-to-adverb'
    assert _is_semantic(semantic)
    assert not _is_semantic(syntactic)

def test_split_word_line():
    line = 'Athens Greece Bangkok Thailand\n'
    words = _collect_word_pairs(line)
    assert len(words) == 4
    assert all([word.islower() for word in words])

def test_include_analogy_sample():
    analogy_words = 'blah blah blah blah'.split(' ')
    assert _include_analogy_sample(analogy_words, None)
    assert not _include_analogy_sample(analogy_words, set([]))
    assert _include_analogy_sample(analogy_words, set(['blah']))

def test_accuracy():
    df = pd.DataFrame({'lsa_correct': [True, True, False]})
    assert _compute_accuracy(df) == 2/3

def test_bigram_count():
    words = 'words in the brown corpus. words in the brown corpus'.split(' ')
    all_bigrams = bigrams(words)
    freq_dist = FreqDist(all_bigrams)
    assert freq_dist[('words', 'in')] == 2
    assert freq_dist[('corpus', 'brown')] == 0
    assert freq_dist[('corpus', 'blah')] == 0

# def test_nltk_brown_words_is_list():
#     assert type(brown.words) == list