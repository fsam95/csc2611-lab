from meaning_construction import lower_cased_brown_words
from numpy.random import rand
from numpy import array
import numpy as np

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