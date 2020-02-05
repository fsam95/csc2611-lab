import argparse
import numpy as np
from numpy import dot, sqrt
from nltk.corpus import brown
from nltk.probability import FreqDist, ConditionalFreqDist, MLEProbDist, ConditionalProbDist
from nltk.lm import NgramCounter
from nltk import bigrams, ngrams
from scipy.stats import spearmanr, pearsonr
from sklearn.decomposition import PCA
from collections import Counter
    
import pandas as pd

from pkl_io import save_pkl, load_pkl
from plotting import plot_correlation

def cos(vA,vB):
    """
    regular cosine similarity for two vectors 'vA' and 'vB'
    """
    return dot(vA, vB) / (sqrt(dot(vA,vA)) * sqrt(dot(vB,vB)))

def n_least_frequent_words(all_words):
    counter = Counter(all_words)
    least_common_words_and_freqs = counter.most_common()[:-n-1:-1]
    return [word_and_freq[0] for word_and_freq in least_common_words_and_freqs] 

def n_most_frequent_words(all_words):
    counter = Counter(all_words)
    return [word_and_freq[0] for word_and_freq in counter.most_common(5000)]

def build_word_context_matrix(words_to_use, bigram_count):
    dimension = len(words_to_use)
    word_context_matrix = np.zeros((dimension, dimension))
    for i in range(len(words_to_use)):
        for j in range(len(words_to_use)):
            word, context = words_to_use[i], words_to_use[j]
            word_context_matrix[i][j] = bigram_count[(context, word)]
    save_pkl('word_context_matrix', word_context_matrix)
    return word_context_matrix

def calculate_pmi_matrix(wc_matrix): 
    total_count = wc_matrix.sum() # scalar
    word_count = wc_matrix.sum(axis = 0) # 5000 dim vector of word counts 
    # context_count = wc_matrix.sum(axis = 0) # 5000 dim vector of context counts
    prob_matrix = total_count * ((wc_matrix / word_count[:, np.newaxis]) / word_count)
    prob_matrix[prob_matrix == np.inf] = 0
    prob_matrix = np.nan_to_num(prob_matrix) 
    pmi_matrix = np.log(prob_matrix)
    return pmi_matrix
    # TODO: apply log
    # TODO: get rid of negative elements 

def lower_cased_brown_words():
    return [word.lower() for word in brown.words()]

def get_words_from_rg_also_in_brown(rg_df, brown_words, top_n_words):
    unique_brown_words = set(brown_words)
    top_n_words_set = set(top_n_words)
    all_rg_words = pd.concat([rg_df['word_1'], rg_df['word_2']]).values
    rg_also_in_brown = []
    for word in set(all_rg_words):
        if (word not in top_n_words_set) and (word in unique_brown_words):
            rg_also_in_brown.append(word)
    return rg_also_in_brown

def get_bigram_conditional_prob_brown():
    words = lower_cased_brown_words()
    all_bigrams = bigrams(words)
    return ConditionalProbDist(get_bigram_conditional_freq_brown(), MLEProbDist)

def get_bigram_freq_brown():
    words = lower_cased_brown_words()
    all_bigrams = bigrams(words)
    return FreqDist(all_bigrams)

def get_bigram_conditional_freq_brown():
    words = lower_cased_brown_words()
    all_bigrams = bigrams(words)
    return ConditionalFreqDist(all_bigrams)

def get_unigram_prob_brown():
    words = lower_cased_brown_words()
    num_words = len(words)
    counter = Counter(words)
    unigram_prob = {}
    for word in set(words):
        unigram_prob[word] = counter[word] / num_words
    return unigram_prob

def zero_out_negative_elements(pmi_matrix):
    """Sets all negative elements to zero
    
    Arguments:
        pmi_matrix {numpy matrix} -- A word-context matrix with PMI values
    
    Returns:
        [numpy matrix] 

    WARNING:    
        Mutates input matrix
    """
    pmi_matrix[pmi_matrix < 0] = 0
    return pmi_matrix

def build_word_index_mapping(words):
    word_indices = {}
    for i in range(len(words)):
        word_indices[words[i]] = i
    return word_indices
        
def build_ppmi_matrix(wc_matrix):
    pmi_matrix = calculate_pmi_matrix(wc_matrix)
    ppmi_matrix = zero_out_negative_elements(pmi_matrix)
    save_pkl('ppmi_matrix', ppmi_matrix)

def compute_pca_matrix(ppmi_matrix, dimensions):
    pca = PCA(dimensions)
    pca_transformed = pca.fit_transform(ppmi_matrix)
    return pca_transformed

def get_words_in_rg_and_wc(word_mapping):
    synonymy_pd = pd.read_csv('synonymy.csv')
    both = synonymy_pd.loc[synonymy_pd['word_1'].isin(word_mapping) & synonymy_pd['word_2'].isin(word_mapping)]
    return both

def compute_correlation(pair_sim_df, pca_matrix, word_index_map):
    calculate_cos = lambda word_one, word_two: cos(pca_matrix[word_index_map[word_one]], pca_matrix[word_index_map[word_two]])
    pair_sim_df['lsa sim'] = pair_sim_df[['word_1', 'word_2']].apply(lambda words: calculate_cos(words[0], words[1]), axis=1)
    print(pearsonr(pair_sim_df['similarity'], pair_sim_df['lsa sim']))
    plot_correlation(pair_sim_df, 'lsa sim', 'similarity', 'lsa_correlation')

    # print(pair_sim_df)

def get_words_in_W():
    words = lower_cased_brown_words()
    most_frequent_words = n_most_frequent_words(words)
    rg_df = pd.read_csv('synonymy.csv')
    words_from_rg_also_in_brown = get_words_from_rg_also_in_brown(rg_df, words, most_frequent_words)
    return set(most_frequent_words + words_from_rg_also_in_brown)

def main(args):
    if args.build_wc_matrix:
        words = lower_cased_brown_words()
        most_frequent_words = n_most_frequent_words(words)
        rg_df = pd.read_csv('synonymy.csv')
        words_from_rg_also_in_brown = get_words_from_rg_also_in_brown(rg_df, words, most_frequent_words)
        
        words_to_use = most_frequent_words + words_from_rg_also_in_brown 
        save_pkl('word_index_mapping', build_word_index_mapping(words_to_use))
        bigram_count = get_bigram_freq_brown()
        build_word_context_matrix(words_to_use, bigram_count)
    elif args.compute_pmi:
        wc_matrix = load_pkl('word_context_matrix')
        build_ppmi_matrix(wc_matrix)

        # bigram_prob = get_bigram_conditional_prob_brown()
        # bigram_prob['i'].prob('am')
        # unigram_prob = get_unigram_prob_brown()
        # most_frequent_words = n_most_frequent_words(words)
        # print(ppmi_matrix)
        # print(unigram_prob['the'].prob())
        #print(wc_matrix.sum())
    elif args.compute_pca_matrix:
        ppmi_matrix = load_pkl('ppmi_matrix')
        pca_matrix_ten = compute_pca_matrix(ppmi_matrix, 10)
        save_pkl('pca_ten', pca_matrix_ten)
        pca_matrix_hundred = compute_pca_matrix(ppmi_matrix, 100)
        save_pkl('pca_hundred', pca_matrix_hundred)
        pca_matrix_three_hundred = compute_pca_matrix(ppmi_matrix, 300)
        save_pkl('pca_three_hundred', pca_matrix_three_hundred)
    elif args.get_words_in_rg_and_wc_matrix:
        word_index_map = load_pkl('word_index_mapping')
        pairs_in_wc_matrix = get_words_in_rg_and_wc(word_index_map)
        save_pkl('rg_sim_values', pairs_in_wc_matrix)
    elif args.compute_correlation:
        word_index_map = load_pkl('word_index_mapping')
        rg_sim_df = load_pkl('rg_sim_values')
        # pca_ten = load_pkl('pca_ten')
        # compute_correlation(rg_sim_df, pca_ten, word_index_map)
        # pca_hundred = load_pkl('pca_hundred')
        # compute_correlation(rg_sim_df, pca_hundred, word_index_map)
        pca_three_hundred = load_pkl('pca_three_hundred')
        compute_correlation(rg_sim_df, pca_three_hundred, word_index_map)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--compute_pmi', action='store_true')
    argparser.add_argument('--build_wc_matrix', action='store_true')
    argparser.add_argument('--compute_pca_matrix', action='store_true')
    argparser.add_argument('--get_words_in_rg_and_wc_matrix', action='store_true')
    argparser.add_argument('--compute_correlation', action='store_true')
    main(argparser.parse_args())