import argparse

from scipy.stats import spearmanr, pearsonr
from gensim.models import KeyedVectors 
from numpy import dot, sqrt

from read_analogy_file import read_analogy_file

from plotting import plot_correlation
from synonymy_handler import get_rg_brown_sim_df
from meaning_construction import get_words_in_W
from pkl_io import load_pkl, save_pkl

def cos(vA,vB):
    """
    regular cosine similarity for two vectors 'vA' and 'vB'
    """
    return dot(vA, vB) / (sqrt(dot(vA,vA)) * sqrt(dot(vB,vB)))

def get_rg_word_vectors():
    sim_df = get_rg_brown_sim_df()
    model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, limit=500000)

    calculate_w2v_sim = lambda word_1, word_2: cos(model[word_1], model[word_2]) 
    sim_df['w2v_sim'] = sim_df[['word_1', 'word_2']].apply(lambda words: calculate_w2v_sim(words[0], words[1]), axis=1)
    print(pearsonr(sim_df['similarity'], sim_df['w2v_sim']))
    plot_correlation(sim_df, 'w2v_sim', 'similarity', 'w2v_correlation')

def compute_lsa_analogy(df, candidate_words):
    lsa_matrix = load_pkl('pca_three_hundred')
    word_index = load_pkl('word_index_mapping')

    def lsa_analogy_sample(word_1, word_2, word_3, target):
        closest_word = None
        largest_cos = float('-inf')
        comparison = lsa_matrix[word_index[word_2]] - lsa_matrix[word_index[word_1]] + lsa_matrix[word_index[word_3]]
        for word in candidate_words:
            curr_cos = cos(comparison, lsa_matrix[word_index[word]])
            if  curr_cos > largest_cos and word not in [word_1, word_2, word_3]:
                largest_cos = curr_cos
                closest_word = word
        # print(word_1, word_2, word_3, target)
        # print(closest_word)
        return closest_word == target

    df['lsa_correct'] = df[['word1', 'word2', 'word3', 'word4']].apply(lambda words: lsa_analogy_sample(words[0], words[1], words[2], words[3]), axis=1)
    return df

def _compute_accuracy(df, label, semantic):
    df = df.loc[df['is_semantic'] == semantic]
    total = len(df)
    correct = len(df.loc[(df[label] == True) & (df['is_semantic'] == semantic)])
    return correct/total

def do_analogy_test_lsa():
    candidate_words = get_words_in_W()
    df = load_pkl('analogy_data')
    compute_lsa_analogy(df, candidate_words)

    lsa_semantic_accuracy = _compute_accuracy(df, 'lsa_correct', True)
    print("LSA semantic analogy accuracy: {}".format(lsa_semantic_accuracy))
    lsa_syntactic_accuracy = _compute_accuracy(df, 'lsa_correct', False)
    print("LSA syntactic analogy accuracy: {}".format(lsa_syntactic_accuracy))

def compute_w2v_analogy(model, analogy_df, candidate_words):
    def w2v_analogy_sample(word_1, word_2, word_3, target):
        comparison = model[word_2] - model[word_1] + model[word_3]
        closest_word = None
        largest_cos = float('-inf')
        for word in candidate_words:
            if word in model or word == target:
                curr_cos = cos(comparison, model[word])
                if  curr_cos > largest_cos and word not in [word_1, word_2, word_3]:
                    largest_cos = curr_cos
                    closest_word = word
        print(word_1, word_2, word_3, target)
        print(closest_word)
        return closest_word == target
    analogy_df['w2v_correct'] = analogy_df[['word1', 'word2', 'word3', 'word4']].apply(lambda words: w2v_analogy_sample(words[0], words[1], words[2], words[3]), axis=1)

def do_analogy_test_w2v():
    candidate_words = get_words_in_W()
    df = read_analogy_file(get_words_in_W())
    save_pkl('analogy_data', df)
    print("Number of semantic analogy test examples: {}".format(len(df.loc[df['is_semantic'] == True])))
    print("Number of syntactic analogy test examples: {}".format(len(df.loc[df['is_semantic'] == False])))
    model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, limit=1000000)
    compute_w2v_analogy(model, df, candidate_words)
    w2v_semantic_accuracy = _compute_accuracy(df, 'w2v_correct', True)
    print("w2v semantic analogy accuracy: {}".format(w2v_semantic_accuracy))
    w2v_syntactic_accuracy = _compute_accuracy(df, 'w2v_correct', False)
    print("w2v syntactic analogy accuracy: {}".format(w2v_syntactic_accuracy))

def main(args):
    if args.do_analogy_test:
        do_analogy_test_w2v()
        do_analogy_test_lsa()
    elif args.compute_w2v_sim:
        get_rg_word_vectors()
    
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--compute_w2v_sim', action='store_true')
    argparser.add_argument('--do_analogy_test', action='store_true')
    main(argparser.parse_args())

    # get_rg_word_vectors()