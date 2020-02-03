from synonymy_handler import get_rg_brown_sim_df
from scipy.stats import spearmanr, pearsonr
from gensim.models import KeyedVectors 
from numpy import dot, sqrt

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



if __name__ == '__main__':
    get_rg_word_vectors()



