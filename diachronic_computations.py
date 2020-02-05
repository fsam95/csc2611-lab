from numpy import array
from sklearn.metrics.pairwise import cosine_similarity

def get_all_embeddings_for_decade(embedding_matrix, decade):
    """Returns all the word embeddings for a given decade (i.e. a 2000 x 300 matrix)
    
    Arguments:
        embedding_matrix {[type]} -- [description]
        decade {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    return embedding_matrix[:,decade,:]

def get_centroid_for_decade(embedding_matrix, decade, indices):
    embedding_matrix_for_decade = get_all_embeddings_for_decade(embedding_matrix, decade)
    return 1/(len(indices)) * (embedding_matrix_for_decade[indices].sum(axis=0))

def get_density_for_decade(embedding_matrix, decade, indices, word_embedding_first_decade):
    embedding_matrix_for_decade = get_all_embeddings_for_decade(embedding_matrix, decade)
    pairwise_sim = cosine_similarity(array([word_embedding_first_decade]), embedding_matrix_for_decade[indices])[0]
    return pairwise_sim.sum() / len(indices)
