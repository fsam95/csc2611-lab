from pkl_io import load_pkl, save_pkl
def save_embedding_index(words):
    word_to_index = {}
    for i in range(len(words)):
        word_to_index[words[i]] = i
    save_pkl('word_to_index_diachronic', word_to_index)

def load_embedding_index():
    return load_pkl('word_to_index_diachronic')

def load_words():
    embeddings_map = load_pkl('data')
    return (embeddings_map['w'])

def load_decades():
    embeddings_map = load_pkl('data')
    return embeddings_map['d']

def load_embedding_matrix():
    embeddings_map = load_pkl('data')
    return embeddings_map['E']