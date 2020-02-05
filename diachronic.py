import argparse
from numpy import dot, sqrt, array, flip
from sklearn.metrics.pairwise import cosine_similarity

from pkl_io import load_pkl, save_pkl
from plotting import plot_time_course
from diachronic_storage import save_embedding_index, load_embedding_index, load_words, load_decades, load_embedding_matrix
from diachronic_computations import get_all_embeddings_for_decade, get_centroid_for_decade, get_density_for_decade
from scipy.stats import spearmanr, pearsonr

def cos(vA,vB):
    """
    regular cosine similarity for two vectors 'vA' and 'vB'
    """
    if sum(vA) == 0 or sum(vB) == 0: 
        return 0
    return dot(vA, vB) / (sqrt(dot(vA,vA)) * sqrt(dot(vB,vB)))

def eval_change_for_word(embedding_matrix, word_index,  word, start_i, end_i):
    embeddings_for_word = embedding_matrix[word_index]
    word_start = embeddings_for_word[start_i]
    word_end = embeddings_for_word[end_i]
    change = cos(word_start, word_end)
    # print("Change for {}: {}".format(word, change))
    return change

def get_nearest_neighbours(k, word, index, embedding_matrix, word_index_map):
    pairwise_similarities = (cosine_similarity(array([embedding_matrix[index]]), embedding_matrix))[0]
    pairwise_similarities = pairwise_similarities.argsort()
    nearest_neigbour_indices = pairwise_similarities[-k - 1: -1]
    return nearest_neigbour_indices


def method_one_cos_dist(embedding_matrix, word_index_map):
    semantic_change_vals = []
    for word, index in word_index_map.items():
        change = eval_change_for_word(embedding_matrix, index, word, 0, 9)
        semantic_change_vals.append((word, change)) 
    sorted_semantic_change_vals = sorted(semantic_change_vals, key=lambda x: x[1])
    words_only = lambda all_sem_change_vals: [change_word[0] for change_word in all_sem_change_vals]
    print("Most changing words: {}\n".format(words_only(sorted_semantic_change_vals[0:20])))
    print("Least changing words: {}".format(words_only(sorted_semantic_change_vals[-20:])))
    return sorted_semantic_change_vals

def method_two_neighbour_centroid(embedding_matrix, word_index_map):
    semantic_change_vals = []
    embedding_matrix = array(embedding_matrix)
    embeddings_for_first_decade = get_all_embeddings_for_decade(embedding_matrix, 0)
    for word, index in word_index_map.items():
        nearest_neighbour_indices = get_nearest_neighbours(1, word, index, embeddings_for_first_decade, word_index_map)
        centroid_start_decade = get_centroid_for_decade(embedding_matrix, 0, nearest_neighbour_indices)
        centroid_end_decade = get_centroid_for_decade(embedding_matrix, 9, nearest_neighbour_indices)
        change = cos(centroid_start_decade, centroid_end_decade)
        semantic_change_vals.append((word, change))
    sorted_semantic_change_vals = sorted(semantic_change_vals, key=lambda x: x[1])
    words_only = lambda all_sem_change_vals: [change_word[0] for change_word in all_sem_change_vals]
    print("Most changing words: {}\n".format(words_only(sorted_semantic_change_vals[0:20])))
    print("Least changing words: {}".format(words_only(sorted_semantic_change_vals[-20:])))
    # print("Most changing words: {}\n".format(sorted_semantic_change_vals[0:20]))
    # print("Least changing words: {}".format(sorted_semantic_change_vals[-20:]))
    return sorted_semantic_change_vals

def method_three_neighbourhood_density(embedding_matrix, word_index_map):
    # semantic_change_vals = []
    # embedding_matrix = array(embedding_matrix)
    # embeddings_for_first_decade = get_all_embeddings_for_decade(embedding_matrix, 0)
    # for word, index in word_index_map.items():
    #     nearest_neighbour_indices = get_nearest_neighbours(10, word, index, embeddings_for_first_decade, word_index_map)
    #     word_embedding = embeddings_for_first_decade[index]
    #     density_for_start_decade = get_density_for_decade(embedding_matrix, 0, nearest_neighbour_indices, word_embedding)
    #     density_for_end_decade = get_density_for_decade(embedding_matrix, 9, nearest_neighbour_indices, word_embedding)
    #     change = abs(density_for_end_decade - density_for_start_decade)
    #     semantic_change_vals.append((word, change))
    # sorted_semantic_change_vals = sorted(semantic_change_vals, key=lambda x: x[1], reverse=True)
    # print("Most changing words: {}\n".format(sorted_semantic_change_vals[0:20]))
    # print("Least changing words: {}".format(sorted_semantic_change_vals[-20:]))
    # return sorted_semantic_change_vals

    semantic_change_vals = []
    embedding_matrix = array(embedding_matrix)
    embeddings_for_first_decade = get_all_embeddings_for_decade(embedding_matrix, 0)
    embeddings_for_end_decade = get_all_embeddings_for_decade(embedding_matrix, 9)
    for word, index in word_index_map.items():
        start_nearest_neighbour_indices = get_nearest_neighbours(3, word, index, embeddings_for_first_decade, word_index_map)
        start_word_embedding = embeddings_for_first_decade[index]
        density_for_start_decade = get_density_for_decade(embedding_matrix, 0, start_nearest_neighbour_indices, start_word_embedding)

        end_nearest_neighbour_indices = get_nearest_neighbours(3, word, index, embeddings_for_end_decade, word_index_map)
        end_word_embedding = embeddings_for_end_decade[index]
        density_for_end_decade = get_density_for_decade(embedding_matrix, 9, end_nearest_neighbour_indices, end_word_embedding)

        change = abs(density_for_end_decade - density_for_start_decade)
        semantic_change_vals.append((word, change))
    sorted_semantic_change_vals = sorted(semantic_change_vals, key=lambda x: x[1], reverse=True)
    # print("Most changing words: {}\n".format(sorted_semantic_change_vals[0:20]))
    # print("Least changing words: {}".format(sorted_semantic_change_vals[-20:]))
    words_only = lambda all_sem_change_vals: [change_word[0] for change_word in all_sem_change_vals]
    print("Most changing words: {}\n".format(words_only(sorted_semantic_change_vals[0:20])))
    print("Least changing words: {}".format(words_only(sorted_semantic_change_vals[-20:])))
    return sorted_semantic_change_vals
    
def calculate_mean_change(embedding_matrix, t, word_index_map):
    semantic_change_vals = []
    for word, index in word_index_map.items():
        change = eval_change_for_word(embedding_matrix, index, word, t, t + 1)
        semantic_change_vals.append(change)
    return (array(semantic_change_vals).mean(), array(semantic_change_vals).std())
 
    
def _calculate_consecutive_decades_change(embedding_matrix, word, index):        
    consecutive_decade_changes = []
    for t in range(9):
        consecutive_decade_changes.append(eval_change_for_word(embedding_matrix, index, word, t, t + 1))
    return consecutive_decade_changes

def changepoint_detection(embedding_matrix, word_index_map, desired_words):
    mean_changes = []
    for t in range(9):
        mean_changes.append(calculate_mean_change(embedding_matrix, t, word_index_map))
    mean_changes, std_changes = zip(*mean_changes)
    mean_changes, std_changes = array(mean_changes), array(std_changes)
    # print(mean_changes)
    for word in desired_words: 
        change_over_decades = _calculate_consecutive_decades_change(embedding_matrix, word, word_index_map[word])
        change_over_decades = array(change_over_decades)
        standardized_changes = (change_over_decades - mean_changes) / std_changes
        standardized_changes[standardized_changes > -3] = 0
        change_x = []
        change_y = []
        for i in range(len(standardized_changes)):
            if standardized_changes[i] != 0:
                change_x.append(i)
                change_y.append(change_over_decades[i])


        plot_time_course(list(range(9)), change_over_decades, change_x, change_y, word)

    


# def method_three_neighbour_density(embedding_matrix, word_index_map):

def main(args):
    if args.load_embedding_index:
        # print(load_decades())
        save_embedding_index(load_words())
    elif args.eval_cos_word:
        word = args.eval_cos_word[0]
        word_index_map = load_embedding_index()
        embedding_matrix = load_embedding_matrix()
        eval_change_for_word(embedding_matrix, word_index_map[word], word, 0, 9)
    elif args.method_one:
        word_index_map = load_embedding_index()
        embedding_matrix = load_embedding_matrix()
        method_one_cos_dist(embedding_matrix, word_index_map)
    elif args.method_two: 
        word_index_map = load_embedding_index()
        embedding_matrix = load_embedding_matrix()
        method_two_neighbour_centroid(embedding_matrix, word_index_map)
    elif args.method_three: 
        word_index_map = load_embedding_index()
        embedding_matrix = load_embedding_matrix()
        method_three_neighbourhood_density(embedding_matrix, word_index_map)
    elif args.eval_correlation_all_methods: 
        word_index_map = load_embedding_index()
        embedding_matrix = load_embedding_matrix()
        get_change = lambda change_results: [word_change[1] for word_change in change_results]
        method_1_sorted_change = method_one_cos_dist(embedding_matrix, word_index_map)
        method_2_sorted_change = method_two_neighbour_centroid(embedding_matrix, word_index_map)
        method_3_sorted_change = method_three_neighbourhood_density(embedding_matrix, word_index_map)

        print("Method 1 and 2 correlation: {}".format(pearsonr(get_change(method_1_sorted_change), get_change(method_2_sorted_change))))

        print("Method 1 and 3 correlation: {}".format(pearsonr(get_change(method_1_sorted_change), get_change(method_3_sorted_change))))

        print("Method 2 and 3 correlation: {}".format(pearsonr(get_change(method_2_sorted_change), get_change(method_3_sorted_change))))
        # print("Method 1 and 2 correlation".format(spearmanr(get_change(method_1_sorted_change), get_change(method_2_sorted_change))))
    elif args.changepoint_detection:
        word_index_map = load_embedding_index()
        embedding_matrix = load_embedding_matrix()
        changepoint_detection(embedding_matrix, word_index_map, ['techniques', 'skills', 'mcgraw'])
        

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--load_embedding_index', action='store_true')
    argparser.add_argument('--eval_cos_word', nargs=1)
    argparser.add_argument('--method_one', action='store_true')
    argparser.add_argument('--method_two', action='store_true')
    argparser.add_argument('--method_three', action='store_true')
    argparser.add_argument('--eval_correlation_all_methods', action='store_true')
    argparser.add_argument('--changepoint_detection', action='store_true')
    # argparser.add_argument('--load_embedding_index', action='store_true')



    main(argparser.parse_args())
    # load_embeddings()
