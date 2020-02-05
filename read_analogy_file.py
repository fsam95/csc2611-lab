import pandas as pd
def _is_semantic(line):
    return line[2: 6] != 'gram'
        
def _collect_word_pairs(line):
    return [word.lower() for word in line.strip().split(' ')]

def _include_analogy_sample(analogy_words, words_to_include):
    if words_to_include is None:
        return True
    return all([word in words_to_include for word in analogy_words])

def read_analogy_file(words_to_include=None):
    """Reads the analogy data into a pandas DataFrame  that 
    has the following columns: [word1, word2, word3, word4, is_semantic]

    Args:
        words_to_include {{str}}: only include the analogy if all of the words are in this set
    """
    word_pairs = []
    categories = []
    is_semantic_category = False
    with open('word-test.v1.txt', 'r') as analogy_data_f:
        for line in analogy_data_f:
            if line.startswith(':'):
                is_semantic_category = True if _is_semantic(line) else False
            else:
                analogy_words = _collect_word_pairs(line)
                if _include_analogy_sample(analogy_words, words_to_include):
                    word_pairs.append(analogy_words)
                    categories.append(is_semantic_category)
    first_pair_word_one, first_pair_word_two, second_pair_word_one, second_pair_word_two = zip(*word_pairs)
    return pd.DataFrame({'word1': first_pair_word_one, 'word2': first_pair_word_two, 'word3': second_pair_word_one, 'word4': second_pair_word_two, 'is_semantic': categories})
