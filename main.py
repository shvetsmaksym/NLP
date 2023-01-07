import os

from Constants import DOC_EXAMPLES_DIR, PROCESSED_DOC1, PROCESSED_DOC2
from dictionary_manipulations import update_dictionary
from initial_processor import *


def update_words_with_matches(words: list):
    with open(DICTIONARY, 'r') as f:
        metadata = json.load(f)
        matches = metadata["Matches"]

    return list(map(lambda w: matches[w], words))


if __name__ == "__main__":
    filepath1 = os.path.join(DOC_EXAMPLES_DIR, 'text1.txt')
    filepath2 = os.path.join(DOC_EXAMPLES_DIR, 'text2.txt')
    filepath3 = os.path.join(DOC_EXAMPLES_DIR, 'text3.txt')
    filepath4 = os.path.join(DOC_EXAMPLES_DIR, 'sienkiewicz_henryk_pan_michael.txt')

    # Get normalized document (list of words in order such as in an initial text) and bag of words for each document
    norm_doc1_words, bow_doc1 = get_bag_of_words(filepath1)
    norm_doc2_words, bow_doc2 = get_bag_of_words(filepath4)

    # Update the dictionary with unseen words from doc1 and doc2
    if bow_doc1:
        update_dictionary(bow=bow_doc1)
    if bow_doc2:
        update_dictionary(bow=bow_doc2)

    processed_words_1 = update_words_with_matches(norm_doc1_words)
    processed_doc_1 = ' '.join(processed_words_1)

    processed_words_2 = update_words_with_matches(norm_doc2_words)
    processed_doc_2 = ' '.join(processed_words_2)

    with open(PROCESSED_DOC1, 'w') as p1:
        p1.write(processed_doc_1)

    with open(PROCESSED_DOC2, 'w') as p2:
        p2.write(processed_doc_2)

    print('DONE.')

