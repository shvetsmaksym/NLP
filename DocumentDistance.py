import json
import os.path
from itertools import combinations
import numpy as np
import pandas as pd

from Constants import DICTIONARY, ENCODING, RESULTS_FOLDER
from DocumentProcessing.dictionary_manipulations import update_dictionary
from DocumentProcessing.initial_processor import get_bag_of_words, clear_dir


def update_words_with_matches(words: list):
    with open(DICTIONARY, 'r', encoding=ENCODING) as f:
        metadata = json.load(f)
        matches = metadata["Matches"]

    return list(map(lambda w: matches[w], words))


def process_documents(**kwargs):
    """
    :param kwargs:
    (doc1_path, doc2_path) or folder_path.
    :return: List of normalized by Levenshtein closeness documents (each split into list of words)
    e.g. [[word1_from_doc1, word2_from_doc1, ...], [word1_from_doc2, word2_from_doc2, ...], ..., [word1_from_docn, word2_from_docn, ...]]

    1. Updates dictionary with new words (from given documents).
    2. Processes given documents replacing words with their matches from dictionary.
    """

    # --- Update Dictionary & get normalized (by Levenshtein closeness) document---
    if 'doc1_path' in kwargs and 'doc2_path' in kwargs:
        norm_doc1_words, bow = get_bag_of_words(kwargs['doc1_path'], mode='text_file')
        norm_doc2_words, bow_ = get_bag_of_words(kwargs['doc2_path'], mode='text_file')

        bow = set(bow)
        bow.update(set(bow_))

        if bow:
            update_dictionary(bow=bow)

        processed_doc1_words = update_words_with_matches(norm_doc1_words)
        processed_doc2_words = update_words_with_matches(norm_doc2_words)

        levenshtein_normalized_docs = [processed_doc1_words, processed_doc2_words]

    elif 'folder_path' in kwargs:
        levenshtein_normalized_docs = []
        norm_docs_words, bow = get_bag_of_words(kwargs['folder_path'], mode='folder')
        if bow:
            update_dictionary(bow=bow)

        for w_ in norm_docs_words:
            levenshtein_normalized_docs.append(update_words_with_matches(w_))

    else:
        raise AttributeError('process_documents function did not get expected arguments: (doc1_path, doc2_path) or folder_path.')

    return levenshtein_normalized_docs


def calculate_tf_idf(normalized_docs):
    """
    :param: normalized_docs: list of normalized documents (each given as list of split words) to calculate their similarities

    Calculates TF (Term Frequency) & IDF (Inverse Document Frequency - only for folder with > 2 documents).
    Note: in this step we consider only a bag of words obtained from documents pointed by parameters (doc1_path, doc2_path) or folder_path.
    We don't consider the whole dictionary, because it would generate unnecessary dimensions (words not present in documents that we process)
    while calculating TF & IDF.
    """

    # --- Calculate TF ---
    docs_number = len(normalized_docs)
    docs_names = ['Document_{}'.format(i) for i in range(docs_number)]
    bow = set(sum(normalized_docs, []))

    t_ = [[normalized_docs[i].count(w) for i in range(docs_number)] for w in bow]  # rows: documents; cols: word's occurences
    word_counts = np.array(t_)
    df_word_counts = pd.DataFrame(word_counts, columns=[i for i in range(docs_number)], index=bow)
    df_cos_similarities = pd.DataFrame(np.ones((docs_number, docs_number)), index=docs_names, columns=docs_names)  # used np.ones, because we don't need to calculate similarity of document with itself (values on a main diagonal stay 1)

    TF = df_word_counts / df_word_counts.sum()
    if docs_number == 2:
        doc1_loc, doc2_loc = 'Document_{}'.format(1), 'Document_{}'.format(0)
        cos_smlrt = cos_similarity(TF[1], TF[0])
        df_cos_similarities.loc[doc2_loc, doc1_loc] = cos_smlrt
        df_cos_similarities.loc[doc1_loc, doc2_loc] = cos_smlrt

    else:
        DOCS_CONTAIN_WORD = df_word_counts.apply(func=np.count_nonzero, axis=1)
        IDF = np.log(docs_number / DOCS_CONTAIN_WORD)
        TF_IDF = pd.DataFrame()
        for col in range(docs_number):
            TF_IDF['Document_{}'.format(col)] = TF[col] * IDF

        for comb in combinations(range(docs_number), 2):
            doc1_loc, doc2_loc = 'Document_{}'.format(comb[0]), 'Document_{}'.format(comb[1])
            cos_smlrt = cos_similarity(TF_IDF[doc1_loc], TF_IDF[doc2_loc])
            df_cos_similarities.loc[doc2_loc, doc1_loc] = cos_smlrt
            df_cos_similarities.loc[doc1_loc, doc2_loc] = cos_smlrt

    clear_dir(RESULTS_FOLDER)
    df_cos_similarities.to_csv(os.path.join(RESULTS_FOLDER, 'DocumentCosSimilarities.csv'), sep=';')


def cos_similarity(v1, v2):
    dot_prod = np.dot(v1, v2)

    mag1 = np.sqrt(sum([x**2 for x in v1]))
    mag2 = np.sqrt(sum([x**2 for x in v2]))

    if mag1 == 0:
        return None

    if mag2 == 0:
        return None

    return dot_prod / (mag1 * mag2)


