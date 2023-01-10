from itertools import combinations, product
import numpy as np


def custom_iter(bow, known_words):
    """Custom iterator used for yielding only those pairs od words, for which levenshtein closeness
    haven't been calculated before (and not present in dictionary yet).
    While processing a new document, new words pairs (to calculate levenshtein closeness) =
    = two-elements combinations of words from a new document + product of words from anew document & all words present in dictionary.
    """

    combs = combinations(bow, 2)
    prod = product(bow, known_words)

    for el in combs:
        yield el

    for el in prod:
        yield el


def calculate_custom_iter_length(bow, known_words):
    bow_len = len(bow)
    return (np.math.factorial(bow_len) / (2 * np.math.factorial(bow_len - 2))) + (bow_len * len(known_words))

