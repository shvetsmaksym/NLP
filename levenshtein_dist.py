import os
from functools import lru_cache
import numpy as np
import csv
from math import factorial, ceil
from itertools import combinations, product

from tqdm import tqdm

from Constants import LRU_CACHE_MAX_SIZE, LEVENSHTEIN_PAIRS, CSV_SEP, MAX_N_PAIRS_PER_FILE, TEMP_DIR
from logger import timer_func


def tail(x):
    return x[1:]


@lru_cache(maxsize=LRU_CACHE_MAX_SIZE)
def lev_recursive(a: str, b: str):
    if len(a) == 0:
        return len(b)
    elif len(b) == 0:
        return len(a)
    elif a[0] == b[0]:
        return lev_recursive(tail(a), tail(b))
    else:
        return 1 + min(lev_recursive(tail(a), b),
                       lev_recursive(a, tail(b)),
                       lev_recursive(tail(a), tail(b))
                       )


def levenshtein_iterative(a: str, b: str):
    """https://blog.paperspace.com/implementing-levenshtein-distance-word-autocomplete-autocorrect/"""
    a_len, b_len = len(a), len(b)
    distances = np.zeros((a_len + 1, b_len + 1))

    for t1 in range(a_len + 1):
        distances[t1][0] = t1

    for t2 in range(b_len + 1):
        distances[0][t2] = t2

    for t1 in range(1, a_len + 1):
        for t2 in range(1, b_len + 1):
            if a[t1 - 1] == b[t2 - 1]:
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
            else:
                x1 = distances[t1][t2 - 1]
                x2 = distances[t1 - 1][t2]
                x3 = distances[t1 - 1][t2 - 1]

                distances[t1][t2] = min(x1, x2, x3) + 1

    return distances[len(a)][len(b)]


def calculate_lev_closeness(a: str, b: str, mode='recursive'):
    if mode is 'recursive':
        lev = lev_recursive(a, b)
    else:
        lev = levenshtein_iterative(a, b)

    m_ = max(len(a), len(b))
    return (m_ - lev) / m_


@timer_func
def create_lev_csl_csv(bow, known_words):
    """Calculate levenshtein closeness for each pair of words and write them into temporary csv file."""
    word_pairs_iterator = combinations(bow, 2)
    iterator_length = factorial(len(bow)) / (2 * factorial(len(bow) - 2))
    word_pairs_iterator_2 = product(bow, known_words)
    iterator2_length = len(bow) * len(known_words)

    with open(LEVENSHTEIN_PAIRS, 'w', newline='\n', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=CSV_SEP)

        # Write header
        writer.writerow(['Word 1', 'Word 2', 'Levenshtein Closeness'])

        for w1, w2 in tqdm(word_pairs_iterator, total=iterator_length, desc='combinations of new words'):
            lev_cl = calculate_lev_closeness(w1, w2, mode='iterative')
            writer.writerow([w1, w2, lev_cl])

        # Calculate lev closeness with all the words out of the current doc, that have been seen earlier in other docs.
        for w1, w2 in tqdm(word_pairs_iterator_2, total=iterator2_length, desc='product of new words and old ones'):
            lev_cl = calculate_lev_closeness(w1, w2, mode='iterative')
            writer.writerow([w1, w2, lev_cl])
