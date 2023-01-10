from functools import lru_cache
import numpy as np
import csv
from math import factorial

from tqdm import tqdm

from Constants import LRU_CACHE_MAX_SIZE, LEVENSHTEIN_PAIRS, CSV_SEP, HEADER_CSV
from CustomIterator import custom_iter, calculate_custom_iter_length
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


def levenshtein_closeness_iterative(a: str, b: str):
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

    m_ = max(len(a), len(b))

    return round((m_ - distances[len(a)][len(b)]) / m_, 2)


@timer_func
def create_lev_csl_csv(bow, known_words):
    """Calculate levenshtein closeness for each pair of words and write them into temporary csv file."""
    word_pairs_iterator = custom_iter(bow, known_words)
    iterator_length = calculate_custom_iter_length(bow, known_words)

    with open(LEVENSHTEIN_PAIRS, 'w', newline='\n', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=CSV_SEP)

        # Write header
        writer.writerow(HEADER_CSV)

        for w1, w2 in tqdm(word_pairs_iterator, total=iterator_length, desc='Calculating Levenshtein closeness...'):
            lev_cl = levenshtein_closeness_iterative(w1, w2)
            writer.writerow([w1, w2, lev_cl])
