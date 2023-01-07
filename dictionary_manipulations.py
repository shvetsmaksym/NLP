import json
import numpy as np
import pandas as pd

from Constants import LEVENSHTEIN_CLOSENESS, DICTIONARY, LEVENSHTEIN_PAIRS, CSV_SEP
from levenshtein_dist import calculate_lev_closeness, create_lev_csl_csv
from logger import timer_func


@timer_func
def update_metadata_with_given_pairs(metadata: dict, df: pd.DataFrame) -> dict:
    """Note: this is the only method where metadata["Groups"] used.
    The reason is that it really matters whether w1, w2 are among keys or in any group's list."""
    for row in df.iterrows():
        w1, w2 = row[1][0], row[1][1]
        w_ = {w1, w2}

        # If any of the words was seen before, create a new group consisted only of them.
        n_intersected = len(w_.intersection(metadata["KnownWords"]))
        if not n_intersected:
            metadata["Groups"][w1] = [w1, w2]
            metadata['Matches'][w1] = w1
            metadata['Matches'][w2] = w1

        # If both words were seen before, they are already in dictionary - skip them.
        elif n_intersected == 2:  # w1 and w2 are already in other groups
            continue

        # if w1 is in KnownWords, then it could be whether:
        # 1) key in Groups -> then we add w2 to that group (condition LEV(key, w2) > THRESHOLD is already fulfilled) or
        # 2) one of the words in any group -> then we have to check if the LEV(key, w2) > THRESHOLD:
        # 2.1) if yes, add w2 to that group
        # 2.2) create a new group consisted only of 'w2'
        elif w1 in metadata["KnownWords"]:

            if w1 in metadata["Groups"].keys():
                metadata["Groups"][w1] = [w2]
                metadata['Matches'][w2] = w1
            else:
                for key, group_words in metadata["Groups"].items():
                    if w1 in group_words:
                        lev_closeness = calculate_lev_closeness(key, w2)
                        if lev_closeness > LEVENSHTEIN_CLOSENESS:
                            group_words.append(w2)
                            metadata['Matches'][w2] = w1
                        else:
                            metadata["Groups"][w2] = [w2]
                            metadata['Matches'][w2] = w2
                        break

        # The same as the above elif for w1
        elif w2 in metadata["KnownWords"]:

            if w2 in metadata["Groups"].keys():
                metadata["Groups"][w2] = [w1]
                metadata['Matches'][w1] = w2
            for key, group_words in metadata["Groups"].items():
                if w2 in group_words:
                    lev_closeness = calculate_lev_closeness(key, w2)
                    if lev_closeness > LEVENSHTEIN_CLOSENESS:
                        group_words.append(w1)
                        metadata['Matches'][w1] = w2
                    else:
                        metadata["Groups"][w1] = [w1]
                        metadata['Matches'][w1] = w1
                    break

            metadata['Matches'][w1] = w2

        metadata["KnownWords"].update(w_)

    return metadata


@timer_func
def update_metadata_with_single_words(metadata: dict, words_left) -> dict:
    """These are 'levenshteinelly far' words."""
    for w in words_left:
        if w not in metadata["KnownWords"]:
            metadata["Groups"][w] = [w]
            metadata["Matches"][w] = w
            metadata["KnownWords"].add(w)

    return metadata


def update_dictionary(bow):
    try:
        with open(DICTIONARY, 'r') as f:
            metadata = json.load(f)
            metadata["KnownWords"] = set(metadata["KnownWords"])

    except FileNotFoundError:
        metadata = {"KnownWords": set(), "Groups": dict(), "Matches": dict()}  # Groups = {word: [word, close_word_1, close_word_2, ...]}

    # Create temporary csv file with lev closeness of new pairs
    create_lev_csl_csv(bow, known_words=metadata['KnownWords'])

    # Load levenshtein pairs
    df = pd.read_csv(LEVENSHTEIN_PAIRS, sep=CSV_SEP, encoding='cp1250')

    df = df.sort_values(by='Levenshtein Closeness', ascending=False)
    df_close_pairs = df[['Word 1', 'Word 2']][(df['Levenshtein Closeness'] > LEVENSHTEIN_CLOSENESS) & (df['Levenshtein Closeness'] < 1)]
    df_left = df[['Word 1', 'Word 2']][(df['Levenshtein Closeness'] <= LEVENSHTEIN_CLOSENESS)]
    words_left = set(np.array(df_left).reshape(-1))

    metadata = update_metadata_with_given_pairs(metadata, df_close_pairs)
    metadata = update_metadata_with_single_words(metadata, words_left)

    metadata['KnownWords'] = list(metadata['KnownWords'])

    js = json.dumps(metadata, ensure_ascii=False)
    with open(DICTIONARY, 'w') as file:
        file.write(js)

    # Tests
    print(set(metadata['KnownWords']) == (set(metadata['Matches'])))
