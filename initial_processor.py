import json
import re

from Constants import DICTIONARY
from logger import log


def initial_normalization(filepath) -> str:
    with open(filepath, 'r', encoding='utf-8') as f:
        txt_file_1 = f.read()

    processed = re.sub(r'[\t\n]', ' ', txt_file_1)
    processed = re.sub(r'[^\w\s]', '', processed)
    processed = re.sub(r'(\s)+', ' ', processed)
    processed = processed.lower()

    return processed


def get_bag_of_words(filepath):
    try:
        with open(DICTIONARY, 'r') as f:
            js = json.load(f)
            known_words = js['KnownWords']
    except FileNotFoundError:
        known_words = set()

    normalized_doc = initial_normalization(filepath=filepath)
    words = normalized_doc.split()
    bow = set(words)

    # Extract only words that haven't been seen before.
    bow_new = bow.difference(known_words)

    log(f'Document: {filepath}\tWords in document: {len(words)}\tUnique words: {len(bow)}\t New words: {len(bow_new)}')

    return words, list(bow_new)
