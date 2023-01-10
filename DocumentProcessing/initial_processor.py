import json
import os
import re
import shutil

import PyPDF2

from Constants import DICTIONARY, ENCODING
from logger import log


def clear_dir(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)

    if not os.path.exists(directory):
        os.mkdir(directory)


def initial_normalization(filepath) -> str:
    if filepath.endswith('.txt'):
        with open(filepath, 'r', encoding=ENCODING) as f:
            text = f.read()

    elif filepath.endswith('.pdf'):
        with open(filepath, 'rb') as pdfFile:
            pdf_reader = PyPDF2.PdfReader(pdfFile)
            pages_txt = [page.extract_text() for page in pdf_reader.pages]
            text = ' '.join(pages_txt)

    processed = re.sub(r'[\t\n]', ' ', text)
    processed = re.sub(r'[^\w\s]', '', processed)
    processed = re.sub(r'(\s)+', ' ', processed)
    processed = processed.lower()

    return processed


def get_bag_of_words(filepath, mode='text_file'):
    """mode:
    'text_file: get bag of words for given file
    folder: get bag of words for all documents in given folder
    """
    try:
        with open(DICTIONARY, 'r', encoding=ENCODING) as f:
            js = json.load(f)
            known_words = js['KnownWords']
    except FileNotFoundError:
        known_words = set()

    if mode == 'text_file':
        normalized_text = initial_normalization(filepath=filepath)
        split_words = normalized_text.split()
        bow = set(split_words)

        # Extract only words that haven't been seen before.
        bow_new = bow.difference(known_words)
        log(f'Document: {filepath}\tWords in document: {len(split_words)}\tUnique words: {len(bow)}\t New words: {len(bow_new)}')

    elif mode == 'folder':
        split_words = []
        files = list(filter(lambda f: re.search('\.(txt|pdf)$', f), os.listdir(filepath)))
        for file in files:
            normalized_text = initial_normalization(filepath=os.path.join(filepath, file))
            split_words.append(normalized_text.split())

        all_docs_words = sum(split_words, [])
        bow = set(all_docs_words)

        # Extract only words that haven't been seen before.
        bow_new = bow.difference(known_words)
        log(f'Folder: {filepath}\tWords in all documents: {len(all_docs_words)}\tUnique words: {len(bow)}\t New words: {len(bow_new)}')

    else:
        raise AttributeError('get_bag_of_words function got unexpected argument mode (expect \'text_file\' or \'folder\').')

    return split_words, list(bow_new)
