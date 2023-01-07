import os

LOG_DIR = 'logs'
TEMP_DIR = 'temp'
DOC_EXAMPLES_DIR = 'doc_examples'

PROCESSED_DOC1 = os.path.join(TEMP_DIR, 'processed_document_1.txt')
PROCESSED_DOC2 = os.path.join(TEMP_DIR, 'processed_document_2.txt')

LEVENSHTEIN_PAIRS = os.path.join(TEMP_DIR, 'levenshtein_pairs.csv')
LEVENSHTEIN_MATRIX = os.path.join(TEMP_DIR, 'levenshtein_matrix.csv')
DICTIONARY = 'dictionary_1.json'
USER_DICTIONARY = os.path.join(TEMP_DIR, 'user_dictionary.json')

CSV_SEP = ';'

######################

MAX_N_PAIRS_PER_FILE = int(5e6)  # we run out of RAM if keep open such large file.
LRU_CACHE_MAX_SIZE = 1024**3  # Number of calls that recursive function will remember. This depends on RAM or L1/L2 cache available
WORD_PAIRS_PER_PARTITION = 1000

######################
LEVENSHTEIN_CLOSENESS = 0.7  # threshold for levenshtein closeness metric
