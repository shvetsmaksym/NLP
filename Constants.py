import os

ENCODING = 'utf-8'

RESULTS_FOLDER = 'results'
LOG_DIR = 'logs'
TEMP_DIR = 'temp'
DOC_EXAMPLES_DIR = 'doc_examples'
LEV_CL_SPLIT_FILES = os.path.join(TEMP_DIR, 'LevenshteinSplitFiles')

PROCESSED_DOC1 = os.path.join(TEMP_DIR, 'processed_document_1.txt')
PROCESSED_DOC2 = os.path.join(TEMP_DIR, 'processed_document_2.txt')

LEV_PAIRS_BASENAME = 'levenshtein_pairs'
LEVENSHTEIN_PAIRS = os.path.join(TEMP_DIR, LEV_PAIRS_BASENAME + '.csv')
LEVENSHTEIN_MATRIX = os.path.join(TEMP_DIR, 'levenshtein_matrix.csv')
DICTIONARY = 'dictionary_3.json'
USER_DICTIONARY = os.path.join(TEMP_DIR, 'user_dictionary.json')

CSV_SEP = ';'
HEADER_CSV = ['Word 1', 'Word 2', 'Levenshtein Closeness']

######################

MAX_N_PAIRS_PER_POOL = int(5e6)  # Hard to find a rule based on RAM available for this number
LRU_CACHE_MAX_SIZE = None  # Number of calls recursive function will remember. Depends on RAM available
WORD_PAIRS_PER_PARTITION = 1000

######################
LEVENSHTEIN_CLOSENESS = 0.7  # threshold for levenshtein closeness metric
