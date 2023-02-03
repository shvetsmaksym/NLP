# NLP

## Goal
To measure distances (similarities) between two or more documents.

## Pileline
### 1. Document processing

1.1. Initial normalization (remove punctuations, white characters, etc.)

1.2. Bag-of-words extraction (split each normalized document into words)

1.3. Calculation of Levenshtein closeness between every pair of words and writing them into csv (temporary file `levenstein_pairs.csv`)

* consider only new words (not found in existing dictionary)

* consider all words if there is no dictionary created yet.

1.4. To create/update the dictionary using data from `levenstein_pairs.csv`.
Dictionary consists of:

* `KnownWords`: set - used for deciding if we need to calculate levenshtein distance for this word
and include it into `levenstein_pairs.csv`
* `Groups`: [keyword of a group: [list of words in this group], ...] single group consists of keyword
and list of words, for which LEV_CLOSENESS(keyword, word) > THRESHOLD.
* `Matches` {word: keyword for its group}

Note: `LEV_CLOSENESS = (max_word_length - LEV_DIST) / max_word_length`,

where `max_word_length` - length of the longest word (out of two).

### 2. Calculation of document similarities
The output is matrix [`n x n`] of cosine similarities, where `n` - number of documents that we compare.

2.1. Metrics

In order to calculate cosine similarities we have to find:
* Bag of words from all documents
* `TF` (Term Frequencies for each document) - matrix [`m x n`], where `m` - size of bag of words, `n` - number of documents
* `IDF` (Inverse Document Frequencies) - vector of `m` size, where `m` - size of bag of words.
* `TFIDF = TF * IDF`

2.2. Cosine similarity
* If we compare two documents, we use TF for cosine similarity calculation 
* If we compare more than two documents, we use TFIDF for cosine similarity calculation

## Run algorithm
Put text documents, that you want to measure similarities between, into `doc_examples` folder.

Next from main project path run: <br>
`python main.py -d1 text-doxument-1.txt -d2 text-doxument-2.txt` or <br>
`python main.py -fp folder-with-text-docs`

After algorithm running is complete, check `DocumentCosSimilarities.csv` in `results` folder.


## TODO: 
`levenshtein_processes.py` is not efficiently optimized yet.
It utilizes a multiprocessing module's features such as queue and pool as well as 
listener, workers and jobs for writing calculated Levenshtein distance for each pair of words.
However, the combination of caching in recursive function and multiple processes causes 
excessive utilization of RAM and in uncontrolled way. This issue has been tried to resolve by
reducing the load on each worker, meaning that the whole set of two-words combinations 
should be divided into smaller partitions (small enough to not running out of RAM) at the beginning. 
However, this manipulation resulted in huge performance drop, eventually making multiprocess approach much 
slower than single process.

Recursive function with `lru_cache()` causes running out of RAM even in single-process approach. That's why it only
pays off to use it for processing the documents ~< 2000 unique words (for 16 GB RAM).

Taking everything above into consideration, as a stable approach I can only use an iterative
single process calculation of levenshtein closeness.