# NLP

Note: LevDistMultiprocessing.py is not used yet, because it's not efficiently optimized.
It utilizes a multiprocessing module's features such as queue, pool as well as 
listener, workers and jobs for writing calculated Levenshtein distance for each pair of words.
However, the combination of caching in recursive function and multiple processes causes 
excessive utilization of RAM and in uncontrolled way. This issue has been tried to resolve by
reducing the load on each worker, meaning that the whole set of two-words combinations 
should be divided into smaller partitions (small enough to not running out of RAM) at the beginning,
which again causes immediate RAM allocation problem (while working with documents >2000 unique words with 16 GB RAM laptop)

Recursive function with lru_cache() causes running out of RAM even in single-process approach. That's why it only
pays off to use it for processing the documents < 2000 unique words (for 16 GB RAM).

### Pileline scratch
1. normalization

2. bag-of-words extraction

3. calculation of levenshtein distances and writing them into csv (temporary file e.g. `levenstein_pairs.csv`)

* consider only new words (not found in existing dictionary)

* consider all words if there is no dictionary created yet.

4. Create/update the dictionary using data from `levenstein_pairs.csv`.
Dictionary consists of:

* `KnownWords`: set - used for deciding if we need to calculate levenshtein distance for this word
and include it into `levenstein_pairs.csv`
* `Groups`: [keyword of a group: [list of words in this group], ...] single group consists of keyword
and list of words, for which LEV(keyword, word) > THRESHOLD.
* `Matches` {word: keyword for its group}
