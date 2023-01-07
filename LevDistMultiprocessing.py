from itertools import combinations
import multiprocessing as mp
import psutil
from tqdm import tqdm

from Constants import *
from dictionary_manipulations import create_dictionary, update_dictionary
from logger import timer_func
from levenshtein_dist import lev_recursive
from initial_processor import *


@timer_func
def calculate_lev_dist_multiprocess(bow):
    # Solution from StackOverflow:
    # https://stackoverflow.com/questions/13446445/python-multiprocessing-safely-writing-to-a-file
    manager = mp.Manager()
    q = manager.Queue()
    pool = mp.Pool(mp.cpu_count() + 2)

    # put listener to work first
    pool.apply_async(listener, (q,))

    # split list of combinations into sub-lists
    combs = combinations(bow, 2)
    # n_packages = ceil(factorial(len(bow)) / (2 * factorial(len(bow) - 2)) / (MAX_NUMBER_OF_JOBS * WORD_PAIRS_PER_PARTITION))
    # partitions = np.array_split(combs, int(ceil(len(combs) / WORD_PAIRS_PER_PARTITION)))

    # fire off workers
    jobs = []
    ram_usage_ok = True
    while True:
        try:
            jobs = []
            i = 0
            while ram_usage_ok:
                partition = []
                for _ in range(WORD_PAIRS_PER_PARTITION):
                    partition.append(combs.__next__())

                job = pool.apply_async(worker, (partition, q))
                jobs.append(job)
                i += 1

                if i >= 100:
                    ram_usage_ok = psutil.virtual_memory()[2] < 75
                    i = 0

            # collect results from the workers through the pool result queue
            for job in tqdm(jobs):
                job.get()

        except StopIteration:
            break

    # collect results once again (after Stop Iteration)
    for job in tqdm(jobs):
        print("here I am :)")
        job.get()

    # now we are done, kill the listener
    q.put('kill')
    pool.close()
    pool.join()


def worker(partition, q):
    """"""
    res = ""
    for w1, w2 in partition:
        lev_dist = lev_recursive(w1, w2)
        m_ = max(len(w1), len(w2))
        lev_closeness = round((m_ - lev_dist) / m_, 4)
        res += f"{w1}{CSV_SEP}{w2}{CSV_SEP}{lev_dist}{CSV_SEP}{lev_closeness}\n"

    q.put(res)
    return res


def listener(q):
    """listens for messages on the q, writes to LEVENSHTEIN_PAIRS."""
    with open(LEVENSHTEIN_PAIRS, 'w') as f:
        f.write(f"word1{CSV_SEP}word2{CSV_SEP}levdist{CSV_SEP}levcloseness\n")
        f.flush()
        while 1:
            m = q.get()
            if m == 'kill':
                break
            f.write(m)
            f.flush()


if __name__ == "__main__":
    filepath1 = os.path.join(TEMP_DIR, 'something.txt')
    filepath2 = os.path.join(TEMP_DIR, 'something.txt')
    filepath3 = os.path.join(TEMP_DIR, 'something.txt')

    # Get bag of words
    bag_of_words = get_bag_of_words(filepath3)

    # Calculate levenshtein distances for word pairs and write them into temp/
    # calculate_lev_dist_multiprocess(bag_of_words)

    # create_dictionary() with default parameter 'name=DICTIONARY' should be called only once.
    # In case we want to build a new dictionary, set 'name=USER_DICTIONARY', so the initial DICTIONARY stays untouched.
    create_dictionary()

    # update_dictionary() updates existing DICTIONARY with bag_of_words.
    # It is expected that update_dictionary() and create_dictionary() will be used interchangeably.
    update_dictionary(bag_of_words)
