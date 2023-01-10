import csv
import multiprocessing as mp

from tqdm import tqdm

from Constants import *
from CustomIterator import custom_iter
from logger import timer_func
from Levenshtein.levenshtein_metrics import levenshtein_closeness_iterative
from DocumentProcessing.initial_processor import *


@timer_func
def create_lev_csl_csv_multiprocess(bow, known_words):
    clear_dir(LEV_CL_SPLIT_FILES)
    word_pairs_iterator = custom_iter(bow, known_words)

    stop_iter = False
    while not stop_iter:
        word_pairs_iterator, stop_iter = handle_processes(word_pairs_iterator)


def handle_processes(iterator):
    stop_iter = False

    # Solution from StackOverflow:
    # https://stackoverflow.com/questions/13446445/python-multiprocessing-safely-writing-to-a-file
    manager = mp.Manager()
    q = manager.Queue()
    pool = mp.Pool(mp.cpu_count() + 2)

    # put listener to work first
    pool.apply_async(listener, (q,))

    # fire off workers
    jobs = []
    for _ in range(MAX_N_PAIRS_PER_POOL):
        try:
            comb = iterator.__next__()
            job = pool.apply_async(worker, (comb, q))
            jobs.append(job)
        except StopIteration:
            stop_iter = True
            break

    # collect results from the workers through the pool result queue
    for job in tqdm(jobs):
        job.get()

    # now we are done, kill the listener
    q.put('kill')
    pool.close()
    pool.join()

    return iterator, stop_iter


def worker(comb, q):
    """"""
    lev_closeness = levenshtein_closeness_iterative(comb[0], comb[1])
    res = f"{comb[0]}{CSV_SEP}{comb[1]}{CSV_SEP}{lev_closeness}"

    q.put(res)
    return res


def listener(q):
    """listens for messages on the q, writes to LEVENSHTEIN_PAIRS."""
    files = os.listdir(LEV_CL_SPLIT_FILES)
    if files:
        last_idx = max(set(map(lambda f: int(re.search(r'\d+(?=\.csv)', f)[0]), files)))
    else:
        last_idx = -1
    new_csv_path = os.path.join(LEV_CL_SPLIT_FILES, LEV_PAIRS_BASENAME + '_' + str(last_idx + 1) + '.csv')
    del files, last_idx

    with open(new_csv_path, 'w', newline='\n', encoding=ENCODING) as f:
        writer = csv.writer(f, delimiter=CSV_SEP)
        # Write header
        writer.writerow(HEADER_CSV)
        f.flush()

        while 1:
            m = q.get()
            if m == 'kill':
                break

            w1, w2, lev_cl = m.split(CSV_SEP)
            writer.writerow([w1, w2, lev_cl])
            f.flush()


if __name__ == "__main__":
    filepath1 = os.path.join(TEMP_DIR, 'something.txt')
    filepath2 = os.path.join(TEMP_DIR, 'something.txt')
    filepath3 = os.path.join(TEMP_DIR, 'something.txt')

    # Get bag of words
    bag_of_words = get_bag_of_words(filepath3)
