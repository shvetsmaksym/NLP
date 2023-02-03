import argparse

from Constants import DOC_EXAMPLES_DIR, TEMP_DIR
from DocumentProcessing.initial_processor import *
from DocumentDistance import process_documents, calculate_tf_idf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d1', '--document1', required=False)
    parser.add_argument('-d2', '--document2', required=False)
    parser.add_argument('-fp', '--folderpath', required=False)
    args = parser.parse_args()

    clear_dir(TEMP_DIR)

    if args.document1 and args.document2:
        # Measure similarities between two documents
        normalized_docs = process_documents(doc1_path=os.path.join(DOC_EXAMPLES_DIR, args.document1),
                                            doc2_path=os.path.join(DOC_EXAMPLES_DIR, args.document2))
        calculate_tf_idf(normalized_docs)

    elif args.folderpath:
        # Measure similarities between documents from given folder
        normalized_docs = process_documents(folder_path=os.path.join(DOC_EXAMPLES_DIR, args.folderpath))
        calculate_tf_idf(normalized_docs)

    else:
        raise AttributeError("Wrong arguments. Please give --document1, --document2 or --folderpath.")

    print('DONE.')

