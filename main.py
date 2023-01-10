from Constants import DOC_EXAMPLES_DIR, TEMP_DIR
from DocumentProcessing.initial_processor import *
from DocumentDistance import process_documents, calculate_tf_idf

if __name__ == "__main__":
    clear_dir(TEMP_DIR)

    # filepath1 = os.path.join(DOC_EXAMPLES_DIR, 'dykdand1.txt')
    # filepath2 = os.path.join(DOC_EXAMPLES_DIR, 'dykdand2.txt')
    # filepath3 = os.path.join(DOC_EXAMPLES_DIR, 'dykdand3.txt')
    # filepath4 = os.path.join(DOC_EXAMPLES_DIR, 'PanTadeusz_framgment.txt')
    # filepath5 = os.path.join(DOC_EXAMPLES_DIR, 'KrolowaSniegu.txt')
    # filepath6 = os.path.join(DOC_EXAMPLES_DIR, 'Pinokio.txt')
    # filepath7 = os.path.join(DOC_EXAMPLES_DIR, 'dykdand4.txt')
    # filepath8 = os.path.join(DOC_EXAMPLES_DIR, 'dykdand5.txt')
    # filepath9 = os.path.join(DOC_EXAMPLES_DIR, 'dykdand6.txt')
    # filepath10 = os.path.join(DOC_EXAMPLES_DIR, 'dykdand7.txt')
    # filepath11 = os.path.join(DOC_EXAMPLES_DIR, 'solaris.txt')

    normalized_docs = process_documents(folder_path=os.path.join(DOC_EXAMPLES_DIR, 'folder_with_docs'))
    calculate_tf_idf(normalized_docs)

    print('DONE.')

