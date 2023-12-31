from collections import Counter
from typing import List, Set
import re


def config_nltk():
    # temporary_nltk_folder = 'venv/nltk_data/'
    # if not os.path.exists(temporary_nltk_folder):
    #     from nltk import download
    #     download(info_or_id='stopwords', download_dir=temporary_nltk_folder)
    #     download(info_or_id='wordnet', download_dir=temporary_nltk_folder)
    from nltk import download

    #from nltk.corpus import wordnet
    #from nltk.corpus import stopwords
    download(info_or_id='stopwords')
    download(info_or_id='wordnet')
    download(info_or_id='omw-1.4')


def create_title(title):
    txt = f" {title} "
    return txt.center(100, "=")


def clean_str(a_str: str) -> str:
    """
    Tokenizing/string cleaning for all data-sets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", a_str)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", r" \( ", string)
    string = re.sub(r"\)", r" \) ", string)
    string = re.sub(r"\?", r" \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def retrieve_stop_words(language: str = 'english') -> Set[str]:
    #temporary_nltk_folder = 'venv/nltk_data/'
    from nltk.corpus import stopwords

    return set(stopwords.words(language))


def extract_word_counts(docs_of_words: List[List[str]]) -> Counter:
    """Extract word counts"""
    word_counts = Counter()
    for words in docs_of_words:
        word_counts.update(words)
    return word_counts


def remove_stop_words(lines_of_words: List[List[str]], stop_words: Set[str]) -> List[List[str]]:
    """ If a word is in stop-words, then remove it"""
    return [[word for word in line if word not in stop_words] for line in lines_of_words]


def remove_rare_words(lines_of_words: List[List[str]], word_counts: Counter, rare_count: int) -> List[List[str]]:
    """ If a word is rare, then remove it"""
    return [[word for word in line if word_counts[word] >= rare_count] for line in lines_of_words]


def glue_lines(lines_of_words: List[List[str]], glue_str: str, with_strip: bool) -> List[str]:
    if with_strip:
        return [glue_str.join(lines).strip() for lines in lines_of_words]
    else:
        return [glue_str.join(lines) for lines in lines_of_words]
