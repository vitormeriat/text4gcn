from typing import Any, Iterable, List, Tuple
#from ..modules import logger
from text4gcn.modules import logger
from os.path import exists
from shutil import rmtree
from os import makedirs
import pickle as pkl


class FileOps():
    def __init__(self, logger: logger.PrintLog):
        self.logger = logger

    def create_dir(self, dir_path: str, overwrite: bool) -> None:
        if exists(dir_path):
            if overwrite:
                rmtree(dir_path)
                makedirs(dir_path)
            else:
                #print('[WARN] directory:%r already exists, not overwritten.' % dir_path)
                self.logger.warning(
                    'Directory:%r already exists, not overwritten.' % dir_path)
        else:
            makedirs(dir_path)

    def write_iterable_to_file(self, an_iterable: Iterable[Any], file_path: str, file_mode: str = 'w'):
        with open(file_path, file_mode) as f:
            f.writelines("%s\n" % item for item in an_iterable)

    def check_paths(self, *paths: str):
        """
        Check paths if they exist or not
        """

        for path in paths:
            if not exists(path):
                raise FileNotFoundError(
                    'Path: {path} is not found.'.format(path=path))

    def load_corpus_meta(self, corpus_meta_path: str) -> Tuple[List[str], List[str], List[str]]:
        all_doc_meta_list = [
            line.strip() for line in open(corpus_meta_path, 'r').readlines()]

        train_doc_meta_list = [
            doc_meta for doc_meta in all_doc_meta_list if doc_meta.split('\t')[
                1].endswith('train')]

        test_doc_meta_list = [
            doc_meta for doc_meta in all_doc_meta_list if doc_meta.split('\t')[
                1].endswith('test')]

        return all_doc_meta_list, train_doc_meta_list, test_doc_meta_list

    def write_picke(self, obj, file):
        pkl.dump(obj=obj, file=open(file, mode='wb'))


# def check_data_set(data_set_name: str, all_data_set_names: List[str]) -> None:
#     if data_set_name not in all_data_set_names:
#         raise AttributeError(
#             "Wrong data-set name, given:%r, however expected:%r" %
#             (data_set_name, all_data_set_names))
