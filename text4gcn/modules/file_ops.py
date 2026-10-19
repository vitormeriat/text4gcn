from typing import Any, Iterable, List, Tuple
from os.path import exists
from shutil import rmtree
from os import makedirs
import pickle as pkl

# Split values in `<name>.meta` that mark a training document; any other value is test.
TRAIN_SPLITS = frozenset({"train", "training", "20news-bydate-train"})


class FileOps():

    def get_lines_info(self, file_path):
        with open(file_path, 'r', encoding="utf-8") as file:
            lines = file.readlines()
            return len(lines)

    def validate_input_meta_columns(self, file_path):
        lines = [line.strip() for line in open(file_path, 'r', encoding="utf-8").readlines()]

        validate_lines = True
        if len(lines) > 0:
            for i, line in enumerate(lines):
                if len(line.split('\t')) != 3 and i < len(lines):
                    return False
            if validate_lines:
                return True
        return False

    def validate_input_files_lines(self, path_txt: str, path_meta: str):
        num_lines_file1 = self.get_lines_info(path_txt)
        num_lines_file2 = self.get_lines_info(path_meta)
        return num_lines_file1, num_lines_file2

    def validate_files(self, path_txt: str, path_meta: str):
        meta_validation = self.validate_input_meta_columns(path_meta)

        n_lines_f1, n_lines_f2 = self.validate_input_files_lines(
            path_txt, path_meta)

        message = "[FILES VALIDATION]"
        status = True

        if meta_validation:
            message += " - [meta] Validated file"
        else:
            status = False
            message += " - [meta] The meta file does not correspond to standard"


        if n_lines_f1 == n_lines_f2:
            message += " - [corpus] Validated file"
        else:
            status = False
            message += " - [corpus] Files do not have the same count of lines"

        return message, status

    def create_dir(self, dir_path: str, overwrite: bool) -> None:
        if exists(dir_path):
            if overwrite:
                rmtree(dir_path)
                makedirs(dir_path)
        else:
            makedirs(dir_path)

    def write_iterable_to_file(self, an_iterable: Iterable[Any], file_path: str, file_mode: str = 'w'):
        encoding = None if 'b' in file_mode else 'utf-8'
        with open(file_path, file_mode, encoding=encoding) as f:
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
            line.strip() for line in open(corpus_meta_path, 'r', encoding="utf-8").readlines()]

        train_doc_meta_list = [
            doc_meta for doc_meta in all_doc_meta_list
            if doc_meta.split('\t')[1] in TRAIN_SPLITS]

        test_doc_meta_list = [
            doc_meta for doc_meta in all_doc_meta_list
            if doc_meta.split('\t')[1] not in TRAIN_SPLITS]

        return all_doc_meta_list, train_doc_meta_list, test_doc_meta_list

    def write_picke(self, obj, file):
        pkl.dump(obj=obj, file=open(file, mode='wb'))


# def check_data_set(data_set_name: str, all_data_set_names: List[str]) -> None:
#     if data_set_name not in all_data_set_names:
#         raise AttributeError(
#             "Wrong data-set name, given:%r, however expected:%r" %
#             (data_set_name, all_data_set_names))
