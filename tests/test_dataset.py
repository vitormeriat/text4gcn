import pytest

from text4gcn.modules import file_ops as flop
from text4gcn.modules import logger
from text4gcn.datasets import data


def test_gets_data_list_len():
    assert len(data.list()) == 3


def test_validate_files_with_success():
    lgg = logger.PrintLog()
    flops = flop.FileOps(logger=lgg)
    num_lines_f1, num_lines_f2 = flops.validate_input_files_lines(
        path_txt="tests/data/file_valid.txt", path_meta="tests/data/file_valid.meta")
    assert num_lines_f1 == num_lines_f2

    meta_validation = flops.validate_input_meta_columns(
        file_path="tests/data/file_valid.meta")

    assert meta_validation == True

    message, status = flops.validate_files(path_txt="tests/data/file_valid.txt", 
                                path_meta="tests/data/file_valid.meta")
    
    assert status == True
    assert message == '[FILES VALIDATION] - [meta] Validated file - [corpus] Validated file'


def test_validate_files_fail_meta_missing_column():
    lgg = logger.PrintLog()
    flops = flop.FileOps(logger=lgg)
    num_lines_f1, num_lines_f2 = flops.validate_input_files_lines(
        path_txt="tests/data/file_valid.txt", path_meta="tests/data/file_meta_2_cols.meta")
    assert num_lines_f1 == num_lines_f2

    meta_validation = flops.validate_input_meta_columns(
        file_path="tests/data/file_meta_2_cols.meta")

    assert meta_validation == False

    message, status = flops.validate_files(path_txt="tests/data/file_valid.txt", 
                                path_meta="tests/data/file_meta_2_cols.meta")
    
    assert status == False
    assert message == '[FILES VALIDATION] - [meta] The meta file does not correspond to standard - [corpus] Validated file'


def test_validate_files_fail_different_rows_count():
    lgg = logger.PrintLog()
    flops = flop.FileOps(logger=lgg)
    num_lines_f1, num_lines_f2 = flops.validate_input_files_lines(
        path_txt="tests/data/file_txt_10_lines.txt", path_meta="tests/data/file_valid.meta")
    
    assert num_lines_f1 == 10
    assert num_lines_f2 == 20
    assert num_lines_f1 < num_lines_f2

    meta_validation = flops.validate_input_meta_columns(
        file_path="tests/data/file_valid.meta")

    assert meta_validation == True

    message, status = flops.validate_files(path_txt="tests/data/file_txt_10_lines.txt", 
                                path_meta="tests/data/file_valid.meta")
    
    assert status == False
    assert message == '[FILES VALIDATION] - [meta] Validated file - [corpus] Files do not have the same count of lines'
