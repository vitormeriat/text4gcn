"""Tests for the datasets module."""

import pytest

from text4gcn.modules import file_ops as flop
from text4gcn.modules import logger
from text4gcn.datasets import data


def test_gets_data_list_len():
    """Test that the data list contains the expected number of datasets."""
    # Verify that exactly 3 datasets are available
    assert len(data.list()) == 3


def test_validate_files_with_success():
    """Test file validation with valid input files."""
    # Initialize logger and file operations
    lgg = logger.PrintLog()
    flops = flop.FileOps(logger=lgg)
    
    # Validate that text and meta files have matching line counts
    num_lines_f1, num_lines_f2 = flops.validate_input_files_lines(
        path_txt="tests/data/file_valid.txt", path_meta="tests/data/file_valid.meta")
    assert num_lines_f1 == num_lines_f2

    # Validate that meta file has correct column structure
    meta_validation = flops.validate_input_meta_columns(
        file_path="tests/data/file_valid.meta")
    assert meta_validation == True

    # Perform complete file validation
    message, status = flops.validate_files(path_txt="tests/data/file_valid.txt", 
                                path_meta="tests/data/file_valid.meta")
    
    # Verify validation succeeded and message indicates success
    assert status == True
    assert message == '[FILES VALIDATION] - [meta] Validated file - [corpus] Validated file'


def test_validate_files_fail_meta_missing_column():
    """Test file validation failure when meta file has incorrect column structure."""
    # Initialize logger and file operations
    lgg = logger.PrintLog()
    flops = flop.FileOps(logger=lgg)
    
    # Validate line counts (should match even if columns are wrong)
    num_lines_f1, num_lines_f2 = flops.validate_input_files_lines(
        path_txt="tests/data/file_valid.txt", path_meta="tests/data/file_meta_2_cols.meta")
    assert num_lines_f1 == num_lines_f2

    # Validate meta columns - should fail because file has only 2 columns instead of expected format
    meta_validation = flops.validate_input_meta_columns(
        file_path="tests/data/file_meta_2_cols.meta")
    assert meta_validation == False

    # Perform complete file validation - should fail due to meta file structure
    message, status = flops.validate_files(path_txt="tests/data/file_valid.txt", 
                                path_meta="tests/data/file_meta_2_cols.meta")
    
    # Verify validation failed with appropriate error message
    assert status == False
    assert message == '[FILES VALIDATION] - [meta] The meta file does not correspond to standard - [corpus] Validated file'


def test_validate_files_fail_different_rows_count():
    """Test file validation failure when text and meta files have different row counts."""
    # Initialize logger and file operations
    lgg = logger.PrintLog()
    flops = flop.FileOps(logger=lgg)
    
    # Validate line counts - should detect mismatch (10 lines in txt vs 20 in meta)
    num_lines_f1, num_lines_f2 = flops.validate_input_files_lines(
        path_txt="tests/data/file_txt_10_lines.txt", path_meta="tests/data/file_valid.meta")
    
    # Verify the mismatch is detected
    assert num_lines_f1 == 10
    assert num_lines_f2 == 20
    assert num_lines_f1 < num_lines_f2

    # Meta file structure is valid, but line count doesn't match
    meta_validation = flops.validate_input_meta_columns(
        file_path="tests/data/file_valid.meta")
    assert meta_validation == True

    # Perform complete file validation - should fail due to line count mismatch
    message, status = flops.validate_files(path_txt="tests/data/file_txt_10_lines.txt", 
                                path_meta="tests/data/file_valid.meta")
    
    # Verify validation failed with appropriate error message
    assert status == False
    assert message == '[FILES VALIDATION] - [meta] Validated file - [corpus] Files do not have the same count of lines'
