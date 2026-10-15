"""
Data Shuffler - Single Responsibility: Shuffle and split data.

This class is responsible for:
- Shuffling documents
- Creating train/test splits
- Managing data indices
"""

import random
from typing import Tuple
from text4gcn.modules import file_ops as flop
from text4gcn.modules import logger as logger
from text4gcn.modules.logger import Process


class DataShuffler:
    """Responsible for shuffling and splitting data."""
    
    def __init__(self, dataset_name: str, dataset_path: str):
        """
        Initialize DataShuffler.
        
        Args:
            dataset_name: Name of the dataset
            dataset_path: Path to dataset files
        """
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        
        # Initialize dependencies
        self.logger = logger.PrintLog()
        self.flop = flop.FileOps(logger=self.logger)
    
    @Process.log("SHUFFLED DATA: Corpus documents shuffled.")
    def shuffle_data(self, cleaned_data_path: str) -> str:
        """
        Shuffle the cleaned data and create train/test splits.
        
        Args:
            cleaned_data_path: Path to cleaned data
            
        Returns:
            Path to shuffled data directory
        """
        ds_corpus_meta = f"{self.dataset_path}/{self.dataset_name}.meta"
        ds_corpus_shuffled = f"{self.dataset_path}/{self.dataset_name}.shuffled/{self.dataset_name}.txt"
        ds_corpus_shuffled_train_idx = f"{self.dataset_path}/{self.dataset_name}.shuffled/{self.dataset_name}.train"
        ds_corpus_shuffled_test_idx = f"{self.dataset_path}/{self.dataset_name}.shuffled/{self.dataset_name}.test"
        ds_corpus_shuffled_meta = f"{self.dataset_path}/{self.dataset_name}.shuffled/{self.dataset_name}.meta"
        
        # Create output directory
        self.flop.create_dir(
            dir_path=f"{self.dataset_path}/{self.dataset_name}.shuffled/",
            overwrite=False
        )
        
        # Load corpus metadata
        all_doc_meta_list, train_doc_meta_list, test_doc_meta_list = self.flop.load_corpus_meta(
            corpus_meta_path=ds_corpus_meta
        )
        
        # Load cleaned documents
        cleaned_doc_lines = [line.strip() for line in open(cleaned_data_path, 'r', encoding="utf-8")]
        
        # Shuffle train indices
        train_doc_meta_ids = [all_doc_meta_list.index(train_doc_meta) 
                             for train_doc_meta in train_doc_meta_list]
        random.shuffle(train_doc_meta_ids)
        
        # Write train indices
        self.flop.write_iterable_to_file(
            an_iterable=train_doc_meta_ids,
            file_path=ds_corpus_shuffled_train_idx,
            file_mode='w'
        )
        
        # Shuffle test indices
        test_doc_meta_ids = [all_doc_meta_list.index(test_doc_meta) 
                            for test_doc_meta in test_doc_meta_list]
        random.shuffle(test_doc_meta_ids)
        
        # Write test indices
        self.flop.write_iterable_to_file(
            an_iterable=test_doc_meta_ids,
            file_path=ds_corpus_shuffled_test_idx,
            file_mode='w'
        )
        
        # Combine and shuffle all indices
        all_doc_meta_ids = train_doc_meta_ids + test_doc_meta_ids
        
        # Write shuffled metadata
        shuffled_doc_meta_list = [all_doc_meta_list[all_doc_meta_id] 
                                 for all_doc_meta_id in all_doc_meta_ids]
        
        self.flop.write_iterable_to_file(
            an_iterable=shuffled_doc_meta_list,
            file_path=ds_corpus_shuffled_meta,
            file_mode='w'
        )
        
        # Write shuffled documents
        shuffled_doc_lines = [cleaned_doc_lines[all_doc_meta_id] 
                             for all_doc_meta_id in all_doc_meta_ids]
        
        self.flop.write_iterable_to_file(
            an_iterable=shuffled_doc_lines,
            file_path=ds_corpus_shuffled,
            file_mode='w'
        )
        
        return f"{self.dataset_path}/{self.dataset_name}.shuffled"
    
    def get_shuffled_data_paths(self) -> Tuple[str, str, str, str]:
        """
        Get paths to shuffled data files.
        
        Returns:
            Tuple of (corpus_path, train_idx_path, test_idx_path, meta_path)
        """
        base_path = f"{self.dataset_path}/{self.dataset_name}.shuffled"
        return (
            f"{base_path}/{self.dataset_name}.txt",
            f"{base_path}/{self.dataset_name}.train",
            f"{base_path}/{self.dataset_name}.test",
            f"{base_path}/{self.dataset_name}.meta"
        )
