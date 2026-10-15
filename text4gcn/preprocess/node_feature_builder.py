"""
Node Feature Builder - Single Responsibility: Build node features.

This class is responsible for:
- Building node features (x, y, tx, ty, allx, ally)
- Managing feature extraction
- Saving node features
"""

from math import ceil
from collections import OrderedDict
from text4gcn.modules.node_features import NodeFeatures
from text4gcn.modules import file_ops as flop
from text4gcn.modules import logger as logger
from text4gcn.modules.logger import Process


class NodeFeatureBuilder:
    """Responsible for building node features."""
    
    def __init__(self, dataset_name: str, dataset_path: str):
        """
        Initialize NodeFeatureBuilder.
        
        Args:
            dataset_name: Name of the dataset
            dataset_path: Path to dataset files
        """
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        
        # Initialize dependencies
        self.logger = logger.PrintLog()
        self.flop = flop.FileOps(logger=self.logger)
        self.nfeatures = NodeFeatures(logger=self.logger)
    
    @Process.log("EXTRACTED NODE FEATURES: x, y, tx, ty, allx, ally.")
    def build_node_features(self, shuffled_data_path: str, 
                          validation_ratio: float = 0.10,
                          use_predefined_word_vectors: bool = False) -> str:
        """
        Build node features for the dataset.
        
        Args:
            shuffled_data_path: Path to shuffled data
            validation_ratio: Ratio for validation split
            use_predefined_word_vectors: Whether to use predefined word vectors
            
        Returns:
            Path to node features directory
        """
        ds_corpus = f"{shuffled_data_path}/{self.dataset_name}.txt"
        ds_corpus_meta = f"{shuffled_data_path}/{self.dataset_name}.meta"
        ds_corpus_vocabulary = f"{shuffled_data_path}/{self.dataset_name}.vocab"
        ds_corpus_train_idx = f"{shuffled_data_path}/{self.dataset_name}.train"
        ds_corpus_test_idx = f"{shuffled_data_path}/{self.dataset_name}.test"
        
        # Output directory
        dir_corpus_node_features = f"{self.dataset_path}/{self.dataset_name}.node_features"
        
        # Check paths
        self.flop.check_paths(ds_corpus, ds_corpus_meta, ds_corpus_vocabulary)
        self.flop.check_paths(ds_corpus_train_idx, ds_corpus_test_idx)
        
        # Create output directory
        self.flop.create_dir(dir_path=dir_corpus_node_features, overwrite=False)
        
        # Calculate sizes
        real_train_size = len(open(ds_corpus_train_idx, encoding="utf-8").readlines())
        adjusted_train_size = ceil(real_train_size * (1.0 - validation_ratio))
        test_size = len(open(ds_corpus_test_idx, encoding="utf-8").readlines())
        
        # Load word vectors
        if use_predefined_word_vectors:
            ds_corpus_word_vectors = f"{shuffled_data_path}/{self.dataset_name}.word_vectors"
            word_vectors, word_emb_dim = self.nfeatures.load_word_to_word_vectors(
                path=ds_corpus_word_vectors)
        else:
            word_vectors, word_emb_dim = OrderedDict(), 300
        
        # Load data
        vocabulary = open(ds_corpus_vocabulary, encoding="utf-8").read().splitlines()
        doc_meta_list = open(ds_corpus_meta, 'r', encoding="utf-8").read().splitlines()
        doc_labels = self.nfeatures.extract_doc_labels(ds_corpus_meta_file=ds_corpus_meta)
        docs_of_words = [line.split() for line in open(ds_corpus, encoding="utf-8")]
        
        # Build features
        x = self.nfeatures.compute_x(
            docs_of_words, adjusted_train_size, word_emb_dim, w_vectors=word_vectors
        )
        
        y = self.nfeatures.compute_y(
            doc_meta_list, train_size=adjusted_train_size, doc_labels=doc_labels
        )
        
        tx = self.nfeatures.compute_tx(
            docs_of_words, test_size, real_train_size, word_emb_dim, w_vectors=word_vectors
        )
        
        ty = self.nfeatures.compute_ty(
            doc_meta_list, test_size=test_size, real_train_size=real_train_size, 
            doc_labels=doc_labels
        )
        
        allx = self.nfeatures.compute_allx(
            docs_of_words, real_train_size, vocabulary, word_vectors, emb_dim=word_emb_dim
        )
        
        ally = self.nfeatures.compute_ally(
            doc_meta_list, real_train_size, doc_labels, vocab_size=len(vocabulary)
        )
        
        # Save node features
        node_feature_matrices = {
            "x": x, "y": y, "tx": tx, "ty": ty, "allx": allx, "ally": ally
        }
        
        self.nfeatures.dump_node_features(
            directory=dir_corpus_node_features,
            ds=self.dataset_name,
            node_features_dict=node_feature_matrices
        )
        
        # Log feature shapes
        self.logger.info(f"x.shape={x.shape}, y.shape={y.shape}")
        self.logger.info(f"tx.shape={tx.shape}, ty.shape={ty.shape}")
        self.logger.info(f"allx.shape={allx.shape}, ally.shape={ally.shape}")
        
        return dir_corpus_node_features
    
    def get_node_features_path(self) -> str:
        """
        Get path to node features directory.
        
        Returns:
            Path to node features directory
        """
        return f"{self.dataset_path}/{self.dataset_name}.node_features"


