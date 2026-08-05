"""
Synthetic Data Generator for Text4GCN Testing

This script generates synthetic text data in the same format as the benchmark datasets
(R8, R52, etc.) to test the TextPipeline functionality.
"""

import os
import random
from pathlib import Path
from typing import List, Tuple, Dict
import argparse


class SyntheticDataGenerator:
    """Generate synthetic text data for testing Text4GCN."""
    
    def __init__(self, output_dir: str = "synthetic_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Define synthetic categories and their characteristics
        self.categories = {
            "technology": {
                "keywords": ["computer", "software", "hardware", "programming", "algorithm", 
                           "data", "network", "system", "digital", "internet", "cloud", "ai"],
                "templates": [
                    "The {keyword} system provides advanced {feature} capabilities",
                    "Our {keyword} solution offers {feature} for modern businesses",
                    "The latest {keyword} technology enables {feature} integration",
                    "Advanced {keyword} algorithms improve {feature} performance"
                ]
            },
            "science": {
                "keywords": ["research", "experiment", "hypothesis", "analysis", "discovery",
                           "theory", "methodology", "observation", "conclusion", "study"],
                "templates": [
                    "The {keyword} study reveals important {feature} findings",
                    "Our {keyword} research demonstrates {feature} effectiveness",
                    "The {keyword} experiment shows {feature} results",
                    "Scientific {keyword} analysis confirms {feature} hypothesis"
                ]
            },
            "business": {
                "keywords": ["company", "market", "revenue", "profit", "investment", "strategy",
                           "growth", "management", "finance", "economy", "sales", "customer"],
                "templates": [
                    "The {keyword} strategy increases {feature} performance",
                    "Our {keyword} approach improves {feature} outcomes",
                    "The {keyword} plan enhances {feature} efficiency",
                    "Business {keyword} solutions drive {feature} success"
                ]
            },
            "sports": {
                "keywords": ["team", "player", "game", "match", "championship", "training",
                           "coach", "victory", "competition", "athlete", "performance", "league"],
                "templates": [
                    "The {keyword} team achieves {feature} success",
                    "Our {keyword} strategy leads to {feature} victory",
                    "The {keyword} performance shows {feature} improvement",
                    "Sports {keyword} training enhances {feature} results"
                ]
            },
            "health": {
                "keywords": ["medical", "treatment", "patient", "health", "medicine", "therapy",
                           "diagnosis", "symptoms", "recovery", "wellness", "care", "hospital"],
                "templates": [
                    "The {keyword} treatment improves {feature} outcomes",
                    "Our {keyword} approach enhances {feature} recovery",
                    "The {keyword} method provides {feature} benefits",
                    "Medical {keyword} care ensures {feature} wellness"
                ]
            }
        }
        
        # Additional features to make text more realistic
        self.features = [
            "innovative", "advanced", "comprehensive", "effective", "efficient",
            "reliable", "scalable", "flexible", "robust", "secure", "fast",
            "accurate", "precise", "optimized", "integrated", "automated"
        ]
    
    def generate_document(self, category: str, length: int = 50) -> str:
        """Generate a single synthetic document."""
        if category not in self.categories:
            raise ValueError(f"Unknown category: {category}")
        
        cat_info = self.categories[category]
        words = []
        
        # Add some random words to make it more realistic
        for _ in range(length):
            if random.random() < 0.3:  # 30% chance to use category keywords
                word = random.choice(cat_info["keywords"])
            elif random.random() < 0.2:  # 20% chance to use features
                word = random.choice(self.features)
            else:  # 50% chance to use common words
                word = random.choice([
                    "the", "and", "of", "to", "in", "is", "it", "that", "for", "with",
                    "as", "was", "be", "on", "have", "he", "at", "by", "this", "from",
                    "they", "she", "or", "an", "had", "but", "not", "what", "all", "were",
                    "when", "we", "there", "can", "an", "your", "which", "their", "said",
                    "each", "which", "do", "how", "if", "will", "up", "other", "about",
                    "out", "many", "then", "them", "these", "so", "some", "her", "would",
                    "make", "like", "into", "him", "time", "has", "two", "more", "go",
                    "no", "way", "could", "my", "than", "first", "been", "call", "who",
                    "its", "now", "find", "long", "down", "day", "did", "get", "come",
                    "made", "may", "part"
                ])
            words.append(word)
        
        return " ".join(words)
    
    def generate_dataset(self, 
                        dataset_name: str, 
                        num_documents: int = 1000,
                        train_ratio: float = 0.8) -> None:
        """Generate a complete synthetic dataset."""
        
        # Calculate number of train/test documents
        num_train = int(num_documents * train_ratio)
        num_test = num_documents - num_train
        
        # Generate documents
        documents = []
        metadata = []
        
        # Generate train documents
        for i in range(num_train):
            category = random.choice(list(self.categories.keys()))
            doc = self.generate_document(category, length=random.randint(30, 100))
            documents.append(doc)
            metadata.append(f"{i}\ttrain\t{category}")
        
        # Generate test documents
        for i in range(num_train, num_documents):
            category = random.choice(list(self.categories.keys()))
            doc = self.generate_document(category, length=random.randint(30, 100))
            documents.append(doc)
            metadata.append(f"{i}\ttest\t{category}")
        
        # Write files into output_dir (used as TextPipeline dataset_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        txt_path = self.output_dir / f"{dataset_name}.txt"
        meta_path = self.output_dir / f"{dataset_name}.meta"

        with open(txt_path, "w", encoding="utf-8") as f:
            for doc in documents:
                f.write(doc + "\n")

        with open(meta_path, "w", encoding="utf-8") as f:
            for meta in metadata:
                f.write(meta + "\n")

        print(f"Generated synthetic dataset '{dataset_name}':")
        print(f"  - Documents: {num_documents}")
        print(f"  - Train: {num_train}")
        print(f"  - Test: {num_test}")
        print(f"  - Categories: {list(self.categories.keys())}")
        print(f"  - Files created: {txt_path}, {meta_path}")
    
    def generate_multiple_datasets(self) -> None:
        """Generate multiple synthetic datasets for testing."""
        
        datasets = [
            ("SyntheticR8", 1000, 0.8),      # Similar to R8
            ("SyntheticR52", 2000, 0.8),    # Similar to R52
            ("SyntheticSmall", 100, 0.7),   # Small dataset for quick testing
            ("SyntheticLarge", 5000, 0.8),  # Large dataset for performance testing
        ]
        
        for name, num_docs, train_ratio in datasets:
            print(f"\nGenerating {name}...")
            self.generate_dataset(name, num_docs, train_ratio)
    
    def create_test_structure(self) -> None:
        """Create the directory structure expected by TextPipeline."""
        # Create the main data directory
        data_dir = Path("synthetic_data")
        data_dir.mkdir(exist_ok=True)
        
        print(f"Created directory structure: {data_dir.absolute()}")
        print("You can now use this directory with TextPipeline:")
        print("""
from text4gcn.preprocess import TextPipeline

pipe = TextPipeline(
    dataset_name="SyntheticR8",
    dataset_path="synthetic_data",
    rare_count=5,
    language="english"
)
pipe.execute()
        """)


def main():
    """Main function to generate synthetic data."""
    parser = argparse.ArgumentParser(description="Generate synthetic data for Text4GCN testing")
    parser.add_argument("--dataset-name", default="SyntheticR8", help="Name of the dataset")
    parser.add_argument("--num-documents", type=int, default=1000, help="Number of documents")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Ratio of train documents")
    parser.add_argument("--output-dir", default="synthetic_data", help="Output directory")
    parser.add_argument("--multiple", action="store_true", help="Generate multiple datasets")
    
    args = parser.parse_args()
    
    # Change to the examples directory
    os.chdir(Path(__file__).parent)
    
    generator = SyntheticDataGenerator(args.output_dir)
    
    if args.multiple:
        generator.generate_multiple_datasets()
    else:
        generator.generate_dataset(
            args.dataset_name, 
            args.num_documents, 
            args.train_ratio
        )
    
    generator.create_test_structure()


if __name__ == "__main__":
    main()
