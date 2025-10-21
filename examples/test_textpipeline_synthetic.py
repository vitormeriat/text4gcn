"""
Test TextPipeline with Synthetic Data

This script demonstrates how to use Text4GCN's TextPipeline with synthetic data
generated in the same format as benchmark datasets.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import text4gcn
sys.path.insert(0, str(Path(__file__).parent.parent))

from text4gcn.preprocess import TextPipeline
from synthetic_data_generator import SyntheticDataGenerator


def test_textpipeline_with_synthetic_data():
    """Test TextPipeline with synthetic data."""
    
    print("🧪 Testing TextPipeline with Synthetic Data")
    print("=" * 60)
    
    # Generate synthetic data
    print("1. Generating synthetic data...")
    generator = SyntheticDataGenerator("synthetic_data")
    generator.generate_dataset("SyntheticR8", num_documents=500, train_ratio=0.8)
    
    # Check if files were created
    data_path = Path("synthetic_data")
    txt_file = data_path / "SyntheticR8.txt"
    meta_file = data_path / "SyntheticR8.meta"
    
    if not txt_file.exists() or not meta_file.exists():
        print("❌ Error: Synthetic data files not created!")
        return False
    
    print(f"✅ Generated files:")
    print(f"   - {txt_file} ({txt_file.stat().st_size} bytes)")
    print(f"   - {meta_file} ({meta_file.stat().st_size} bytes)")
    
    # Test TextPipeline
    print("\n2. Testing TextPipeline...")
    
    try:
        # Create TextPipeline instance
        pipe = TextPipeline(
            dataset_name="SyntheticR8",
            rare_count=5,
            dataset_path="synthetic_data",
            language="english"
        )
        
        print("✅ TextPipeline created successfully")
        
        # Execute the pipeline
        print("\n3. Executing TextPipeline...")
        pipe.execute()
        
        print("✅ TextPipeline executed successfully!")
        
        # Check output files
        print("\n4. Checking output files...")
        check_output_files("synthetic_data", "SyntheticR8")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during TextPipeline execution: {e}")
        return False


def check_output_files(data_path: str, dataset_name: str):
    """Check if TextPipeline created the expected output files."""
    
    base_path = Path(data_path)
    
    # Expected output directories and files
    expected_outputs = [
        f"{dataset_name}.cleaned/{dataset_name}.txt",
        f"{dataset_name}.shuffled/{dataset_name}.txt",
        f"{dataset_name}.shuffled/{dataset_name}.train",
        f"{dataset_name}.shuffled/{dataset_name}.test",
        f"{dataset_name}.shuffled/{dataset_name}.meta",
        f"{dataset_name}.shuffled/{dataset_name}.vocab",
        f"{dataset_name}.shuffled/{dataset_name}.word_vectors",
        f"{dataset_name}.node_features/ind.{dataset_name}.x",
        f"{dataset_name}.node_features/ind.{dataset_name}.y",
        f"{dataset_name}.node_features/ind.{dataset_name}.tx",
        f"{dataset_name}.node_features/ind.{dataset_name}.ty",
        f"{dataset_name}.node_features/ind.{dataset_name}.allx",
        f"{dataset_name}.node_features/ind.{dataset_name}.ally",
    ]
    
    print("📁 Checking output files:")
    
    for output_path in expected_outputs:
        full_path = base_path / output_path
        if full_path.exists():
            size = full_path.stat().st_size
            print(f"   ✅ {output_path} ({size} bytes)")
        else:
            print(f"   ❌ {output_path} (missing)")
    
    # Check log directory
    log_dir = base_path / "log"
    if log_dir.exists():
        log_files = list(log_dir.glob("*.txt"))
        print(f"   📝 Log files: {len(log_files)} files")
        for log_file in log_files:
            print(f"      - {log_file.name}")


def demonstrate_usage():
    """Demonstrate how to use the synthetic data with Text4GCN."""
    
    print("\n" + "=" * 60)
    print("📚 Usage Example")
    print("=" * 60)
    
    print("""
# 1. Generate synthetic data
python examples/synthetic_data_generator.py --dataset-name SyntheticR8 --num-documents 1000

# 2. Use with TextPipeline
from text4gcn.preprocess import TextPipeline

pipe = TextPipeline(
    dataset_name="SyntheticR8",
    rare_count=5,
    dataset_path="synthetic_data",
    language="english"
)
pipe.execute()

# 3. Use with builders
from text4gcn.builder import Frequency

builder = Frequency()
adjacency = builder.tfidf(
    dataset_name="SyntheticR8",
    dataset_path="synthetic_data"
)

# 4. Use with models
from text4gcn.models import GNN, Layer, Builder

model = GNN(
    dataset="SyntheticR8",
    path="synthetic_data",
    log_dir="logs",
    builder=Builder.Tfidf,
    layer=Layer.GCN,
    epochs=50,
    dropout=0.5
)
model.fit()
    """)


def main():
    """Main function."""
    
    # Change to the examples directory
    os.chdir(Path(__file__).parent)
    
    print("🚀 Text4GCN Synthetic Data Testing")
    print("=" * 60)
    
    # Test TextPipeline
    success = test_textpipeline_with_synthetic_data()
    
    if success:
        print("\n🎉 All tests passed!")
        demonstrate_usage()
    else:
        print("\n❌ Tests failed!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
