#!/usr/bin/env python3
"""
Simple synthetic data test for Text4GCN.

This script creates minimal synthetic data and tests basic functionality.
"""

import os
import random
from pathlib import Path


def create_simple_synthetic_data(dataset_name="SyntheticR8", num_docs=100):
    """Create simple synthetic data for testing."""
    
    # Create output directory
    output_dir = Path("synthetic_data")
    output_dir.mkdir(exist_ok=True)
    
    # Define categories and sample texts
    categories = ["technology", "science", "business", "sports", "health"]
    
    # Generate documents
    documents = []
    metadata = []
    
    for i in range(num_docs):
        category = random.choice(categories)
        
        # Create simple document
        if category == "technology":
            doc = f"computer software programming algorithm data network system digital internet cloud technology innovation development"
        elif category == "science":
            doc = f"research experiment hypothesis analysis discovery theory methodology observation conclusion study scientific investigation"
        elif category == "business":
            doc = f"company market revenue profit investment strategy growth management finance economy sales customer business development"
        elif category == "sports":
            doc = f"team player game match championship training coach victory competition athlete performance league sports activity"
        else:  # health
            doc = f"medical treatment patient health medicine therapy diagnosis symptoms recovery wellness care hospital medical practice"
        
        # Add some random words
        common_words = ["the", "and", "of", "to", "in", "is", "it", "that", "for", "with", "as", "was", "be", "on", "have"]
        for _ in range(random.randint(5, 15)):
            doc += " " + random.choice(common_words)
        
        documents.append(doc)
        
        # Create metadata
        split = "train" if i < int(num_docs * 0.8) else "test"
        metadata.append(f"{i}\t{split}\t{category}")
    
    # Write files
    txt_file = output_dir / f"{dataset_name}.txt"
    meta_file = output_dir / f"{dataset_name}.meta"
    
    with open(txt_file, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(doc + "\n")
    
    with open(meta_file, "w", encoding="utf-8") as f:
        for meta in metadata:
            f.write(meta + "\n")
    
    print(f"✅ Generated synthetic data:")
    print(f"   - {txt_file} ({txt_file.stat().st_size} bytes)")
    print(f"   - {meta_file} ({meta_file.stat().st_size} bytes)")
    print(f"   - Documents: {num_docs}")
    print(f"   - Categories: {categories}")
    
    return str(output_dir)


def test_basic_functionality():
    """Test basic Text4GCN functionality."""
    
    print("\n🧪 Testing Basic Functionality")
    print("=" * 40)
    
    try:
        # Test imports
        print("1. Testing imports...")
        import sys
        from pathlib import Path
        
        # Add parent directory to path
        parent_dir = Path(__file__).parent.parent
        sys.path.insert(0, str(parent_dir))
        
        import text4gcn
        print(f"   ✅ Text4GCN imported successfully (version: {text4gcn.__version__})")
        
        # Test device management
        print("2. Testing device management...")
        from text4gcn.utils.device import check_pytorch_installation
        pytorch_info = check_pytorch_installation()
        print(f"   ✅ PyTorch available: {pytorch_info['torch_available']}")
        if pytorch_info['torch_available']:
            print(f"   ✅ PyTorch version: {pytorch_info['torch_version']}")
            print(f"   ✅ CUDA available: {pytorch_info['cuda_available']}")
        
        # Test TextPipeline creation
        print("3. Testing TextPipeline creation...")
        from text4gcn.preprocess import TextPipeline
        
        data_path = create_simple_synthetic_data("SyntheticR8", 50)
        
        pipe = TextPipeline(
            dataset_name="SyntheticR8",
            rare_count=5,
            dataset_path=data_path,
            language="english",
            seed=1234,
        )
        print("   ✅ TextPipeline created successfully")
        
        # Test pipeline execution
        print("4. Testing TextPipeline execution...")
        pipe.execute(validation_ratio=0.10)
        print("   ✅ TextPipeline executed successfully")
        
        # Check output files
        print("5. Checking output files...")
        data_path = Path(data_path)
        output_dirs = [
            "SyntheticR8.cleaned",
            "SyntheticR8.shuffled", 
            "SyntheticR8.node_features"
        ]
        
        for dir_name in output_dirs:
            dir_path = data_path / dir_name
            if dir_path.exists():
                files = list(dir_path.glob("*"))
                print(f"   ✅ {dir_name}: {len(files)} files")
            else:
                print(f"   ❌ {dir_name}: missing")
        
        print("\n🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    
    print("🚀 Text4GCN Simple Synthetic Data Test")
    print("=" * 50)
    
    # Change to examples directory
    os.chdir(Path(__file__).parent)
    
    # Run test
    success = test_basic_functionality()
    
    if success:
        print("\n📚 Next steps:")
        print("   - Try with larger datasets")
        print("   - Test with different builders")
        print("   - Experiment with model training")
        print("   - Check the generated files in synthetic_data/")
    else:
        print("\n🔧 Troubleshooting:")
        print("   - Check if Text4GCN is installed")
        print("   - Verify Python environment")
        print("   - Check file permissions")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
