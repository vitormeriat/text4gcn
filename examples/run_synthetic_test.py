#!/usr/bin/env python3
"""
Quick test script for Text4GCN with synthetic data.

This script generates synthetic data and tests the TextPipeline functionality.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    """Main function to run synthetic data test."""
    
    print("🚀 Text4GCN Synthetic Data Test")
    print("=" * 50)
    
    # Change to examples directory
    os.chdir(Path(__file__).parent)
    
    try:
        # Import and run the test
        from test_textpipeline_synthetic import test_textpipeline_with_synthetic_data, demonstrate_usage
        
        # Run the test
        success = test_textpipeline_with_synthetic_data()
        
        if success:
            print("\n🎉 Test completed successfully!")
            demonstrate_usage()
        else:
            print("\n❌ Test failed!")
            return 1
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're in the correct directory and Text4GCN is installed.")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
