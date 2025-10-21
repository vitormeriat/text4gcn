# Text4GCN Examples

This directory contains examples and utilities for testing Text4GCN with synthetic data.

## 📁 Files Overview

### Data Generation
- `synthetic_data_generator.py` - Generate synthetic text data in Text4GCN format
- `test_textpipeline_synthetic.py` - Test TextPipeline with synthetic data
- `run_synthetic_test.py` - Quick test script
- `synthetic_data_demo.ipynb` - Jupyter notebook demonstration

### Device Management
- `device_management.py` - Device management examples and utilities

## 🚀 Quick Start

### 1. Generate Synthetic Data

```bash
# Generate a single dataset
python synthetic_data_generator.py --dataset-name SyntheticR8 --num-documents 1000

# Generate multiple datasets
python synthetic_data_generator.py --multiple

# Generate large dataset for performance testing
python synthetic_data_generator.py --dataset-name SyntheticLarge --num-documents 5000
```

### 2. Test TextPipeline

```bash
# Run the complete test
python run_synthetic_test.py

# Or run the detailed test
python test_textpipeline_synthetic.py
```

### 3. Jupyter Notebook

```bash
# Start Jupyter and open the notebook
jupyter notebook synthetic_data_demo.ipynb
```

## 📊 Synthetic Data Format

The synthetic data generator creates files in the same format as benchmark datasets:

### Input Files
- `{dataset_name}.txt` - Text documents (one per line)
- `{dataset_name}.meta` - Metadata (id, split, label)

### Output Files (after TextPipeline)
- `{dataset_name}.cleaned/` - Cleaned text data
- `{dataset_name}.shuffled/` - Shuffled data with train/test splits
- `{dataset_name}.node_features/` - Node features (x, y, tx, ty, allx, ally)
- `log/` - Processing logs

## 🧪 Testing Categories

The synthetic data includes 5 categories:
- **Technology**: computer, software, programming, etc.
- **Science**: research, experiment, analysis, etc.
- **Business**: company, market, revenue, etc.
- **Sports**: team, player, game, etc.
- **Health**: medical, treatment, patient, etc.

## 📝 Example Usage

```python
from text4gcn.preprocess import TextPipeline

# Create pipeline
pipe = TextPipeline(
    dataset_name="SyntheticR8",
    rare_count=5,
    dataset_path="synthetic_data",
    language="english"
)

# Execute preprocessing
pipe.execute()

# Use with builders
from text4gcn.builder import Frequency
builder = Frequency()
adjacency = builder.tfidf(
    dataset_name="SyntheticR8",
    dataset_path="synthetic_data"
)

# Use with models
from text4gcn.models import GNN, Layer, Builder
model = GNN(
    dataset="SyntheticR8",
    path="synthetic_data",
    log_dir="logs",
    builder=Builder.Tfidf,
    layer=Layer.GCN,
    epochs=50
)
model.fit()
```

## 🔧 Command Line Options

### Synthetic Data Generator

```bash
python synthetic_data_generator.py --help
```

Options:
- `--dataset-name`: Name of the dataset (default: SyntheticR8)
- `--num-documents`: Number of documents (default: 1000)
- `--train-ratio`: Ratio of train documents (default: 0.8)
- `--output-dir`: Output directory (default: synthetic_data)
- `--multiple`: Generate multiple datasets

### Examples

```bash
# Small dataset for quick testing
python synthetic_data_generator.py --dataset-name SyntheticSmall --num-documents 100

# Large dataset for performance testing
python synthetic_data_generator.py --dataset-name SyntheticLarge --num-documents 5000

# Custom train/test split
python synthetic_data_generator.py --dataset-name SyntheticCustom --num-documents 2000 --train-ratio 0.7
```

## 📈 Performance Testing

Use synthetic data to test Text4GCN performance:

```bash
# Generate large dataset
python synthetic_data_generator.py --dataset-name PerformanceTest --num-documents 10000

# Test with different configurations
python test_textpipeline_synthetic.py
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Make sure you're in the examples directory
   cd examples
   python synthetic_data_generator.py
   ```

2. **Missing Dependencies**
   ```bash
   # Install Text4GCN with dependencies
   pip install text4gcn[cpu]  # or text4gcn[gpu]
   ```

3. **File Permission Errors**
   ```bash
   # Check directory permissions
   ls -la synthetic_data/
   ```

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with verbose output
python test_textpipeline_synthetic.py
```

## 📚 Additional Resources

- [Text4GCN Documentation](../README.md)
- [Device Management Guide](../PYTORCH_DEVICE_MANAGEMENT.md)
- [Contributing Guide](../CONTRIBUTING.md)

## 🤝 Contributing

To add new examples:

1. Create your example file
2. Add documentation to this README
3. Test with synthetic data
4. Submit a pull request

## 📄 License

Same as Text4GCN project - MIT License.
