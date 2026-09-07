# Vector DB From Scratch

A minimalist, educational vector database built in Python from first principles without external vector search libraries.

## Project Structure

```
vector-db-from-scratch/
├── app/
│   ├── __init__.py
│   ├── data/
│   │   └── __init__.py
│   ├── indexes/
│   │   └── __init__.py
│   ├── api/
│   │   └── __init__.py
│   └── utils/
│       └── __init__.py
├── scripts/
│   └── generate_data.py
├── tests/
├── data/
├── requirements.txt
├── README.md
└── .gitignore
```

## Step 1: Dataset Generation

Generate a synthetic clustered vector dataset for testing and benchmarking:

```bash
python scripts/generate_data.py
```

### Dataset Specifications

- **Vectors**: 50,000
- **Dimensions**: 128
- **Clusters**: 100 (exactly 500 vectors per cluster)
- **Data Type**: `float32`
- **Normalized**: Yes (L2 unit length for cosine similarity via dot product)
- **Output Files**:
  - `data/vectors.npy`
  - `data/ids.npy`
  - `data/metadata.json`
