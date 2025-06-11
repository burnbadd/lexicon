# Lexicon Generator

A Python-based tool for generating lexicons of cooperative and uncooperative words using semantic similarity and transformer models.

## Overview

This project uses sentence transformers and BERT to generate two lexicons:
- Cooperative words (e.g., collaborate, agree, support)
- Uncooperative words (e.g., oppose, criticize, refuse)

The tool uses semantic similarity to find words that are semantically related to seed words in each category.

## Requirements

- Python 3.x
- Required packages:
  - sentence-transformers
  - transformers
  - scikit-learn
  - numpy

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd lexicon
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install required packages:
```bash
pip install sentence-transformers transformers scikit-learn
```

## Usage

The project consists of two main files:
- `dictionary_generation.py`: The main script for generating the lexicons
- `dic.ipynb`: A Jupyter notebook version of the same functionality

To generate the lexicons, run:
```bash
python dictionary_generation.py
```

This will create two CSV files:
- `coop_final.csv`: Contains the list of cooperative words
- `uncoop_final.csv`: Contains the list of uncooperative words

## How it Works

1. The script uses the `all-MiniLM-L6-v2` sentence transformer model and BERT tokenizer
2. It starts with seed words for both cooperative and uncooperative categories
3. Uses semantic similarity to find related words from the BERT vocabulary
4. Filters and processes the results to create two distinct lexicons
5. Outputs the results to CSV files

## Output

The generated lexicons are saved in two CSV files:
- `coop_final.csv`: Contains words semantically related to cooperation
- `uncoop_final.csv`: Contains words semantically related to uncooperation

Each file contains a single column with the respective words. 