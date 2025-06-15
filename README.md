Proof of concept for router that chooses the best model to answer a particular prompt.

## Prerequisites

### System Requirements
- Python 3.10 or 3.11
- Git
- Hugging Face CLI (for data download)
- Google Cloud CLI (for data download)
- CUDA-compatible GPU with CUDA toolkit installed (required for Jina embeddings model)
- keys for cloud providers as env vars - checkout .env.template (required if you want to test router on your own data)


### Setup
1. Clone the repository
2. Create and activate a virtual environment
3. Install dependencies using `pip install -e .`
4. Login to Hugging Face using `huggingface-cli login`
5. Configure Google Cloud CLI if needed for data download

## Results

Accuracy on mmlu compared to using single best model (Claude Sonnet 3.5 from the ones analyzed) improves from 87% to 92%.

To reproduce:
1. Download raw data by running `python scripts/download_data.py`
2. Run `notebooks/eda_standford_mmlu.ipynb` jupyter notebook to select subset of data we'll be working with
3. Run `python scripts/vectorize_data.py --input data/intermediate/stanford_mmlu_results.parquet` to pre-calculate embeddings used by routing models
4. Run `python scripts/train_models.py` to train routing models
5. Run `notebooks/evaluate_models.ipynb` to check model evaluation