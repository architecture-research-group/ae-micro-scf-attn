# LongSight: Compute-Enabled Memory to Accelerate Large-Context LLMs via Sparse Attention

This repository provides a software implementation of a sparse attention algorithm for **LongSight**, an algorithm-hardware co-design framework for sparse attention in large-context transformer-based language models. By leveraging Compute-Enabled Memory (CEM) alongside token filtering techniques, LongSight efficiently scales the context window up to 1 million tokens on modern hardware.

## Key Components
- **Sparse Attention Module**: Implements `L3_Sparse_Attn` and `L3KSparseDotProd` to model sparse attention with a combination of top-k selection and sign-based filters.
- **Iterative Quantization (ITQ)**: Provides `L3ITQ` and utilities for training per-head rotation matrices to improve sign-based filtering performance.
- **KSparseModel Wrapper**: Extends `transformers.AutoModelForCausalLM` to inject sparse attention layers into Llama 3 models via the `KSparseModel` interface.
- **Example Scripts**:
  - `src/example.py`: Run sparse vs. dense evaluation and compute key-value load reduction metrics.
  - `src/condition_itq.py`: Condition and train ITQ rotation matrices on a text dataset (e.g., WikiText).

## Repository Structure
```
.
├── src/
│   ├── example.py         # Evaluation example for sparse attention
│   ├── condition_itq.py   # ITQ training on text datasets
│   ├── models.py          # KSparseModel definition and utilities
│   └── llama_3_attn.py    # Sparse attention and ITQ implementations
│   ├── cu-popcount/       # Cuda implementation of fast popcount used in sign-based filtering
└── venv/                  # (Optional) Python virtual environment
```

## Installation
1. Clone this repository:
   ```bash
   git clone <repo_url>
   cd <repo_dir>
   ```
2. (Optional) Set up a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Install the `xor_popcount_cuda` extension (CUDA required):
   ```bash
   # From the src/cu-popcount
   python setup.py install
   ```

## Usage Examples

### Sparse vs. Dense Evaluation
```bash
python src/example.py \
    --model_id meta-llama/Llama-3.2-1B \
    --device cuda \
    --threshold 28 \
    --k 8 \
    --window_size 4
```

### ITQ Conditioning
```bash
python src/condition_itq.py \
    --model_id meta-llama/Llama-3.2-1B \
    --dataset_triple wikitext/wikitext-2-raw-v1/validation \
    --tokens 1024 \
    --outfile saved_model.pt
```
