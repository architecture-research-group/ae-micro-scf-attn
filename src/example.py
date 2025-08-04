import argparse

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from models import KSparseModel

def example_input():
    text = """LongSight: Compute-Enabled Memory to Accelerate Large-Context LLMs via Sparse Attention
Large input context windows for transformer-based LLMs help minimize hallucinations and generate accurate, personalized output. As the context window increases, the attention phase in transformerbased LLMs tends to dominate execution time. In that regard, KV caching helps reduce computational cost by avoiding redundant computation. However, with larger context windows, the size of the KV cache can quickly exceed the capacity of current GPUs’ highbandwidth memory (HBM). In this work, we present LongSight, an algorithm-hardware codesign framework for attention acceleration in large-context scenarios. LongSight leverages a novel ComputeEnabled Memory (CEM) architecture that seamlessly combines inand near-memory processing to elevate the value of relatively lowcost LPDDR DRAM to that of high-end, expensive HBM. Specifically, CEM can be used to scale the memory capacity of a neural processing unit (e.g., GPUs or TPUs) beyond that of its attached HBM, and CEM’s built-in processing capability ensures performance scalability. We demonstrate that LongSight, equipped with a single GPU and a single CEM device, can efficiently support context lengths of up to 1 million tokens for state-of-the-art Llama models."""

    # Max length must be a multiple of 128
    chunks = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    chunks.to(device)
    return chunks

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", '-m', type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--model_path", '-mp', type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threshold", "-t", type=float, default=28)
    parser.add_argument('--k', '-k', type=int, default=8)
    parser.add_argument('--window_size', '-ws', type=int, default=4)
    args = parser.parse_args()


    model_id = args.model_id
    device = args.device
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    torch.set_float32_matmul_precision('high')

    input_data = example_input()["input_ids"]
    num_tokens = input_data.shape[1]


    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    model.to("cuda")
    model.eval()
    with torch.no_grad(), torch.amp.autocast("cuda"):
        base_perplexity = model(input_ids=input_data, labels=input_data).loss.item()
    print(f"K=all Model={args.model_id} ppl={base_perplexity:.2f}")


    model_sparse = KSparseModel.from_pretrained(model_id).cuda()
    layers, heads, kvh = KSparseModel.get_shape(model_sparse)
    if args.model_path is not None:
        state_dict = torch.load(args.model_path)
        model_sparse.load_state_dict(state_dict, strict=False)

    model_sparse.eval()
    
    # Set up thresholds, K, and window size for each attention layer.
    thresholds = torch.full((layers,kvh), args.threshold, dtype=torch.float)
    for attn,th in zip(KSparseModel.get_attns(model_sparse),thresholds):
        attn.set_k(args.k)
        attn.set_window_size(args.window_size)
        attn.set_threshold(th)
        attn.dot_product.reset_analytics()

    with torch.no_grad(), torch.amp.autocast("cuda"):
        sparse_perplexity = model_sparse(input_ids=input_data, labels=input_data).loss.item()


    # Computing filter ratio.
    keys_loaded = 0
    values_loaded = 0
    baseline_kv_loaded = 0
    filter_ratios = []

    for layer,attn in enumerate(KSparseModel.get_attns(model_sparse)):
        layer_ratios = attn.dot_product.filter_ratios()
        for head, ratio in enumerate(layer_ratios):
            filter_ratios.append((layer,head,ratio))
    for layer in range(layers):
        # Take the minimum FR since this is bottlenecked by the worst head in hardware.
        this_block_filters = [r for l,_h,r in filter_ratios if l==layer]
        min_filter_ratio = min(this_block_filters)
        cem_total_tokens = num_tokens - args.window_size
        cem_keys = cem_total_tokens/min_filter_ratio

        # If less than K pass filtering, CEM returns fewer values.
        cem_values = min(cem_keys, args.k)
        
        keys_loaded += cem_keys
        values_loaded += cem_values

        kv_loaded = keys_loaded + values_loaded
        baseline_kv_loaded += 2 * (num_tokens - args.window_size)  # Baseline is all keys and values loaded.


    print(f"Model={args.model_id} ws={args.window_size} itq={args.model_path is not None} context_length={num_tokens} K={args.k} ppl={sparse_perplexity:.2f} kv_fr={baseline_kv_loaded/kv_loaded:.2f}")

