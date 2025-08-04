import argparse

import torch
from transformers import AutoTokenizer

from models import KSparseModel
from datasets import load_dataset

def prepare_chunks(dataset_triple, tokenizer, chunk_size, stride, device):
    ds_name, ds_subset, ds_split = dataset_triple.split("/")

    dataset = load_dataset(ds_name, ds_subset, split=ds_split)

    entries= []
    for i, entry in enumerate(dataset):
        entries.append(entry['text'])
        if i > 10000:
            break

    full_text = "\n".join(entries)

    encoded = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=chunk_size)
    print(f"Total tokens: {len(encoded['input_ids'][0])}")
    chunks = []
    start = 0
    total_len = len(encoded["input_ids"][0])
    while start < total_len:
        end = min(start + chunk_size, total_len)
        chunks.append({k: v[:, start:end] for k, v in encoded.items() if k != "overflow"})
        start += stride
    return [{k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)} for inputs in chunks]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", "-m", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--dataset_triple", "-ds", type=str, default="wikitext/wikitext-2-raw-v1/validation")
    parser.add_argument("--tokens", "-t", type=int, default=1024)
    parser.add_argument("--outfile", "-mp", type=str, default="saved_model.pt")
    args = parser.parse_args()

    model_id = args.model_id
    dataset_triple = args.dataset_triple

    model = KSparseModel.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.to("cuda")
    model.eval()

    chunk = prepare_chunks(dataset_triple, tokenizer, args.tokens, args.tokens, "cuda")
    with torch.no_grad(), torch.amp.autocast("cuda"):
        model(input_ids = chunk[0]["input_ids"], train_itq = True)

    torch.save(model.state_dict(), args.outfile)