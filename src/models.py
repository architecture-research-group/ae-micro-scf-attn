from transformers import AutoModelForCausalLM
from llama_3_attn import L3_Sparse_Attn, copy_llama_attention_weights
import torch



class KSparseModel(AutoModelForCausalLM):
    def __init__(self):
        super.__init__()

    def from_pretrained(pretrained_model_name_or_path):
        match pretrained_model_name_or_path:
            case "meta-llama/Llama-3.2-1B":
                model_original = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path)
                model_new = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path)
                config = model_original.config
                for source, dest in zip(model_original.model.layers, model_new.model.layers):
                    dest.self_attn = L3_Sparse_Attn(config)

                    copy_llama_attention_weights(source.self_attn, dest.self_attn, config)
                model_new.sparse_kind = "llama3.2"
                return model_new
            case "meta-llama/Llama-3.1-8B":
                model_original = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.bfloat16)
                model_new = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.bfloat16)
                config = model_original.config
                for source, dest in zip(model_original.model.layers, model_new.model.layers):
                    dest.self_attn = L3_Sparse_Attn(config)

                    copy_llama_attention_weights(source.self_attn, dest.self_attn, config)
                model_new.sparse_kind = "llama3.1"
                return model_new
            case _:
                print("Model not supported")
    
    def get_attns(model):
        match model.sparse_kind:
            case "llama3.2"| "llama3.1":
                return [l.self_attn for l in model.model.layers]
            case _:
                print("Model not supported")
    
    def get_shape(model):
        match model.sparse_kind:
            case "llama3.2":
                return (16, 32, 8)
            case "llama3.1":
                return (32, 32, 8)
            case _:
                print("Model not supported")
