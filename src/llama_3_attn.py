import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch import einsum
import torch.nn as nn
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
import code
import xor_popcount_cuda as xpc
from functools import lru_cache

def copy_llama_attention_weights(source, dest, config):

    with torch.no_grad():
        q_weight = source.q_proj.weight.contiguous()
        k_weight = source.k_proj.weight.contiguous()
        v_weight = source.v_proj.weight.contiguous()
        o_weight = source.o_proj.weight.contiguous()

        dest.q_proj.weight.copy_(q_weight)
        dest.k_proj.weight.copy_(k_weight)
        dest.v_proj.weight.copy_(v_weight)
        dest.out_proj.weight.copy_(o_weight)



def itq_no_pca(
    X, 
    n_iter=50, 
    device="cpu"
):
    """
    ITQ (Iterative Quantization) in PyTorch, *without* PCA.
    
    Args:
        X (torch.Tensor): Shape [N, D], your data.
        n_iter (int): Number of iterations for the rotation update.
        device (str): 'cpu' or 'cuda'.

    Returns:
        B (torch.Tensor): [N, D] final binary codes (+1 / -1).
        R (torch.Tensor): [D, D] learned rotation matrix.
    """
    # Move data to device
    X = X.to(device).float()
    N, D = X.shape

    # Initialize rotation to identity or random orthonormal
    R = torch.eye(D, device=device)

    for _ in range(n_iter):
        # 1) Compute binary codes by sign
        Z = X @ R        # [N, D]
        B = torch.sign(Z).float()

        # 2) Orthogonal Procrustes: R = argmin_{R^T R=I} ||B - X R||^2
        #    => R = V U^T from SVD of (B^T X)
        M = B.t() @ X    # shape [D, D]
        U, _, Vt = torch.svd(M.float())  # full SVD
        # Orthogonal R = V * U^T
        R = Vt.t() @ U.t()

    # Final codes
    B = torch.sign(X @ R)

    return R

def itq_no_pca_on_heads(X, n_iter=50, device="cpu"):
    """
    Perform ITQ for each head in X, shape [B, H, L, D], *without* PCA.
    """
    B, H, L, D = X.shape
    X = X.to(device).to(torch.float32)

    complete_R = torch.empty(H, D, D, device=device)

    for h in range(H):
        # Extract the head => [B, L, D]
        X_head = X[:, h, :, :]  # shape [B, L, D]
        # Flatten to [B*L, D]
        X_flat = X_head.reshape(B * L, D).float()

        # Run ITQ without PCA
        R = itq_no_pca(X_flat, n_iter=n_iter, device=device)
        complete_R[h] = R

    return complete_R


class L3KSparseDotProd(nn.Module):
    def __init__(self, lsh_threshold, num_heads):
        """
        Args:
            k: number of top dot products to retain per query.
            lsh_threshold: static threshold (number of matching bits) that a key must exceed.
        """
        super().__init__()
        # Store thresholds per head (assuming 12 heads as default)
        self.register_buffer("lsh_threshold", torch.full((num_heads,), lsh_threshold, dtype=torch.float))
        self.register_buffer("lsh_survivors", torch.zeros(num_heads, dtype=torch.float))
        self.register_buffer("causal_survivors", torch.tensor(0.0, dtype=torch.float))
        self.register_buffer("dp_cache", torch.empty((1,), dtype=torch.float))
        self.register_buffer("final_mask", torch.empty((1,), dtype=torch.bool))

    @torch.jit.export
    def reset_analytics(self):
        self.lsh_survivors.zero_()
        self.causal_survivors.zero_()

    @torch.jit.export
    def set_lsh_threshold(self, threshold):
        self.lsh_threshold.copy_(threshold)

    @torch.jit.export
    def get_mask(self):
        return self.final_mask

    @torch.jit.export
    def filter_ratios(self):
        # ratio = causal / lsh
        # shape: (num_heads,)
        return self.causal_survivors / (self.lsh_survivors + 1e-8)

    @torch.jit.export
    def packbits64(self, t: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
        """
        t : bool / 0-1 int tensor
        packs every 64 bits along `dim` into one int64
        drops into cuda if `t` is on cuda
        """
        bitw = 64
        orig_len = t.size(dim)
        pad = (-orig_len) % bitw          # zero-pad so len % 64 == 0
        if pad:
            pad_cfg = [0]*(-2*dim-1) + [pad, 0]   # F.pad wants pairs per dim
            t = F.pad(t, pad_cfg)

        t = t.view(*t.shape[:-1], -1, bitw)        # (..., n_words, 64)
        shifts = torch.arange(bitw, device=t.device, dtype=torch.int64)   # 0..63
        packed = (t.long() << shifts).sum(-1)      # (..., n_words)
        return packed.contiguous()


    @torch.compile
    def lsh_then_topk(self, \
            # LSH args
            # Query/key signs
            qsb: torch.Tensor, \
            ksb: torch.Tensor, \
            
            # Threshold for LSH. Refactor to use self... and remove from args
            lsh_threshold_tensor: torch.Tensor, \
            
            # Original query/key used for top-k and softmax
            query: torch.Tensor, \
            key: torch.Tensor, \
            
            # Value tensor used for E'
            value: torch.Tensor, \
            
            # Top-k args
            top_k: int, \
            
            # Attention mask (typically causal)
            mask: torch.Tensor, \
            
            # Window mask for short-term attention
            window_mask: torch.Tensor\
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        B, KH, QH, chunk_size, D = query.shape
        D_tensor = torch.tensor(D, dtype=torch.float)

        # What is the search space?
        lsh_options = torch.logical_and(mask, torch.logical_not(window_mask))        
        causal_survivors = torch.einsum("bkqcl->k", lsh_options)

        out = D_tensor - xpc.xor_popcount(qsb, ksb)
        out_max = out.max(dim=2, keepdim=True).values

        # Check thresholding and update mask
        lsh_mask = out_max > lsh_threshold_tensor
        mask = torch.logical_and(mask, lsh_mask)

        # Within the search space, how many survive LSH?
        top_k_options = torch.logical_and(mask, torch.logical_not(window_mask))
        lsh_survivors = torch.einsum("bkqcl->k", top_k_options)

        # Compute top-k mask. Would like to use a sparse matmul eventually.
        scores_chunk = torch.einsum("bkqcd,bkqld->bkqcl", query, key) / torch.sqrt(D_tensor)
        scores_chunk = scores_chunk.masked_fill_(~mask & ~window_mask, -1e4) # Remove non-window and non-LSH tokens

        # BEGIN TOP-K
        # scores_long_term = scores_chunk.clone()
        scores_long_term = scores_chunk.masked_fill(window_mask, -1e4) # Exclude dense window from top-k

        _, topk_indices = scores_long_term.topk(top_k, dim=-1, sorted=False)
        mask = torch.zeros_like(scores_long_term, dtype=torch.bool) \
            .scatter_(-1, topk_indices, True)
        # END TOP-K


        # Combine masks
        both_masks = window_mask | mask

        # Apply top-k mask and softmax
        constant = torch.full_like(scores_chunk, -1e4)
        scores_masked = scores_chunk.where(both_masks, constant)
        scores_probs= torch.nn.functional.softmax(scores_masked, dim=-1)

        # Compute E'
        output = torch.einsum("bkqcl,bkqld->bckqd", scores_probs, value)
        output = output.reshape(B, chunk_size, -1)
        return (output, causal_survivors, lsh_survivors, lsh_mask)
                      

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, top_k: int, window_size: int) -> torch.Tensor:

        B, H, q_len, D = query.shape
        _, KH, k_len, _ = key.shape

        QH = H // KH
        query = query.view(B, KH, QH, q_len, D)
        key = key.unsqueeze(2).repeat(1, 1, QH, 1, 1)
        value = value.unsqueeze(2).repeat(1, 1, QH, 1, 1)

        # --- LSH Masking (XNOR of sign bits) ---
        ksb = self.packbits64(key >= 0)
        qsb = self.packbits64(query >= 0)

        lsh_threshold_tensor = self.lsh_threshold.view(1, KH, 1,1,1)

        chunk_size = 128


        causal_survivors_local = torch.zeros(KH, dtype=torch.float, device=query.device)
        lsh_survivors_local = torch.zeros(KH, dtype=torch.float, device=query.device)

        output = torch.zeros(B, q_len, D*H, dtype=query.dtype, device=query.device)


        for i in range(0, q_len, chunk_size):
            causal_mask_chunk = (i + torch.arange(chunk_size, device=query.device).unsqueeze(1) -
                                torch.arange(k_len, device=query.device).unsqueeze(0)) >= 0

            final_mask_chunk = torch.ones(B, KH, QH, chunk_size, k_len, dtype=torch.bool, device=query.device)
            window_mask_chunk = (i + torch.arange(chunk_size, device=query.device).unsqueeze(1) -
                                torch.arange(k_len, device=query.device).unsqueeze(0)).abs() < window_size
            
            # Take first 128 tokens
            q_idx = i + torch.arange(chunk_size, device=query.device).unsqueeze(1)  # [chunk_size, 1]
            k_idx = torch.arange(k_len,  device=query.device).unsqueeze(0)   
            sink_mask_chunk   = k_idx < 16          # shape [1, k_len] → broadcast later
            window_mask_chunk = window_mask_chunk | sink_mask_chunk

            orig_q_chunk = query[:, :, :, i:i+chunk_size, :]
            qsb_chunk = qsb[:, :, :, i:i+chunk_size, :]

            final_mask_chunk = final_mask_chunk & causal_mask_chunk
            window_mask_chunk = window_mask_chunk & causal_mask_chunk

            output_chunk, causal_survivors, lsh_survivors, long_term_chunk = self.lsh_then_topk(qsb_chunk, ksb, lsh_threshold_tensor, orig_q_chunk, key, value, top_k, final_mask_chunk, window_mask_chunk)
            causal_survivors_local += causal_survivors
            lsh_survivors_local += lsh_survivors

            output[:, i:i+chunk_size, :] = output_chunk



        self.lsh_survivors = self.lsh_survivors + lsh_survivors_local
        self.causal_survivors = self.causal_survivors + causal_survivors_local

        return output

class L3ITQ(torch.nn.Module):
    def __init__(self, num_attention_heads=32, num_key_value_heads = 8, head_dim= 64, hidden_size=2048):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_size = hidden_size
        self.head_dim = head_dim

        self.k_t = torch.eye(head_dim).repeat(num_key_value_heads, 1, 1).to(torch.bfloat16)
        self.q_t = torch.eye(head_dim).repeat(num_attention_heads, 1, 1).to(torch.bfloat16)
        self.k_t = torch.nn.Parameter(self.k_t)
        self.q_t = torch.nn.Parameter(self.q_t)


    def store(self, itq_matrix):
        self.k_t = torch.nn.Parameter(itq_matrix)
        repeats = int(self.num_attention_heads/self.num_key_value_heads)
        self.q_t = torch.nn.Parameter(itq_matrix.repeat_interleave(repeats, dim=0))

    def forward(self, Q,K):
        Q = Q @ self.q_t
        K = K @ self.k_t
        return Q, K
        

class L3_Sparse_Attn(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size

        self.head_dim = self.hidden_size // self.num_attention_heads
        self.split_size = self.hidden_size

        self.q_proj = torch.nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=False, dtype=torch.bfloat16)
        self.k_proj = torch.nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False, dtype=torch.bfloat16)
        self.v_proj = torch.nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False, dtype=torch.bfloat16)
        self.out_proj = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False, dtype=torch.bfloat16)
        self.dot_product = L3KSparseDotProd(16, self.num_key_value_heads)
        self.k = 128

        self.trained = False
        self.itq = L3ITQ(self.num_attention_heads, self.num_key_value_heads, self.head_dim, self.hidden_size)
        self.window_size = 0

    def set_k(self,k):
        self.k = k

    def set_threshold(self, threshold):
        self.dot_product.set_lsh_threshold(threshold)
    
    def set_window_size(self, window_size):
        self.window_size = window_size

    def get_threshold(self):
        return self.dot_product.lsh_threshold

    def _split_heads(self, x, batch_size):
        # x: [batch, seq_length, hidden_size] -> [batch, num_heads, seq_length, head_dim]
        num_heads = x.size(2) // self.head_dim
        return x.contiguous().view(batch_size, -1, num_heads, self.head_dim).transpose(1, 2)
    
    def _merge_heads(self, x, batch_size):
        # Merge heads: [batch, num_heads, seq_len, head_dim] into [batch, seq_len, hidden_size]
        return x.transpose(1, 2).contiguous().view(batch_size, -1, self.split_size)


    @torch.compile
    def forward(self, hidden_states, position_embeddings, attention_mask,
                use_cache=False, output_attentions=False, **kwargs):
        
        batch_size, q_len, _ = hidden_states.size()

        query = self.q_proj(hidden_states)

        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = self._split_heads(query, batch_size)
        key = self._split_heads(key, batch_size)
        value = self._split_heads(value, batch_size)

        # if kwargs.get("train_itq", False) and not self.trained:
        #     R = itq_no_pca_on_heads(key.to(torch.float32), n_iter=10, device='cuda')
        #     self.itq.store(R)
        #     self.trained = True

        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        if kwargs.get("train_itq", False) and not self.trained:
            R = itq_no_pca_on_heads(key.to(torch.float32), n_iter=10, device='cuda')
            self.itq.store(R)
            self.trained = True
        query, key = self.itq(query, key)


        if kwargs.get("return_query", False):
            return query, key, value

        present = (key, value) if use_cache else None


        attn_probs = self.dot_product(query, key, value, self.k, self.window_size)


        attn_output = self.out_proj(attn_probs)

        return attn_output, present

