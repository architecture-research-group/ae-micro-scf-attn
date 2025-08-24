import torch, xor_popcount_cuda as xpc
import time
import numpy as np
import torch.nn as nn

B,K = 1, 8
C = 1024*1024
Q = 4
D = 64 // 64
# L = 32

times = []

ksb = torch.randint(0, 15, (B,K,Q,C,D), dtype=torch.int64).contiguous()
qsb = torch.randint(0, 15, (B,K,Q,1,D), dtype=torch.int64).contiguous()
ksb = ksb.to("cuda")
qsb = qsb.to("cuda")
xpc.xor_popcount(qsb, ksb)

i = 0

for i in range(1000):
    qsb = torch.randint(0, 15, (B,K,Q,1,D), dtype=torch.int64).contiguous()
    qsb = qsb.to("cuda")
    s = time.time()
    out = xpc.xor_popcount(qsb, ksb)
    e = time.time()
    # print(out[0,0,0,0])
    if i > 2:
        times.append(e - s)

time = sum(times) / len(times)

query_size = B * K * Q * D * 8
key_size = K * C * D * 8
print(f"Query size: {query_size / 2**20:.2f} MB")
print(f"Key size: {key_size / 2**20:.2f} MB")
print(f"{time * 10**6:.2f} µs")

print(f"{key_size / time / 2**30:.2f} GB/s")



    
