#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>

namespace {

/* 64-bit intrinsic, falls back to 32-bit if you ever use int32 */
__device__ inline int pc64(uint64_t x) { return __popcll(x); }
__device__ inline int pc32(uint32_t x) { return __popc(x); }

/* one thread ↔ one (b,q,k) triple */
template <typename scalar_t>
__global__ void xor_popcount_kernel(const scalar_t* __restrict__ A,
                                    const scalar_t* __restrict__ B,
                                    int64_t* __restrict__ O,
                                    int64_t Bsz, int64_t Q, int64_t K,
                                    int64_t D)
{
    const int64_t b = blockIdx.x;
    const int64_t q = blockIdx.y;
    const int64_t k = blockIdx.z * blockDim.x + threadIdx.x;
    if (k >= K) return;

    const int64_t rowA = ((b * Q) + q) * D;
    const int64_t rowB = ((b * K) + k) * D;
    int64_t pop = 0;

    #pragma unroll 4
    for (int64_t d = 0; d < D; ++d) {
        if constexpr (sizeof(scalar_t) == 8)
            pop += pc64( static_cast<uint64_t>(A[rowA + d] ^ B[rowB + d]) );
        else
            pop += pc32( static_cast<uint32_t>(A[rowA + d] ^ B[rowB + d]) );
    }
    O[((b * Q) + q) * K + k] = pop;
}

} // anon ns

#include <torch/extension.h>
#include <ATen/ATen.h>
using torch::IntArrayRef;

at::Tensor xor_popcount_launcher(const at::Tensor& A, const at::Tensor& B)
{
    TORCH_CHECK(A.scalar_type() == B.scalar_type(), "dtype mismatch");
    TORCH_CHECK(A.size(-1) == B.size(-1), "last-dim (D) mismatch");

    // --- figure shapes ---
    const int64_t D = A.size(-1);
    const int64_t Q = A.size(-2); 
    const int64_t K = B.size(-2); 

    auto leadA = A.sizes().slice(0, A.dim() - 2);
    auto leadB = B.sizes().slice(0, B.dim() - 2);
    std::vector<int64_t> leadOut = at::infer_size(leadA, leadB);   // broadcast prefix

    // build full shapes
    auto shapeA = leadOut; shapeA.push_back(Q); shapeA.push_back(D);   // [..., Q, D]
    auto shapeB = leadOut; shapeB.push_back(K); shapeB.push_back(D);   // [..., K, D]

    // --- broadcast & add singleton dims ---
    auto Aexp = A.expand(shapeA).unsqueeze(-2);   // [...,  Q, 1, D]
    auto Bexp = B.expand(shapeB).unsqueeze(-3);   // [...,  1, K, D]

    // flatten for the kernel
    auto Aflat = Aexp.contiguous().view({-1, Q, D});
    auto Bflat = Bexp.contiguous().view({-1, K, D});


    // ----- 3. allocate output + launch
    auto Out = at::empty({Aflat.size(0), Q, K},
                         A.options().dtype(at::kLong));

    const dim3 grid(Aflat.size(0), Q, (K + 31) / 32);
    const dim3 block(32);
    AT_DISPATCH_INTEGRAL_TYPES(A.scalar_type(), "xor_popcount_kernel", ([&]{
        xor_popcount_kernel<scalar_t><<<grid, block>>>(
            Aflat.data_ptr<scalar_t>(),
            Bflat.data_ptr<scalar_t>(),
            Out.data_ptr<int64_t>(),
            Aflat.size(0), Q, K, D);
    }));

    // ----- 4. reshape back to broadcasted shape + {Q, K}
    leadOut.push_back(Q);
    leadOut.push_back(K);
    return Out.view(leadOut);
}

