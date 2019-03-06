#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <array>

namespace {

#define PARALLEL_FOR(i, n) \
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

    template <typename scalar_t>
    __global__ void unfold_dot_cuda_forward_kernel(
        scalar_t* __restrict__ ret,         // (batch, head, time1, restrict)
        const scalar_t* __restrict__ query, // (batch, head, time1, feat)
        const scalar_t* __restrict__ key,   // (batch, head, time2, feat)
        size_t restrict_size,
        size_t batch_head_size,
        size_t time_query,
        size_t time_key,
        size_t feat_size,
        size_t parallel_size
        ) {
        const ptrdiff_t window = (restrict_size - 1) / 2;
        PARALLEL_FOR(i, parallel_size) {
            const ptrdiff_t batch_head_index = i / time_query;
            const ptrdiff_t time_query_index = i % time_query;

            scalar_t* ret_i = ret + batch_head_index * time_query * restrict_size + time_query_index * restrict_size;
            const scalar_t* query_i = query + batch_head_index * time_query * feat_size + time_query_index * feat_size;
            const scalar_t* key_i = key + batch_head_index * time_key * feat_size + time_query_index * feat_size;

            for (ptrdiff_t w = -window; w <= window; ++w) {
                const ptrdiff_t time_key_index = time_query_index + w;
                if (time_key_index < 0) continue;
                if (time_key_index >= time_key) break;

                ret_i[w + window] = 0;
                for (ptrdiff_t f = 0; f < feat_size; ++f) {
                    ret_i[w + window] += query_i[f] * key_i[w * feat_size + f];
                }
            }
        }
    }


    template <typename scalar_t>
    __global__ void unfold_dot_cuda_backward_kernel(
        scalar_t* __restrict__ dquery, // (batch, head, time1, feat)
        scalar_t* __restrict__ dkey,   // (batch, head, time2, feat)
        const scalar_t* __restrict__ dret,  // (batch, head, time1, restrict)
        const scalar_t* __restrict__ query, // (batch, head, time1, feat)
        const scalar_t* __restrict__ key,   // (batch, head, time2, feat)
        size_t restrict_size,
        size_t batch_head_size,
        size_t time_query,
        size_t time_key,
        size_t feat_size,
        size_t parallel_size
        ) {
        const ptrdiff_t window = (restrict_size - 1) / 2;
        const ptrdiff_t rev_offset = restrict_size - 1 - window;
        // parallel for each (batch, head, time1, feat). sequential for each (restrict)
        PARALLEL_FOR(i, parallel_size) {
            const ptrdiff_t batch_head_index = i / (time_query * feat_size);
            const ptrdiff_t remain = i % (time_query * feat_size);
            const ptrdiff_t time_query_index = remain / feat_size;
            const ptrdiff_t f = remain % feat_size;

            // poautoer to (batch, head, time1)
            const scalar_t* dret_i = dret + batch_head_index * time_query * restrict_size + time_query_index * restrict_size;
            // poautoer to (batch, head, time1, feat)
            const scalar_t* query_i = query + batch_head_index * time_query * feat_size + time_query_index * feat_size + f;
            const scalar_t* key_i = key + batch_head_index * time_key * feat_size + time_query_index * feat_size + f;
            scalar_t* dquery_i = dquery + batch_head_index * time_query * feat_size + time_query_index * feat_size + f;
            scalar_t* dkey_i = dkey + batch_head_index * time_key * feat_size + time_query_index * feat_size + f;

            *dquery_i = 0;
            *dkey_i = 0;
            // for (auto w = -min(time_query_index, window); w <= min(time_query - time_query_index + 1, window); ++w) {
            for (auto w = -window; w <= window; ++w) {
                auto t = time_query_index + w;
                if (0 <= t && t < time_key) {
                    *dquery_i += dret_i[w + window] * key_i[w * feat_size];
                    *dkey_i += dret_i[w * restrict_size + (rev_offset - w)] * query_i[w * feat_size];
                }
            }
        }
    }

} // namespace

at::Tensor unfold_dot_cuda_forward(
    at::Tensor query,           // (batch, head, time1, feat)
    at::Tensor key,             // (batch, head, time2, feat)
    int64_t restrict_size
    )
{
    auto batch = query.size(0);
    auto head = query.size(1);
    auto time1 = query.size(2);
    auto feat = query.size(3);
    // (batch, head, time1, restrict)
    auto ret = at::zeros({batch, head, time1, restrict_size}, query.options());
    const size_t parallel_size = batch * head * time1;

    const int threads = 1024;
    const dim3 blocks((parallel_size + threads - 1) / threads, parallel_size);

    AT_DISPATCH_FLOATING_TYPES(ret.type(), "unfold_dot_forward_cuda", ([&] {
                unfold_dot_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
                    ret.data<scalar_t>(),
                    query.data<scalar_t>(),
                    key.data<scalar_t>(),
                    restrict_size,
                    batch * head,
                    query.size(2),
                    key.size(2),
                    feat,
                    parallel_size);
            }));

    return ret;
}


std::array<at::Tensor, 2> unfold_dot_cuda_backward(
    at::Tensor dret,            // (batch, head, time1, restrict)
    at::Tensor query,           // (batch, head, time1, feat)
    at::Tensor key              // (batch, head, time2, feat)
    )
{
    auto batch = query.size(0);
    auto head = query.size(1);
    auto time1 = query.size(2);
    auto feat = query.size(3);
    auto restrict_size = dret.size(3);
    // (batch, head, time1, restrict)
    auto dquery = at::empty_like(query);
    auto dkey = at::empty_like(key);
    const size_t parallel_size = batch * head * time1 * feat;
    const int threads = 1024;
    const dim3 blocks((parallel_size + threads - 1) / threads, parallel_size);

    AT_DISPATCH_FLOATING_TYPES(dret.type(), "unfold_dot_backward_cuda", ([&] {
                unfold_dot_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
                    dquery.data<scalar_t>(),
                    dkey.data<scalar_t>(),
                    dret.data<scalar_t>(),
                    query.data<scalar_t>(),
                    key.data<scalar_t>(),
                    restrict_size,
                    batch * head,
                    query.size(2),
                    key.size(2),
                    feat,
                    parallel_size);
            }));

    return {dquery, dkey};
}

