#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

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
        const int window = (restrict_size - 1) / 2;
        PARALLEL_FOR(i, parallel_size) {
            const int batch_head_index = i / time_query;
            const int time_query_index = i % time_query;

            scalar_t* ret_i = ret + batch_head_index * time_query * restrict_size + time_query_index * restrict_size;
            const scalar_t* query_i = query + batch_head_index * time_query * feat_size + time_query_index * feat_size;
            const scalar_t* key_i = key + batch_head_index * time_key * feat_size + time_query_index * feat_size;

            for (int w = -window; w <= window; ++w) {
                const int time_key_index = time_query_index + w;
                if (time_key_index < 0) continue;
                if (time_key_index >= time_key) break;

                ret_i[w + window] = 0;
                for (int f = 0; f < feat_size; ++f) {
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
        const int window = (restrict_size - 1) / 2;
        // parallel for each (batch, head, time1, feat). sequential for each (restrict)
        PARALLEL_FOR(i, parallel_size) {
            const int batch_head_index = i / (time_query * feat_size);
            const int remain = i % (time_query * feat_size);
            const int time_query_index = remain / feat_size;
            const int f = remain % feat_size;

            const scalar_t* query_i = query + batch_head_index * time_query * feat_size + time_query_index * feat_size;
            const scalar_t* key_i = key + batch_head_index * time_key * feat_size + time_query_index * feat_size;
            const scalar_t* dret_i = dret + batch_head_index * time_query * restrict_size + time_query_index * restrict_size;
            scalar_t* dquery_i = dquery + batch_head_index * time_query * feat_size + time_query_index * feat_size;
            scalar_t* dkey_i = dkey + batch_head_index * time_key * feat_size + time_query_index * feat_size;

            // TODO: parallel for each f
            dquery_i[f] = 0;
            dkey_i[f] = 0;
            for (int w = -window; w <= window; ++w) {
                const int time_key_index = time_query_index + w;
                if (time_key_index < 0) continue;
                if (time_key_index >= time_key) break;
                
                dquery_i[f] += dret_i[w + window] * key_i[w * feat_size + f];
                dkey_i[f] += dret_i[w + window] * query_i[w * feat_size + f];
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


std::vector<at::Tensor> unfold_dot_cuda_backward(
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

