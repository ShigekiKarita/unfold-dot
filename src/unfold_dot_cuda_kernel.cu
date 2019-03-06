#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <array>
#include <assert.h>

namespace {

    template <typename T, int N>
    struct Stride {
        T* data;
        ptrdiff_t strides[N];
        ptrdiff_t sizes[N];

        Stride(at::Tensor t) {
            this->data = t.data<T>();
            for (int n = 0; n < N; ++n) {
                this->strides[n] = t.stride(n);
                this->sizes[n] = t.size(n);
            }
        }

        __host__ __device__
        T* pointer(std::initializer_list<ptrdiff_t> il) {
            return const_cast<T*>(static_cast<const Stride<T, N>&>(*this).pointer(il));
        }

        __host__ __device__
        const T* pointer(std::initializer_list<ptrdiff_t> il) const {
            ptrdiff_t ret = 0;
            int n = 0;
            for (auto i : il) {
                assert(0 <= i);
                assert(i < this->sizes[n]);
                ret += i * this->strides[n];
                ++n;
            }
            return data + ret;
        }

    };

#define PARALLEL_FOR(i, n) \
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

    template <typename scalar_t>
    __global__ void unfold_dot_cuda_forward_kernel(
        Stride<scalar_t, 4> ret_tensor,         // (batch, head, time1, restrict)
        const Stride<scalar_t, 4> query_tensor, // (batch, head, time1, feat)
        const Stride<scalar_t, 4> key_tensor,   // (batch, head, time2, feat)
        size_t parallel_size
        ) {
        const auto batch_size = ret_tensor.sizes[0];
        const auto head_size = ret_tensor.sizes[1];
        const auto time_query = query_tensor.sizes[2];
        const auto time_key = key_tensor.sizes[2];
        const auto feat_size = query_tensor.sizes[3];
        const auto restrict_size = ret_tensor.sizes[3];
        const ptrdiff_t window = (restrict_size - 1) / 2;
        PARALLEL_FOR(i, parallel_size) {
            const ptrdiff_t batch_index = i / (head_size * time_query);
            const ptrdiff_t head_index = (i % (head_size * time_query)) / time_query;
            const ptrdiff_t time_query_index = i % time_query;

            auto* ret_i = ret_tensor.pointer({batch_index, head_index, time_query_index, 0});
            const auto* query_i = query_tensor.pointer({batch_index, head_index, time_query_index, 0});
            const auto* key_i = key_tensor.pointer({batch_index, head_index, time_query_index, 0});

            for (ptrdiff_t w = -window; w <= window; ++w) {
                const ptrdiff_t time_key_index = time_query_index + w;
                if (time_key_index < 0) continue;
                if (time_key_index >= time_key) break;

                // TODO parallel reduction
                scalar_t acc = 0;
                for (ptrdiff_t f = 0; f < feat_size; ++f) {
                    acc += query_i[f * query_tensor.strides[3]] * key_i[w * key_tensor.strides[2] + f * key_tensor.strides[3]];
                }
                ret_i[(w + window) * ret_tensor.strides[3]] = acc;
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
                    Stride<scalar_t, 4>(ret),
                    Stride<scalar_t, 4>(query),
                    Stride<scalar_t, 4>(key),

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

