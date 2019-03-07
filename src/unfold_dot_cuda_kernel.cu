#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <array>
#include <assert.h>
#include <iostream>

namespace {

    template <typename T, int N>
    struct Stride {
        T* data;
        ptrdiff_t strides[N];
        ptrdiff_t sizes[N];

        Stride(const at::Tensor& t) : data(t.data<T>()) {
            static_assert(sizeof(Stride<T, N>) < sizeof(void*) * 16,
                          "CUDA kernel launch will exeed resources");
            std::copy(t.strides().begin(), t.strides().end(), strides);
            std::copy(t.sizes().begin(), t.sizes().end(), sizes);
        }

        __device__
        T* pointer(std::initializer_list<ptrdiff_t> il) {
            return const_cast<T*>(static_cast<const Stride<T, N>&>(*this).pointer(il));
        }

        __device__
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
        const auto head_size = ret_tensor.sizes[1];
        const auto time_query = query_tensor.sizes[2];
        const auto feat_size = query_tensor.sizes[3];
        const auto restrict_size = ret_tensor.sizes[3];
        const ptrdiff_t window = (restrict_size - 1) / 2;
        // parallel for batch, head, time_query
        PARALLEL_FOR(i, parallel_size) {
            const ptrdiff_t batch_index = i / (head_size * time_query);
            const ptrdiff_t head_index = (i % (head_size * time_query)) / time_query;
            const ptrdiff_t time_query_index = i % time_query;

            auto* ret_i = ret_tensor.pointer({batch_index, head_index, time_query_index, 0});
            const auto* query_i = query_tensor.pointer({batch_index, head_index, time_query_index, 0});
            const auto* key_i = key_tensor.pointer({batch_index, head_index, time_query_index, 0});

            for (ptrdiff_t w = -min(time_query_index, window); w <= window; ++w) {
                if (key_tensor.sizes[2] <= time_query_index + w) break;

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
        Stride<scalar_t, 4> dquery_tensor,      // (batch, head, time1, feat)
        Stride<scalar_t, 4> dkey_tensor,        // (batch, head, time2, feat)
        const Stride<scalar_t, 4> dret_tensor,  // (batch, head, time1, restrict)
        const Stride<scalar_t, 4> query_tensor, // (batch, head, time1, feat)
        const Stride<scalar_t, 4> key_tensor,   // (batch, head, time2, feat)
        size_t parallel_size
        )
    {
        const auto head_size = dret_tensor.sizes[1];
        const auto time_query = query_tensor.sizes[2];
        const auto feat_size = query_tensor.sizes[3];
        const auto restrict_size = dret_tensor.sizes[3];

        const ptrdiff_t window = (restrict_size - 1) / 2;
        const ptrdiff_t rev_offset = restrict_size - 1 - window;

        // parallel for each (batch, head, time1, feat). sequential for each (restrict)
        PARALLEL_FOR(i, parallel_size) {
            const ptrdiff_t batch_head_index = i / (time_query * feat_size);
            const ptrdiff_t remain = i % (time_query * feat_size);
            const ptrdiff_t time_query_index = remain / feat_size;
            const ptrdiff_t feat_index = remain % feat_size;
            const ptrdiff_t batch_index = batch_head_index / head_size;
            const ptrdiff_t head_index = batch_head_index % head_size;

            const auto* dret_i = dret_tensor.pointer({batch_index, head_index, time_query_index, 0});
            const auto* query_i = query_tensor.pointer({batch_index, head_index, time_query_index, feat_index});
            const auto* key_i = key_tensor.pointer({batch_index, head_index, time_query_index, feat_index});
            scalar_t dquery_i = 0;
            scalar_t dkey_i = 0;

            // TODO parallel reduction
            for (auto w = -min(time_query_index, window); w <= window; ++w) {
                if (key_tensor.sizes[2] <= time_query_index + w) break;
                dquery_i += dret_i[(w + window) * dret_tensor.strides[3]] * key_i[w * key_tensor.strides[2]];
                dkey_i += dret_i[w * dret_tensor.strides[2] + (rev_offset - w) * dret_tensor.strides[3]]
                    * query_i[w * query_tensor.strides[2]];
            }
            *dquery_tensor.pointer({batch_index, head_index, time_query_index, feat_index}) = dquery_i;
            *dkey_tensor.pointer({batch_index, head_index, time_query_index, feat_index}) = dkey_i;
        }
    }

    template <typename scalar_t>
    __global__ void unfold_matmul_cuda_forward_kernel(
        Stride<scalar_t, 4> ret,         // (batch, head, time_query, feat)
        const Stride<scalar_t, 4> score, // (batch, head, time_query, restrict)
        const Stride<scalar_t, 4> value, // (batch, head, time_value, feat)
        const size_t parallel_size
        )
    {
        const auto head_size = score.sizes[1];
        const auto time_query = score.sizes[2];
        const auto restrict_size = score.sizes[3];
        const auto feat_size = value.sizes[3];
        const ptrdiff_t window = (restrict_size - 1) / 2;

        // parallel for (batch, head, time, feat)
        PARALLEL_FOR(i, parallel_size) {
            const ptrdiff_t batch_head_index = i / (time_query * feat_size);
            const ptrdiff_t remain = i % (time_query * feat_size);
            const ptrdiff_t time_query_index = remain / feat_size;
            const ptrdiff_t feat_index = remain % feat_size;
            const ptrdiff_t batch_index = batch_head_index / head_size;
            const ptrdiff_t head_index = batch_head_index % head_size;

            const auto* value_i = value.pointer({batch_index, head_index, time_query_index, feat_index});
            const auto* score_i = score.pointer({batch_index, head_index, time_query_index});

            // TODO parallel reduction
            scalar_t acc = 0;
            for (ptrdiff_t w = -min(time_query_index, window); w <= window; ++w) {
                if (time_query_index + w >= value.sizes[2]) break;

                acc += score_i[(w + window) * score.strides[3]] * value_i[w * value.strides[2]];
            }
            *ret.pointer({batch_index, head_index, time_query_index, feat_index}) = acc;
        }
    }

    template <typename scalar_t>
    __global__ void unfold_matmul_cuda_backward_score_kernel(
        Stride<scalar_t, 4> dscore,      // (batch, head, time_query, restrict)
        const Stride<scalar_t, 4> dret,  // (batch, head, time_query, feat)
        const Stride<scalar_t, 4> value, // (batch, head, time_value, feat)
        const size_t parallel_size
        )
    {
        const auto head_size = dscore.sizes[1];
        const auto time_query = dscore.sizes[2];
        const auto restrict_size = dscore.sizes[3];
        const auto feat_size = value.sizes[3];
        const ptrdiff_t window = (restrict_size - 1) / 2;

        // parallel for batch, head, time_query, restrict
        PARALLEL_FOR(i, parallel_size) {
            const ptrdiff_t batch_head_index = i / (time_query * restrict_size);
            const ptrdiff_t remain = i % (time_query * restrict_size);
            const ptrdiff_t time_query_index = remain / restrict_size;
            const ptrdiff_t restrict_index = remain % restrict_size;
            const ptrdiff_t batch_index = batch_head_index / head_size;
            const ptrdiff_t head_index = batch_head_index % head_size;

            const ptrdiff_t time_value_index = time_query_index + (restrict_index - window);
            if (time_value_index < 0 || value.sizes[2] <=  time_value_index) continue;

            const auto* value_i = value.pointer({batch_index, head_index, time_value_index});
            const auto* dret_i = dret.pointer({batch_index, head_index, time_query_index});

            // TODO parallel reduction
            scalar_t acc = 0;
            for (ptrdiff_t f = 0; f < feat_size; ++f) {
                acc += dret_i[f * dret.strides[3]] * value_i[f * value.strides[3]];
            }
            *dscore.pointer({batch_index, head_index, time_query_index, restrict_index}) = acc;
        }
    }

    template <typename scalar_t>
    __global__ void unfold_matmul_cuda_backward_value_kernel(
        Stride<scalar_t, 4> dvalue,      // (batch, head, time_query, feat)
        const Stride<scalar_t, 4> dret,  // (batch, head, time_query, feat)
        const Stride<scalar_t, 4> score, // (batch, head, time_value, restrict)
        const size_t parallel_size
        )
    {
        const auto head_size = score.sizes[1];
        const auto time_query = score.sizes[2];
        const auto restrict_size = score.sizes[3];
        const auto feat_size = dvalue.sizes[3];
        const ptrdiff_t window = (restrict_size - 1) / 2;

        // parallel for batch, head, time_query, feat
        PARALLEL_FOR(i, parallel_size) {
            const ptrdiff_t batch_head_index = i / (time_query * feat_size);
            const ptrdiff_t remain = i % (time_query * feat_size);
            const ptrdiff_t time_query_index = remain / feat_size;
            const ptrdiff_t feat_index = remain % feat_size;
            const ptrdiff_t batch_index = batch_head_index / head_size;
            const ptrdiff_t head_index = batch_head_index % head_size;

            const auto dret_i = dret.pointer({batch_index, head_index, time_query_index, feat_index});
            const auto score_i = score.pointer({batch_index, head_index, time_query_index, window});

            // TODO parallel reduction
            scalar_t acc = 0;
            for (ptrdiff_t w = -min(time_query_index, window); w <= window; ++w) {
                if (dret.sizes[2] <= time_query_index + w) break;
                acc += dret_i[w * dret.strides[2]] * score_i[w * score.strides[2] - w * score.strides[3]];
            }
            *dvalue.pointer({batch_index, head_index, time_query_index, feat_index}) = acc;
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
    const int blocks = (parallel_size + threads - 1) / threads;

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

    auto dquery = at::empty_like(query);
    auto dkey = at::empty_like(key);
    const size_t parallel_size = batch * head * time1 * feat;
    const int threads = 1024;
    const int blocks = (parallel_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(dret.type(), "unfold_dot_backward_cuda", ([&] {
                unfold_dot_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
                    Stride<scalar_t, 4>(dquery),
                    Stride<scalar_t, 4>(dkey),
                    Stride<scalar_t, 4>(dret),
                    Stride<scalar_t, 4>(query),
                    Stride<scalar_t, 4>(key),
                    parallel_size);
            }));

    return {dquery, dkey};
}


at::Tensor unfold_matmul_cuda_forward(
    at::Tensor score,           // (batch, head, time, restrict)
    at::Tensor value            // (batch, head, time, feat)
    )
{
    // NOTE: reason of this transpose is the output `ret` will be transposed after this op
    // TODO: add option like `bool transpose` or optional value `at::Tensor ret`
    // (batch, time, head, feat) -> (batch, head, time, feat)
    auto batch = value.size(0);
    auto head = value.size(1);
    auto time = value.size(2);
    auto feat = value.size(3);
    auto ret = at::empty({batch, time, head, feat}, value.options());
    ret = ret.transpose(1, 2);

    const size_t parallel_size = ret.size(0) * ret.size(1) * ret.size(2) * ret.size(3);
    const int threads = 1024;
    const int blocks = (parallel_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(ret.type(), "unfold_matmul_forward_cuda", ([&] {
                unfold_matmul_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
                    Stride<scalar_t, 4>(ret),
                    Stride<scalar_t, 4>(score),
                    Stride<scalar_t, 4>(value),
                    parallel_size);
            }));

    return ret;
}


std::array<at::Tensor, 2> unfold_matmul_cuda_backward(
    at::Tensor dret,            // (batch, head, time, feat)
    at::Tensor score,           // (batch, head, time, restrict)
    at::Tensor value            // (batch, head, time, feat)
    )
{
    auto dscore = at::zeros_like(score);
    auto dvalue = at::zeros_like(value);
    {
        // batch x head x time x restrict
        const size_t parallel_size = dscore.size(0) * dscore.size(1) * dscore.size(2) * dscore.size(3);
        const int threads = 1024;
        const int blocks = (parallel_size + threads - 1) / threads;

        AT_DISPATCH_FLOATING_TYPES(dscore.type(), "unfold_matmul_backward_score_cuda", ([&] {
                    unfold_matmul_cuda_backward_score_kernel<scalar_t><<<blocks, threads>>>(
                        Stride<scalar_t, 4>(dscore),
                        Stride<scalar_t, 4>(dret),
                        Stride<scalar_t, 4>(value),
                        parallel_size);
                }));
    }
    {
        // batch x head x time x feat
        const size_t parallel_size = dvalue.size(0) * dvalue.size(1) * dvalue.size(2) * dvalue.size(3);
        const int threads = 1024;
        const int blocks = (parallel_size + threads - 1) / threads;

        AT_DISPATCH_FLOATING_TYPES(dvalue.type(), "unfold_matmul_backward_value_cuda", ([&] {
                    unfold_matmul_cuda_backward_value_kernel<scalar_t><<<blocks, threads>>>(
                        Stride<scalar_t, 4>(dvalue),
                        Stride<scalar_t, 4>(dret),
                        Stride<scalar_t, 4>(score),
                        parallel_size);
                }));
    }
    return {dscore, dvalue};
}
