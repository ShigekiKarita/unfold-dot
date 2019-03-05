#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {
    template <typename scalar_t>
    __device__ __forceinline__ scalar_t sigmoid(scalar_t z) {
        return 1.0 / (1.0 + exp(-z));
    }

    template <typename scalar_t>
    __device__ __forceinline__ scalar_t d_sigmoid(scalar_t z) {
        const auto s = sigmoid(z);
        return (1.0 - s) * s;
    }

    template <typename scalar_t>
    __device__ __forceinline__ scalar_t d_tanh(scalar_t z) {
        const auto t = tanh(z);
        return 1 - (t * t);
    }

    template <typename scalar_t>
    __device__ __forceinline__ scalar_t elu(scalar_t z, scalar_t alpha = 1.0) {
        return fmax(scalar_t{0.0}, z) + fmin(0.0, alpha * (exp(z) - 1.0));
    }

    template <typename scalar_t>
    __device__ __forceinline__ scalar_t d_elu(scalar_t z, scalar_t alpha = 1.0) {
        const auto e = exp(z);
        const auto d_relu = z < 0.0 ? 0.0 : 1.0;
        return d_relu + (((alpha * (e - 1.0)) < 0.0) ? (alpha * e) : 0.0);
    }

    template <typename scalar_t>
    __global__ void lltm_cuda_forward_kernel(
        const scalar_t* __restrict__ gates,
        const scalar_t* __restrict__ old_cell,
        scalar_t* __restrict__ new_h,
        scalar_t* __restrict__ new_cell,
        scalar_t* __restrict__ input_gate,
        scalar_t* __restrict__ output_gate,
        scalar_t* __restrict__ candidate_cell,
        size_t state_size) {
        const int column = blockIdx.x * blockDim.x + threadIdx.x;
        const int index = blockIdx.y * state_size + column;
        const int gates_row = blockIdx.y * (state_size * 3);
        if (column < state_size) {
            input_gate[index] = sigmoid(gates[gates_row + column]);
            output_gate[index] = sigmoid(gates[gates_row + state_size + column]);
            candidate_cell[index] = elu(gates[gates_row + 2 * state_size + column]);
            new_cell[index] =
                old_cell[index] + candidate_cell[index] * input_gate[index];
            new_h[index] = tanh(new_cell[index]) * output_gate[index];
        }
    }

    template <typename scalar_t>
    __global__ void lltm_cuda_backward_kernel(
        scalar_t* __restrict__ d_old_cell,
        scalar_t* __restrict__ d_gates,
        const scalar_t* __restrict__ grad_h,
        const scalar_t* __restrict__ grad_cell,
        const scalar_t* __restrict__ new_cell,
        const scalar_t* __restrict__ input_gate,
        const scalar_t* __restrict__ output_gate,
        const scalar_t* __restrict__ candidate_cell,
        const scalar_t* __restrict__ gate_weights,
        size_t state_size) {
        const int column = blockIdx.x * blockDim.x + threadIdx.x;
        const int index = blockIdx.y * state_size + column;
        const int gates_row = blockIdx.y * (state_size * 3);
        if (column < state_size) {
            const auto d_output_gate = tanh(new_cell[index]) * grad_h[index];
            const auto d_tanh_new_cell = output_gate[index] * grad_h[index];
            const auto d_new_cell =
                d_tanh(new_cell[index]) * d_tanh_new_cell + grad_cell[index];


            d_old_cell[index] = d_new_cell;
            const auto d_candidate_cell = input_gate[index] * d_new_cell;
            const auto d_input_gate = candidate_cell[index] * d_new_cell;


            const auto input_gate_index = gates_row + column;
            const auto output_gate_index = gates_row + state_size + column;
            const auto candidate_cell_index = gates_row + 2 * state_size + column;

            d_gates[input_gate_index] =
                d_input_gate * d_sigmoid(gate_weights[input_gate_index]);
            d_gates[output_gate_index] =
                d_output_gate * d_sigmoid(gate_weights[output_gate_index]);
            d_gates[candidate_cell_index] =
                d_candidate_cell * d_elu(gate_weights[candidate_cell_index]);
        }
    }

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


std::vector<at::Tensor> lltm_cuda_forward(
    at::Tensor input,
    at::Tensor weights,
    at::Tensor bias,
    at::Tensor old_h,
    at::Tensor old_cell) {
    auto X = at::cat({old_h, input}, /*dim=*/1);
    auto gates = at::addmm(bias, X, weights.transpose(0, 1));

    const auto batch_size = old_cell.size(0);
    const auto state_size = old_cell.size(1);

    auto new_h = at::zeros_like(old_cell);
    auto new_cell = at::zeros_like(old_cell);
    auto input_gate = at::zeros_like(old_cell);
    auto output_gate = at::zeros_like(old_cell);
    auto candidate_cell = at::zeros_like(old_cell);

    const int threads = 1024;
    const dim3 blocks((state_size + threads - 1) / threads, batch_size);

    AT_DISPATCH_FLOATING_TYPES(gates.type(), "lltm_forward_cuda", ([&] {
                lltm_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
                    gates.data<scalar_t>(),
                    old_cell.data<scalar_t>(),
                    new_h.data<scalar_t>(),
                    new_cell.data<scalar_t>(),
                    input_gate.data<scalar_t>(),
                    output_gate.data<scalar_t>(),
                    candidate_cell.data<scalar_t>(),
                    state_size);
            }));

    return {new_h, new_cell, input_gate, output_gate, candidate_cell, X, gates};
}

std::vector<at::Tensor> lltm_cuda_backward(
    at::Tensor grad_h,
    at::Tensor grad_cell,
    at::Tensor new_cell,
    at::Tensor input_gate,
    at::Tensor output_gate,
    at::Tensor candidate_cell,
    at::Tensor X,
    at::Tensor gate_weights,
    at::Tensor weights) {
    auto d_old_cell = at::zeros_like(new_cell);
    auto d_gates = at::zeros_like(gate_weights);

    const auto batch_size = new_cell.size(0);
    const auto state_size = new_cell.size(1);

    const int threads = 1024;
    const dim3 blocks((state_size + threads - 1) / threads, batch_size);

    AT_DISPATCH_FLOATING_TYPES(X.type(), "lltm_forward_cuda", ([&] {
                lltm_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
                    d_old_cell.data<scalar_t>(),
                    d_gates.data<scalar_t>(),
                    grad_h.data<scalar_t>(),
                    grad_cell.data<scalar_t>(),
                    new_cell.data<scalar_t>(),
                    input_gate.data<scalar_t>(),
                    output_gate.data<scalar_t>(),
                    candidate_cell.data<scalar_t>(),
                    gate_weights.data<scalar_t>(),
                    state_size);
            }));

    auto d_weights = d_gates.t().mm(X);
    auto d_bias = d_gates.sum(/*dim=*/0, /*keepdim=*/true);

    auto d_X = d_gates.mm(weights);
    auto d_old_h = d_X.slice(/*dim=*/1, 0, state_size);
    auto d_input = d_X.slice(/*dim=*/1, state_size);

    return {d_old_h, d_input, d_weights, d_bias, d_old_cell, d_gates};
}
