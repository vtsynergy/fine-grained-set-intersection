/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/** ---------------------------------------------------------------------------*
 * @brief The cugraph Jaccard core functionality
 *
 * @file jaccard.cu
 * ---------------------------------------------------------------------------**/


#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <iostream>
#ifndef STANDALONE
  #include "graph.hpp"
  #include "utilities/graph_utils.cuh"
  #include <rmm/thrust_rmm_allocator.h>
  #include <utilities/error.hpp>
#else
  #include "standalone_algorithms.hpp"
  #include "standalone_csr.hpp"
  #include <chrono>
  #include <iostream>
  #define EC_MAX_THREADS_PER_BLOCK 512
// from chpltypes.h
#define LINEAR_SEARCH 1
typedef double _real64;

  #define chpl_seconds_timer(time) ((_real64)((time).tv_sec))

  #define chpl_microseconds_timer(time) ((_real64)((time).tv_usec))

#ifndef EC_MAX_THREADS_PER_BLOCK
#define EC_MAX_THREADS_PER_BLOCK 512
#endif

#define BIN_KERNEL_1D 1
#define BIN_KERNEL_2D 2

// current timer, in fractional seconds
double gettimer(struct timeval *timer) {
  int dummy = gettimeofday(timer, NULL); // ignore the return code
  return chpl_seconds_timer(*timer) + 1.0e-6 * chpl_microseconds_timer(*timer);
}
// From RAFT at commit 48063dc08
__host__ __device__ constexpr inline int warp_size() {
  return 32;
}

__host__ __device__ constexpr inline unsigned int warp_full_mask() {
  return 0xffffffff;
}
// From utilties/graph_utils.cuh
template <typename count_t, typename index_t, typename value_t>
__inline__ __device__ value_t parallel_prefix_sum(count_t n, index_t const *ind, value_t const *w) {
  count_t i, j, mn;
  value_t v, last;
  value_t sum = 0.0;
  bool valid;

  // Parallel prefix sum (using __shfl)
  mn = (((n + blockDim.x - 1) / blockDim.x) * blockDim.x); // n in multiple of blockDim.x
  for (i = threadIdx.x; i < mn; i += blockDim.x) {
    // All threads (especially the last one) must always participate
    // in the shfl instruction, otherwise their sum will be undefined.
    // So, the loop stopping condition is based on multiple of n in loop increments,
    // so that all threads enter into the loop and inside we make sure we do not
    // read out of bounds memory checking for the actual size n.

    // check if the thread is valid
    valid = i < n;

    // Notice that the last thread is used to propagate the prefix sum.
    // For all the threads, in the first iteration the last is 0, in the following
    // iterations it is the value at the last thread of the previous iterations.

    // get the value of the last thread
    last = __shfl_sync(warp_full_mask(), sum, blockDim.x - 1, blockDim.x);

    // if you are valid read the value from memory, otherwise set your value to 0
    sum = (valid) ? w[ind[i]] : 0.0;

    // do prefix sum (of size warpSize=blockDim.x =< 32)
    for (j = 1; j < blockDim.x; j *= 2) {
      v = __shfl_up_sync(warp_full_mask(), sum, j, blockDim.x);
      if (threadIdx.x >= j) sum += v;
    }
    // shift by last
    sum += last;
    // notice that no __threadfence or __syncthreads are needed in this implementation
  }
  // get the value of the last thread (to all threads)
  last = __shfl_sync(warp_full_mask(), sum, blockDim.x - 1, blockDim.x);

  return last;
}

// Custom Thrust simplifications
template <typename T>
void __global__ fill_kernel(T *ptr, T value, size_t n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int incr = blockDim.x * gridDim.x;
  for (; idx < n; idx += incr) {
    ptr[idx] = value;
  }
}

template <typename T>
void fill(size_t n, T *x, T value) {
  size_t block = std::min((size_t)n, (size_t)CUDA_MAX_KERNEL_THREADS);
  size_t grid = std::min((size_t)(n / block) + ((n % block) ? 1 : 0), (size_t)CUDA_MAX_BLOCKS);
  // TODO, do we need to emulate their stream behavior?
  struct timeval kernel_time;
  double fill_start = gettimer(&kernel_time);
  fill_kernel<<<grid, block>>>(x, value, n);
  cudaError_t error = cudaDeviceSynchronize();
  double fill_stop = gettimer(&kernel_time);
 // std::cout << "VC_Fill Elapsed (s): " << fill_stop - fill_start << std::endl;
  if (error != cudaSuccess) std::cerr << "Error in fill_kernel " << error << std::endl;
}
  // Directly from the CUDA programming guide
  #if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
  #else
__device__ double atomicAdd(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);

  return __longlong_as_double(old);
}
  #endif

#endif

namespace cugraph {
namespace detail {


// Volume of neighboors (*weight_s)
template <bool weighted, typename vertex_t, typename edge_t, typename weight_t>
__global__ void jaccard_row_sum(vertex_t n, edge_t const *csrPtr, vertex_t const *csrInd,
                                weight_t const *v, weight_t *work) {
  vertex_t row;
  edge_t start, end, length;
  weight_t sum;

  for (row = threadIdx.y + blockIdx.y * blockDim.y; row < n; row += gridDim.y * blockDim.y) {
    start = csrPtr[row];
    end = csrPtr[row + 1];
    length = end - start;

    // compute row sums
    if constexpr (weighted) {
      sum = parallel_prefix_sum(length, csrInd + start, v);
      if (threadIdx.x == 0) work[row] = sum;
    } else {
      work[row] = static_cast<weight_t>(length);
    }
  }
}

template <typename vertex_t, typename edge_t, typename weight_t>
__global__ void jaccard_ec_scan(vertex_t n, edge_t const *csrPtr, vertex_t const *csrInd,
                                vertex_t *dest_ind) {
  edge_t tid, i;
  tid = blockIdx.x * EC_MAX_THREADS_PER_BLOCK + threadIdx.x;
  if (tid < n) {

    // Ni=csrPtr[tid+1]-csrPtr[tid];
    for (i = csrPtr[tid]; i < csrPtr[tid + 1]; i++) {
      dest_ind[i] = tid;
    }
  }
}


template <typename edge_t>
__global__ void init_work_ids(edge_t nitems, edge_t *ids) {
  edge_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < nitems) {
    ids[tid] = tid;
  }
}

template <typename vertex_t, typename edge_t>
__global__ void compute_edge_costs_lists(edge_t nitems,
                                         vertex_t const *src_list,
                                         vertex_t const *dst_list,
                                         edge_t const *csrPtr,
                                         unsigned int *cost_out,
                                         unsigned int *max_cost_out) {
  edge_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nitems) return;

  vertex_t src = src_list[tid];
  vertex_t dst = dst_list[tid];

  edge_t Ni = csrPtr[src + 1] - csrPtr[src];
  edge_t Nj = csrPtr[dst + 1] - csrPtr[dst];

  edge_t refLen = (Ni < Nj) ? Ni : Nj;
  edge_t curLen = (Ni < Nj) ? Nj : Ni;

  unsigned int cost = 0;

  if (refLen != 0 && curLen != 0) {
    edge_t curDepth = 0;
    if constexpr (sizeof(edge_t) == 4) {
      curDepth = 32 - __clz(static_cast<unsigned int>(curLen));
    } else {
      curDepth = 64 - __clzll(static_cast<unsigned long long>(curLen));
    }

    edge_t linearCost = refLen;
    edge_t binaryCost = refLen * curDepth;
    edge_t bestCost = (linearCost < binaryCost) ? linearCost : binaryCost;

    cost = static_cast<unsigned int>(bestCost);
  }

  cost_out[tid] = cost;
  atomicMax(max_cost_out, cost);
}

template <typename vertex_t, typename edge_t, typename weight_t>
__global__ void set_intersection_ec_lists(edge_t num_work_items,
                                          edge_t const *work_ids,
                                          vertex_t const *src_list,
                                          vertex_t const *dst_list,
                                          edge_t const *csrPtr,
                                          vertex_t const *csrInd,
                                          weight_t *intersection_count) {
  edge_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_work_items) return;

  edge_t wid = work_ids[tid];

  vertex_t src = src_list[wid];
  vertex_t dst = dst_list[wid];

  edge_t Ni = csrPtr[src + 1] - csrPtr[src];
  edge_t Nj = csrPtr[dst + 1] - csrPtr[dst];

  vertex_t ref = (Ni < Nj) ? src : dst;
  vertex_t cur = (Ni < Nj) ? dst : src;

  if (Ni == 0 || Nj == 0) {
    intersection_count[wid] = 0;
    return;
  }

  {
    vertex_t ref_first = csrInd[csrPtr[ref]];
    vertex_t ref_last  = csrInd[csrPtr[ref + 1] - 1];
    vertex_t cur_first = csrInd[csrPtr[cur]];
    vertex_t cur_last  = csrInd[csrPtr[cur + 1] - 1];

    if (ref_first > cur_last || cur_first > ref_last) {
      intersection_count[wid] = 0;
      return;
    }
  }

  weight_t intersections = 0;

#ifdef LINEAR_SEARCH
  edge_t refLen = (Ni < Nj) ? Ni : Nj;
  edge_t curLen = (Ni < Nj) ? Nj : Ni;

  edge_t curDepth = 0;
  if constexpr (sizeof(edge_t) == 4) {
    curDepth = 32 - __clz(static_cast<unsigned int>(curLen));
  } else {
    curDepth = 64 - __clzll(static_cast<unsigned long long>(curLen));
  }

  bool useLinear = ((curLen + refLen) < (refLen * curDepth));

  if (!useLinear) {
#endif
    for (edge_t i = csrPtr[ref]; i < csrPtr[ref + 1]; i++) {
      vertex_t ref_col = csrInd[i];

      edge_t left  = csrPtr[cur];
      edge_t right = csrPtr[cur + 1] - 1;

      while (left <= right) {
        edge_t middle = static_cast<edge_t>((static_cast<int64_t>(left) + right) >> 1);
        vertex_t cur_col = csrInd[middle];

        if (cur_col > ref_col) {
          right = middle - 1;
        } else if (cur_col < ref_col) {
          left = middle + 1;
        } else {
          intersections += 1;
          break;
        }
      }
    }
#ifdef LINEAR_SEARCH
  } else {
    edge_t ref_idx = csrPtr[ref];
    edge_t ref_end = csrPtr[ref + 1] - 1;
    edge_t cur_idx = csrPtr[cur];
    edge_t cur_end = csrPtr[cur + 1] - 1;

    while (ref_idx <= ref_end && cur_idx <= cur_end) {
      vertex_t ref_col = csrInd[ref_idx];
      vertex_t cur_col = csrInd[cur_idx];

      if (ref_col == cur_col) {
        intersections += 1;
        ref_idx++;
        cur_idx++;
      } else if (cur_col > ref_col) {
        ref_idx++;
      } else {
        cur_idx++;
      }
    }
  }
#endif

  intersection_count[wid] = intersections;
}

template <typename vertex_t, typename edge_t, typename weight_t>
__global__ void set_intersection_lb_pairs_lists(edge_t num_work_items,
                                                edge_t const *work_ids,
                                                vertex_t const *src_list,
                                                vertex_t const *dst_list,
                                                edge_t const *csrPtr,
                                                vertex_t const *csrInd,
                                                weight_t *intersection_count) {
  for (edge_t p = threadIdx.z + blockIdx.z * blockDim.z;
       p < num_work_items;
       p += gridDim.z * blockDim.z) {

    edge_t wid = work_ids[p];

    vertex_t src = src_list[wid];
    vertex_t dst = dst_list[wid];

    edge_t Ni = csrPtr[src + 1] - csrPtr[src];
    edge_t Nj = csrPtr[dst + 1] - csrPtr[dst];

    vertex_t ref = (Ni < Nj) ? src : dst;
    vertex_t cur = (Ni < Nj) ? dst : src;

    if (Ni == 0 || Nj == 0) continue;

    {
      vertex_t ref_first = csrInd[csrPtr[ref]];
      vertex_t ref_last  = csrInd[csrPtr[ref + 1] - 1];
      vertex_t cur_first = csrInd[csrPtr[cur]];
      vertex_t cur_last  = csrInd[csrPtr[cur + 1] - 1];

      if (ref_first > cur_last || cur_first > ref_last) continue;
    }

    for (edge_t i = csrPtr[ref] + threadIdx.x + blockIdx.x * blockDim.x;
         i < csrPtr[ref + 1];
         i += gridDim.x * blockDim.x) {

      vertex_t ref_col = csrInd[i];

      edge_t left = csrPtr[cur];
      edge_t right = csrPtr[cur + 1] - 1;

      while (left <= right) {
        edge_t middle = static_cast<edge_t>((static_cast<int64_t>(left) + right) >> 1);
        vertex_t cur_col = csrInd[middle];

        if (cur_col > ref_col) {
          right = middle - 1;
        } else if (cur_col < ref_col) {
          left = middle + 1;
        } else {
          atomicAdd(&intersection_count[wid], static_cast<weight_t>(1));
          break;
        }
      }
    }
  }
}


// Volume of intersections (*weight_i) and cumulated volume of neighboors (*weight_s)
template <bool weighted, typename vertex_t, typename edge_t, typename weight_t>
__global__ void jaccard_is(vertex_t n, edge_t const *csrPtr, vertex_t const *csrInd,
                           weight_t const *v, weight_t *work, weight_t *weight_i,
                           weight_t *weight_s) {
  edge_t i, j, Ni, Nj;
  vertex_t row, col;
  vertex_t ref, cur, ref_col, cur_col, match;
  weight_t ref_val;

  for (row = threadIdx.z + blockIdx.z * blockDim.z; row < n; row += gridDim.z * blockDim.z) {
    for (j = csrPtr[row] + threadIdx.y + blockIdx.y * blockDim.y; j < csrPtr[row + 1];
         j += gridDim.y * blockDim.y) {
      col = csrInd[j];
      // find which row has least elements (and call it reference row)
      Ni = csrPtr[row + 1] - csrPtr[row];
      Nj = csrPtr[col + 1] - csrPtr[col];
      ref = (Ni < Nj) ? row : col;
      cur = (Ni < Nj) ? col : row;
#ifdef LINEAR_SEARCH
      edge_t curDepth;
      edge_t refLen = (Ni < Nj) ? Ni : Nj;
      edge_t curLen = (Ni < Nj) ? Nj : Ni;
      if constexpr (sizeof(edge_t) == 4) {
        curDepth = 32 - __clz(curLen); // Fast Int Log2
      } else {
        curDepth = 64 - __clzll(curLen);
      }
      bool useLinear = ((curLen + (refLen / (gridDim.x * blockDim.x))) <
                        ((refLen / (gridDim.x * blockDim.x)) * curDepth));
      // Cur is now fixed, and we want to retain progress inside the linear search across for i
      // iterations
      edge_t cur_idx = csrPtr[cur], cur_end = csrPtr[cur + 1] - 1;
#endif

      // compute new sum weights
      weight_s[j] = work[row] + work[col];

      // compute new intersection weights
      // search for the element with the same column index in the reference row
      for (i = csrPtr[ref] + threadIdx.x + blockIdx.x * blockDim.x; i < csrPtr[ref + 1];
           i += gridDim.x * blockDim.x) {
        match = -1;
        ref_col = csrInd[i];
        if constexpr (weighted) {
          ref_val = v[ref_col];
        } else {
          ref_val = 1.0;
        }

#ifdef LINEAR_SEARCH
        if (!useLinear) {
#endif
          // binary search (column indices are sorted within each row)
          edge_t left = csrPtr[cur];
          edge_t right = csrPtr[cur + 1] - 1;
          while (left <= right) {
            edge_t middle = static_cast<edge_t>((static_cast<int64_t>(left) + right) >> 1);
            cur_col = csrInd[middle];
            if (cur_col > ref_col) {
              right = middle - 1;
            } else if (cur_col < ref_col) {
              left = middle + 1;
            } else {
              match = middle;
              break;
            }
          }
#ifdef LINEAR_SEARCH
        } else {
          while (cur_idx <= cur_end) {
            cur_col = csrInd[cur_idx];
            if (ref_col == cur_col) {
              match = cur_idx;
              cur_idx++; // Advance the comparison, since this thread's next value necessarily must
                         // be > ref_col
              break;
            } else if (cur_col > ref_col) {
              break; // neighbor lists are sorted, abort early, but do not advance the comparison,
                     // since this thread's next value may be < cur_col
            } else {
              cur_idx++;
            }
          }
        }
#endif // LINEAR_SEARCH
       // if the element with the same column index in the reference row has been found
        if (match != -1) {
          atomicAdd(&weight_i[j], ref_val);
        }
      }
    }
  }
}

// Volume of intersections (*weight_i) and cumulated volume of neighboors (*weight_s)
// Using list of node pairs
template <bool weighted, typename vertex_t, typename edge_t, typename weight_t>
__global__ void jaccard_is_pairs(edge_t num_pairs, edge_t const *csrPtr, vertex_t const *csrInd,
                                 vertex_t const *first_pair, vertex_t const *second_pair,
                                 weight_t const *v, weight_t *work, weight_t *weight_i,
                                 weight_t *weight_s) {
  edge_t i, idx, Ni, Nj, match;
  vertex_t row, col, ref, cur, ref_col, cur_col;
  weight_t ref_val;

  for (idx = threadIdx.z + blockIdx.z * blockDim.z; idx < num_pairs;
       idx += gridDim.z * blockDim.z) {
    row = first_pair[idx];
    col = second_pair[idx];

    // find which row has least elements (and call it reference row)
    Ni = csrPtr[row + 1] - csrPtr[row];
    Nj = csrPtr[col + 1] - csrPtr[col];
    ref = (Ni < Nj) ? row : col;
    cur = (Ni < Nj) ? col : row;

    // compute new sum weights
    weight_s[idx] = work[row] + work[col];

    // compute new intersection weights
    // search for the element with the same column index in the reference row
    for (i = csrPtr[ref] + threadIdx.x + blockIdx.x * blockDim.x; i < csrPtr[ref + 1];
         i += gridDim.x * blockDim.x) {
      match = -1;
      ref_col = csrInd[i];
      if constexpr (weighted) {
        ref_val = v[ref_col];
      } else {
        ref_val = 1.0;
      }

      // binary search (column indices are sorted within each row)
      edge_t left = csrPtr[cur];
      edge_t right = csrPtr[cur + 1] - 1;
      while (left <= right) {
        edge_t middle = (left + right) >> 1;
        cur_col = csrInd[middle];
        if (cur_col > ref_col) {
          right = middle - 1;
        } else if (cur_col < ref_col) {
          left = middle + 1;
        } else {
          match = middle;
          break;
        }
      }

      // if the element with the same column index in the reference row has been found
      if (match != -1) {
        atomicAdd(&weight_i[idx], ref_val);
      }
    }
  }
}

// Jaccard  weights (*weight)
template <bool weighted, typename vertex_t, typename edge_t, typename weight_t>
__global__ void jaccard_jw(edge_t e, weight_t const *weight_i, weight_t const *weight_s,
                           weight_t *weight_j) {
  edge_t j;
  weight_t Wi, Ws, Wu;

  for (j = threadIdx.x + blockIdx.x * blockDim.x; j < e; j += gridDim.x * blockDim.x) {
    Wi = weight_i[j];
    Ws = weight_s[j];
    Wu = Ws - Wi;
    weight_j[j] = (Wi / Wu);
  }
}

template <bool edge_centric, bool weighted, typename vertex_t, typename edge_t, typename weight_t>
int jaccard(vertex_t n, edge_t e, edge_t const *csrPtr, vertex_t const *csrInd,
            weight_t const *weight_in, weight_t *weight_j) {
  struct timeval kernel_time;
  if constexpr (edge_centric) {
  vertex_t *dest_ind = nullptr;
  cudaError_t error = cudaSuccess;

  error = cudaMalloc(&dest_ind, static_cast<int64_t>(e) * sizeof(vertex_t));
  if (error != cudaSuccess) {
    std::cerr << "CUDA ERROR in cudaMalloc(dest_ind): " << error << std::endl;
    error = cudaSuccess;
  }

  int num_of_blocks = 1;
  int num_of_threads_per_block = n;

  if (n > EC_MAX_THREADS_PER_BLOCK) {
    num_of_blocks = (int)ceil(n / (double)EC_MAX_THREADS_PER_BLOCK);
    num_of_threads_per_block = EC_MAX_THREADS_PER_BLOCK;
  }

  jaccard_ec_scan<vertex_t, edge_t, weight_t>
      <<<num_of_blocks, num_of_threads_per_block>>>(n, csrPtr, csrInd, dest_ind);
  cudaDeviceSynchronize();

  // ----------------------------------------------------------
  // build explicit source < destination lists on host
  // NOT included in timing
  // ----------------------------------------------------------
  vertex_t *h_dest_ind = (vertex_t *)malloc(sizeof(vertex_t) * static_cast<int64_t>(e));
  vertex_t *h_csrInd   = (vertex_t *)malloc(sizeof(vertex_t) * static_cast<int64_t>(e));

  cudaMemcpy(h_dest_ind,
             dest_ind,
             sizeof(vertex_t) * static_cast<int64_t>(e),
             cudaMemcpyDeviceToHost);

  cudaMemcpy(h_csrInd,
             csrInd,
             sizeof(vertex_t) * static_cast<int64_t>(e),
             cudaMemcpyDeviceToHost);

  std::vector<vertex_t> h_src_list_vec;
  std::vector<vertex_t> h_dst_list_vec;

  h_src_list_vec.reserve(static_cast<size_t>(e / 2));
  h_dst_list_vec.reserve(static_cast<size_t>(e / 2));

  for (edge_t k = 0; k < e; k++) {
    vertex_t src = h_dest_ind[k];
    vertex_t dst = h_csrInd[k];

    if (src < dst) {
      h_src_list_vec.push_back(src);
      h_dst_list_vec.push_back(dst);
    }
  }

  edge_t h_num_half_edges = static_cast<edge_t>(h_src_list_vec.size());

  vertex_t *d_src_list = nullptr;
  vertex_t *d_dst_list = nullptr;

  cudaMalloc((void **)&d_src_list, sizeof(vertex_t) * h_num_half_edges);
  cudaMalloc((void **)&d_dst_list, sizeof(vertex_t) * h_num_half_edges);

  cudaMemcpy(d_src_list,
             h_src_list_vec.data(),
             sizeof(vertex_t) * h_num_half_edges,
             cudaMemcpyHostToDevice);

  cudaMemcpy(d_dst_list,
             h_dst_list_vec.data(),
             sizeof(vertex_t) * h_num_half_edges,
             cudaMemcpyHostToDevice);

  free(h_dest_ind);
  free(h_csrInd);

  // ----------------------------------------------------------
  // allocate sort inputs/outputs
  // ----------------------------------------------------------
  unsigned int *d_cost_in  = nullptr;
  unsigned int *d_cost_out = nullptr;
  unsigned int *d_max_cost = nullptr;

  edge_t *d_work_ids_in  = nullptr;
  edge_t *d_work_ids_out = nullptr;

  cudaMalloc((void **)&d_cost_in, sizeof(unsigned int) * h_num_half_edges);
  cudaMalloc((void **)&d_cost_out, sizeof(unsigned int) * h_num_half_edges);
  cudaMalloc((void **)&d_max_cost, sizeof(unsigned int));
  cudaMemset(d_max_cost, 0, sizeof(unsigned int));

  cudaMalloc((void **)&d_work_ids_in, sizeof(edge_t) * h_num_half_edges);
  cudaMalloc((void **)&d_work_ids_out, sizeof(edge_t) * h_num_half_edges);

  // ----------------------------------------------------------
  // compute costs + initialize work ids on GPU
  // ----------------------------------------------------------

  double sortprep_start = gettimer(&kernel_time);
  {
    int threads = 256;
    int blocks  = (int)((h_num_half_edges + threads - 1) / threads);

    init_work_ids<edge_t><<<blocks, threads>>>(h_num_half_edges, d_work_ids_in);

    compute_edge_costs_lists<vertex_t, edge_t>
        <<<blocks, threads>>>(
            h_num_half_edges,
            d_src_list,
            d_dst_list,
            csrPtr,
            d_cost_in,
            d_max_cost);

    cudaDeviceSynchronize();
  }

  double sortprep_cost_stop = gettimer(&kernel_time);

  unsigned int max_cost = 0;
  cudaMemcpy(&max_cost,
             d_max_cost,
             sizeof(unsigned int),
             cudaMemcpyDeviceToHost);

  int sort_end_bit = 1;
  if (max_cost != 0U) {
    sort_end_bit = 32 - __builtin_clz(max_cost);
  }

  // ----------------------------------------------------------
  // CUB sort pairs: (cost, work_id), ascending by cost
  // ----------------------------------------------------------
  void *d_sort_temp_storage = nullptr;
  size_t sort_temp_storage_bytes = 0;

  double sortprep_sort_start = gettimer(&kernel_time);
  cub::DeviceRadixSort::SortPairs(
      d_sort_temp_storage,
      sort_temp_storage_bytes,
      d_cost_in,
      d_cost_out,
      d_work_ids_in,
      d_work_ids_out,
      h_num_half_edges,
      0,
      sort_end_bit);

  cudaMalloc(&d_sort_temp_storage, sort_temp_storage_bytes);
  cub::DeviceRadixSort::SortPairs(
      d_sort_temp_storage,
      sort_temp_storage_bytes,
      d_cost_in,
      d_cost_out,
      d_work_ids_in,
      d_work_ids_out,
      h_num_half_edges,
      0,
      sort_end_bit);

  cudaDeviceSynchronize();

  double sortprep_stop = gettimer(&kernel_time);
  double sort_overhead = (sortprep_cost_stop - sortprep_start) +
                         (sortprep_stop - sortprep_sort_start);

  // ----------------------------------------------------------
  // prefix scan on sorted costs
  // ----------------------------------------------------------
  unsigned int *d_prefix_cost = nullptr;
  cudaMalloc((void **)&d_prefix_cost, sizeof(unsigned int) * h_num_half_edges);

  void *d_scan_temp_storage = nullptr;
  size_t scan_temp_storage_bytes = 0;

  double prefix_start = gettimer(&kernel_time);

  cub::DeviceScan::InclusiveSum(
      d_scan_temp_storage,
      scan_temp_storage_bytes,
      d_cost_out,
      d_prefix_cost,
      h_num_half_edges);

  cudaMalloc(&d_scan_temp_storage, scan_temp_storage_bytes);

  cub::DeviceScan::InclusiveSum(
      d_scan_temp_storage,
      scan_temp_storage_bytes,
      d_cost_out,
      d_prefix_cost,
      h_num_half_edges);

  cudaDeviceSynchronize();

  double prefix_stop = gettimer(&kernel_time);
  double prefix_overhead = prefix_stop - prefix_start;

  // ----------------------------------------------------------
  // copy sorted costs + prefix sums to host
  // ----------------------------------------------------------
  unsigned int *h_cost_sorted =
      (unsigned int *)malloc(sizeof(unsigned int) * h_num_half_edges);

  unsigned int *h_prefix_cost =
      (unsigned int *)malloc(sizeof(unsigned int) * h_num_half_edges);

  cudaMemcpy(h_cost_sorted,
             d_cost_out,
             sizeof(unsigned int) * h_num_half_edges,
             cudaMemcpyDeviceToHost);

  cudaMemcpy(h_prefix_cost,
             d_prefix_cost,
             sizeof(unsigned int) * h_num_half_edges,
             cudaMemcpyDeviceToHost);

  // ----------------------------------------------------------
  // choose split so cumulative cost is closest to half total
  // using host-side binary search
  // TODO: Push the version where this is offloaded to GPU.
  // ----------------------------------------------------------
  edge_t e1d = 0;
  edge_t e2d = h_num_half_edges;
  double split_fraction_1d = 0.0;
  unsigned int transition_cost = 0;
  unsigned int total_estimated_cost = 0;
  unsigned int target_half_cost = 0;

  if (h_num_half_edges > 0) {
    total_estimated_cost = h_prefix_cost[h_num_half_edges - 1];
    target_half_cost = total_estimated_cost / 2U;

    if (total_estimated_cost == 0U) {
      e1d = h_num_half_edges / 2;
    } else {
      edge_t low = 0;
      edge_t high = h_num_half_edges - 1;
      edge_t pos = h_num_half_edges - 1;

      while (low <= high) {
        edge_t mid = low + (high - low) / 2;

        if (h_prefix_cost[mid] >= target_half_cost) {
          pos = mid;
          if (mid == 0) {
            break;
          }
          high = mid - 1;
        } else {
          low = mid + 1;
        }
      }

      edge_t best_idx = pos;

      if (pos > 0) {
        unsigned int diff_pos =
            (h_prefix_cost[pos] > target_half_cost)
                ? (h_prefix_cost[pos] - target_half_cost)
                : (target_half_cost - h_prefix_cost[pos]);

        unsigned int diff_prev =
            (h_prefix_cost[pos - 1] > target_half_cost)
                ? (h_prefix_cost[pos - 1] - target_half_cost)
                : (target_half_cost - h_prefix_cost[pos - 1]);

        if (diff_prev < diff_pos) {
          best_idx = pos - 1;
        }
      }

      e1d = best_idx + 1;
    }

    if (e1d > h_num_half_edges) {
      e1d = h_num_half_edges;
    }
  }

  e2d = h_num_half_edges - e1d;

  split_fraction_1d =
      (h_num_half_edges > 0)
          ? (static_cast<double>(e1d) / static_cast<double>(h_num_half_edges))
          : 0.0;

  if (e2d > 0 && e1d < h_num_half_edges) {
    transition_cost = h_cost_sorted[e1d];
  } else {
    transition_cost = 0;
  }

  edge_t total_nnz_pairs = h_num_half_edges;

  // ----------------------------------------------------------
  // output buffer
  // ----------------------------------------------------------
  weight_t *weight_j = nullptr;
  cudaMalloc((void **)&weight_j, sizeof(weight_t) * h_num_half_edges);

  // ----------------------------------------------------------
  // streams
  // ----------------------------------------------------------
  cudaStream_t s1, s2;
  cudaStreamCreate(&s1);
  cudaStreamCreate(&s2);

  edge_t *d_work_ids_1d = d_work_ids_out;
  edge_t *d_work_ids_2d = d_work_ids_out + e1d;

  // ----------------------------------------------------------
  // 1D launch config
  // ----------------------------------------------------------
  int threads_1d = EC_MAX_THREADS_PER_BLOCK;
  if (e1d > 0 && e1d < threads_1d) {
    threads_1d = (int)e1d;
  }
  if (threads_1d <= 0) {
    threads_1d = 1;
  }

  int blocks_1d = (int)((e1d + threads_1d - 1) / threads_1d);

  // ----------------------------------------------------------
  // 2D launch config
  // ----------------------------------------------------------
  dim3 nthreads_2d(32, 1, 8);

  unsigned int blocks_z_2d =
      static_cast<unsigned int>(
          std::min<edge_t>(
              static_cast<edge_t>(
                  (e2d + static_cast<edge_t>(nthreads_2d.z) - 1) /
                  static_cast<edge_t>(nthreads_2d.z)),
              static_cast<edge_t>(CUDA_MAX_BLOCKS)));

  if (blocks_z_2d == 0) {
    blocks_z_2d = 1;
  }

  dim3 nblocks_2d(1, 1, blocks_z_2d);

  // ----------------------------------------------------------
  // run kernels 5 times with same split
  // ----------------------------------------------------------
  const int num_runs = 5;
  double total_runtime = 0.0;

  for (int run = 0; run < num_runs; run++) {
    cudaMemset(weight_j, 0, sizeof(weight_t) * h_num_half_edges);
    cudaDeviceSynchronize();

    double intersect_start = gettimer(&kernel_time);

    if (e1d > 0) {
      set_intersection_ec_lists<vertex_t, edge_t, weight_t>
          <<<blocks_1d, threads_1d, 0, s1>>>(
              e1d,
              d_work_ids_1d,
              d_src_list,
              d_dst_list,
              csrPtr,
              csrInd,
              weight_j);
    }

    if (e2d > 0) {
      set_intersection_lb_pairs_lists<vertex_t, edge_t, weight_t>
          <<<nblocks_2d, nthreads_2d, 0, s2>>>(
              e2d,
              d_work_ids_2d,
              d_src_list,
              d_dst_list,
              csrPtr,
              csrInd,
              weight_j);
    }

    cudaStreamSynchronize(s1);
    cudaStreamSynchronize(s2);

    double intersect_stop = gettimer(&kernel_time);
    total_runtime += (intersect_stop - intersect_start);
  }

  double avg_runtime = total_runtime / static_cast<double>(num_runs);

  // ----------------------------------------------------------
  // read back last run to count nonzero pairs
  // ----------------------------------------------------------
  weight_t *w_host = (weight_t *)malloc(sizeof(weight_t) * h_num_half_edges);
  cudaMemcpy(w_host,
             weight_j,
             sizeof(weight_t) * h_num_half_edges,
             cudaMemcpyDeviceToHost);

  edge_t total_nonzero_pairs = 0;
  weight_t thresh = static_cast<weight_t>(0.00001);

  for (edge_t k = 0; k < h_num_half_edges; k++) {
    if (w_host[k] > thresh) {
      total_nonzero_pairs++;
    }
  }

  std::cout << "avg_kernel_runtime,sort_overhead,prefix_overhead,total_edges,total_estimated_cost,"
            << "transition_cost,split_fraction_1d,e1d,e2d,total_nonzero_jaccard_pairs,num_runs"
            << std::endl;

  std::cout << avg_runtime << ","
            << sort_overhead << ","
            << prefix_overhead << ","
            << total_nnz_pairs << ","
            << total_estimated_cost << ","
            << transition_cost << ","
            << split_fraction_1d << ","
            << e1d << ","
            << e2d << ","
            << total_nonzero_pairs << ","
            << num_runs
            << std::endl;

  free(w_host);

  cudaStreamDestroy(s1);
  cudaStreamDestroy(s2);

  cudaFree(weight_j);

  free(h_cost_sorted);
  free(h_prefix_cost);

  cudaFree(d_scan_temp_storage);
  cudaFree(d_prefix_cost);

  cudaFree(d_sort_temp_storage);

  cudaFree(d_cost_in);
  cudaFree(d_cost_out);
  cudaFree(d_max_cost);

  cudaFree(d_work_ids_in);
  cudaFree(d_work_ids_out);

  cudaFree(d_src_list);
  cudaFree(d_dst_list);

  cudaFree(dest_ind);
}
  else { // Vertex-Centric
    cudaError_t error = cudaSuccess;
    weight_t *weight_i;
    error = cudaMalloc(&weight_i, static_cast<int64_t>(e) * sizeof(weight_t));
    if (error != cudaSuccess) {
      std::cerr << " CUDA ERROR " << error << " at line " << __LINE__ << std::endl;
      error = cudaSuccess;
    }
    weight_t *weight_s;
    error = cudaMalloc(&weight_s, static_cast<int64_t>(e) * sizeof(weight_t));
    if (error != cudaSuccess) {
      std::cerr << " CUDA ERROR " << error << " at line " << __LINE__ << std::endl;
      error = cudaSuccess;
    }
    weight_t *work;
    error = cudaMalloc(&work, static_cast<int64_t>(n) * sizeof(weight_t));
    if (error != cudaSuccess) {
      std::cerr << " CUDA ERROR " << error << " at line " << __LINE__ << std::endl;
      error = cudaSuccess;
    }
    dim3 nthreads, nblocks;
    int y = 4;

    // setup launch configuration
    nthreads.x = 32;
    nthreads.y = y;
    nthreads.z = 1;
    nblocks.x = 1;
    nblocks.y =
        std::min((size_t)(n + nthreads.y - 1) / nthreads.y, (size_t)vertex_t{CUDA_MAX_BLOCKS});
    nblocks.z = 1;

    // launch kernel
    double rowsum_start = gettimer(&kernel_time);
    jaccard_row_sum<weighted, vertex_t, edge_t, weight_t>
        <<<nblocks, nthreads>>>(n, csrPtr, csrInd, weight_in, work);
    error = cudaDeviceSynchronize();
    double rowsum_stop = gettimer(&kernel_time);
    double rtime = rowsum_stop - rowsum_start;
    //std::cout << "VC_RowSum Elapsed (s): " << rowsum_stop - rowsum_start << std::endl;
    if (error != cudaSuccess) std::cerr << "Error in jaccard_row_sum " << error << std::endl;
#ifdef DEBUG_2
    std::cout << "DEBUG: Post-RowSum Work matrix of " << n << " elements" << std::endl;
    weight_t *debug_work = new weight_t[n];
    cudaMemcpy(debug_work, work, static_cast<int64_t>(n) * sizeof(weight_t),
               cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) {
      std::cout << debug_work[i] << std::endl;
    }
    delete debug_work;
#endif // DEBUG_2
    fill(e, weight_i, weight_t{0.0});
#ifdef DEBUG_2
    std::cout << "DEBUG: Post-Fill Weight_i matrix of " << e << " elements" << std::endl;
    weight_t *debug_wi = new weight_t[e];
    cudaMemcpy(debug_wi, weight_i, static_cast<int64_t>(e) * sizeof(weight_t),
               cudaMemcpyDeviceToHost);
    for (int i = 0; i < e; i++) {
      std::cout << debug_wi[i] << std::endl;
    }
    delete debug_wi;
#endif // DEBUG_2

    // setup launch configuration
    nthreads.x = 32 / y;
    nthreads.y = y;
    nthreads.z = 8;
    nblocks.x = 1;
    nblocks.y = 1;
    nblocks.z = std::min((size_t)(n + nthreads.z - 1) / nthreads.z,
                         (size_t)vertex_t{CUDA_MAX_BLOCKS}); // 1;

    // launch kernel
    double intersect_start = gettimer(&kernel_time);
    jaccard_is<weighted, vertex_t, edge_t, weight_t>
        <<<nblocks, nthreads>>>(n, csrPtr, csrInd, weight_in, work, weight_i, weight_s);
    error = cudaDeviceSynchronize(); // Added, not necessary
    double intersect_stop = gettimer(&kernel_time);
    double itime = intersect_stop - intersect_start;
    //std::cout << "VC_Intersect Elapsed (s): " << intersect_stop - intersect_start << std::endl;
    if (error != cudaSuccess) std::cerr << "Error in jaccard_is " << error << std::endl;
#ifdef DEBUG_2
    std::cout << "DEBUG: Post-IS Weight_i and Weight_s matrices of " << e << " elements"
              << std::endl;
    debug_wi = new weight_t[e];
    weight_t *debug_ws = new weight_t[e];
    cudaMemcpy(debug_wi, weight_i, static_cast<int64_t>(e) * sizeof(weight_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(debug_ws, weight_s, static_cast<int64_t>(e) * sizeof(weight_t),
               cudaMemcpyDeviceToHost);
    for (int i = 0; i < e; i++) {
      std::cout << debug_wi[i] << " " << debug_ws[i] << std::endl;
    }
    delete debug_wi;
    delete debug_ws;
#endif // DEBUG_2

    // setup launch configuration
    nthreads.x = std::min((size_t)e, (size_t)edge_t{CUDA_MAX_KERNEL_THREADS});
    nthreads.y = 1;
    nthreads.z = 1;
    nblocks.x =
        std::min((size_t)(e + nthreads.x - 1) / nthreads.x, (size_t)edge_t{CUDA_MAX_BLOCKS});
    nblocks.y = 1;
    nblocks.z = 1;

    // launch kernel
    double weights_start = gettimer(&kernel_time);
    jaccard_jw<weighted, vertex_t, edge_t, weight_t>
        <<<nblocks, nthreads>>>(e, weight_i, weight_s, weight_j);
    // FIXME, remove this
    // cudaMemcpy(weight_j, weight_s, sizeof(weight_t) * static_cast<int64_t>(e),
    // cudaMemcpyDeviceToDevice);
    error = cudaDeviceSynchronize(); // Added, not necessary
    double weights_stop = gettimer(&kernel_time);
    double jtime = weights_stop - weights_start ;
    double ttime = rtime + itime + jtime;
    std::cout<<ttime<<std::endl;
    //std::cout << "VC_Weights Elapsed (s): " << weights_stop - weights_start << std::endl;
    if (error != cudaSuccess) std::cerr << "Error in jaccard_jw " << error << std::endl;
    cudaFree(weight_i);
    cudaFree(weight_s);
    cudaFree(work);
  }
  return 0;
}

template <bool weighted, typename vertex_t, typename edge_t, typename weight_t>
int jaccard_pairs(vertex_t n, edge_t num_pairs, edge_t const *csrPtr, vertex_t const *csrInd,
                  vertex_t const *first_pair, vertex_t const *second_pair,
                  weight_t const *weight_in, weight_t *work, weight_t *weight_i, weight_t *weight_s,
                  weight_t *weight_j) {
  dim3 nthreads, nblocks;
  int y = 4;

  // setup launch configuration
  nthreads.x = 32;
  nthreads.y = y;
  nthreads.z = 1;
  nblocks.x = 1;
  nblocks.y =
      std::min((size_t)(n + nthreads.y - 1) / nthreads.y, (size_t)vertex_t{CUDA_MAX_BLOCKS});
  nblocks.z = 1;

  // launch kernel
  jaccard_row_sum<weighted, vertex_t, edge_t, weight_t>
      <<<nblocks, nthreads>>>(n, csrPtr, csrInd, weight_in, work);
  cudaDeviceSynchronize();

  // NOTE: initilized weight_i vector with 0.0
  // fill(num_pairs, weight_i, weight_t{0.0});

  // setup launch configuration
  nthreads.x = 32;
  nthreads.y = 1;
  nthreads.z = 8;
  nblocks.x = 1;
  nblocks.y = 1;
  nblocks.z =
      std::min((size_t)(n + nthreads.z - 1) / nthreads.z, (size_t)vertex_t{CUDA_MAX_BLOCKS}); // 1;

  // launch kernel
  jaccard_is_pairs<weighted, vertex_t, edge_t, weight_t><<<nblocks, nthreads>>>(
      num_pairs, csrPtr, csrInd, first_pair, second_pair, weight_in, work, weight_i, weight_s);

  // setup launch configuration
  nthreads.x = std::min((size_t)num_pairs, (size_t)edge_t{CUDA_MAX_KERNEL_THREADS});
  nthreads.y = 1;
  nthreads.z = 1;
  nblocks.x =
      std::min((size_t)(num_pairs + nthreads.x - 1) / nthreads.x, (size_t)edge_t{CUDA_MAX_BLOCKS});
  nblocks.y = 1;
  nblocks.z = 1;

  // launch kernel
  jaccard_jw<weighted, vertex_t, edge_t, weight_t>
      <<<nblocks, nthreads>>>(num_pairs, weight_i, weight_s, weight_j);

  return 0;
}
} // namespace detail

template <bool edge_centric, typename VT, typename ET, typename WT>
void jaccard(GraphCSRView<VT, ET, WT> const &graph, WT const *weights, WT *result) {
  CUGRAPH_EXPECTS(result != nullptr, "Invalid input argument: result pointer is NULL");
  if (weights == nullptr) {
    cugraph::detail::jaccard<edge_centric, false, VT, ET, WT>(graph.number_of_vertices,
                                                              graph.number_of_edges, graph.offsets,
                                                              graph.indices, weights, result);
  } else {
    cugraph::detail::jaccard<edge_centric, true, VT, ET, WT>(graph.number_of_vertices,
                                                             graph.number_of_edges, graph.offsets,
                                                             graph.indices, weights, result);
  }
}

template <typename VT, typename ET, typename WT>
void jaccard_list(GraphCSRView<VT, ET, WT> const &graph, WT const *weights, ET num_pairs,
                  VT const *first, VT const *second, WT *result) {
  CUGRAPH_EXPECTS(result != nullptr, "Invalid input argument: result pointer is NULL");
  CUGRAPH_EXPECTS(first != nullptr, "Invalid input argument: first is NULL");
  CUGRAPH_EXPECTS(second != nullptr, "Invalid input argument: second in NULL");

  WT *weight_i;
  cudaMalloc(&weight_i, static_cast<int64_t>(graph.number_of_edges) * sizeof(WT));
  fill(graph.number_of_edges, weight_i, (WT)0.0);
  WT *weight_s;
  cudaMalloc(&weight_s, static_cast<int64_t>(graph.number_of_edges) * sizeof(WT));
  WT *work;
  cudaMalloc(&work, static_cast<int64_t>(graph.number_of_vertices) * sizeof(WT));

  if (weights == nullptr) {
    cugraph::detail::jaccard_pairs<false, VT, ET, WT>(graph.number_of_vertices, num_pairs,
                                                      graph.offsets, graph.indices, first, second,
                                                      weights, work, weight_i, weight_s, result);
  } else {
    cugraph::detail::jaccard_pairs<true, VT, ET, WT>(graph.number_of_vertices, num_pairs,
                                                     graph.offsets, graph.indices, first, second,
                                                     weights, work, weight_i, weight_s, result);
  }
  cudaFree(weight_i);
  cudaFree(weight_s);
  cudaFree(work);
}

template void jaccard<true, int32_t, int32_t, float>(GraphCSRView<int32_t, int32_t, float> const &,
                                                     float const *, float *);
template void jaccard<false, int32_t, int32_t, float>(GraphCSRView<int32_t, int32_t, float> const &,
                                                      float const *, float *);
#ifndef DISABLE_DP_WEIGHT
template void
jaccard<true, int32_t, int32_t, double>(GraphCSRView<int32_t, int32_t, double> const &,
                                        double const *, double *);
template void
jaccard<false, int32_t, int32_t, double>(GraphCSRView<int32_t, int32_t, double> const &,
                                         double const *, double *);
#endif // DISABLE_DP_WEIGHT
#ifndef DISABLE_DP_INDEX
template void jaccard<true, int64_t, int64_t, float>(GraphCSRView<int64_t, int64_t, float> const &,
                                                     float const *, float *);
template void jaccard<false, int64_t, int64_t, float>(GraphCSRView<int64_t, int64_t, float> const &,
                                                      float const *, float *);
  #ifndef DISABLE_DP_WEIGHT
template void
jaccard<true, int64_t, int64_t, double>(GraphCSRView<int64_t, int64_t, double> const &,
                                        double const *, double *);
template void
jaccard<false, int64_t, int64_t, double>(GraphCSRView<int64_t, int64_t, double> const &,
                                         double const *, double *);
  #endif // DISABLE_DP_WEIGHT
#endif   // DISABLE_DP_INDEX
#ifndef DISABLE_LIST
template void jaccard_list<int32_t, int32_t, float>(GraphCSRView<int32_t, int32_t, float> const &,
                                                    float const *, int32_t, int32_t const *,
                                                    int32_t const *, float *);
  #ifndef DISABLE_DP_WEIGHT
template void jaccard_list<int32_t, int32_t, double>(GraphCSRView<int32_t, int32_t, double> const &,
                                                     double const *, int32_t, int32_t const *,
                                                     int32_t const *, double *);
  #endif // DISABLE_DP_WEIGHT
  #ifndef DISABLE_DP_INDEX
template void jaccard_list<int64_t, int64_t, float>(GraphCSRView<int64_t, int64_t, float> const &,
                                                    float const *, int64_t, int64_t const *,
                                                    int64_t const *, float *);
    #ifndef DISABLE_DP_WEIGHT
template void jaccard_list<int64_t, int64_t, double>(GraphCSRView<int64_t, int64_t, double> const &,
                                                     double const *, int64_t, int64_t const *,
                                                     int64_t const *, double *);
    #endif // DISABLE_DP_WEIGHT
  #endif   // DISABLE_DP_INDEX
#endif     // DISABLE_LIST

} // namespace cugraph
