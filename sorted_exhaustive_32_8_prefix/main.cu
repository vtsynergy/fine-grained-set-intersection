#include "filetypes.hpp"
#include "readMtxToCSR.hpp" //implicitly includes standalone_csr.hpp
#include "standalone_algorithms.hpp"
#include "standalone_csr.hpp"
#include <cstring>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#ifndef WEIGHT_TYPE
  #ifndef DISABLE_DP_WEIGHT
    #define WEIGHT_TYPE double
  #else
    #define WEIGHT_TYPE float
  #endif
#endif

template <typename vertex_t>
__global__ void presum_kernel(vertex_t *indices, vertex_t n) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n) {
    indices[tid] = tid;
  }
}

template <typename vertex_t, typename edge_t, typename weight_t>
void cudaMemcpyCSR(GraphCSRView<vertex_t, edge_t, weight_t> dst,
                   GraphCSRView<vertex_t, edge_t, weight_t> src, enum cudaMemcpyKind dir) {
  cudaError_t error = cudaSuccess;
  if (dst.offsets != nullptr && src.offsets != nullptr)
    error =
        cudaMemcpy(dst.offsets, src.offsets, sizeof(edge_t) * (static_cast<int64_t>(dst.number_of_vertices) + 1), dir);
  if (error != cudaSuccess) {
    std::cerr << " CUDA ERROR " << error << " at line " << __LINE__ << std::endl;
    error = cudaSuccess;
  }
  if (dst.indices != nullptr && src.indices != nullptr)
    error = cudaMemcpy(dst.indices, src.indices, sizeof(vertex_t) * static_cast<int64_t>(dst.number_of_edges), dir);
  if (error != cudaSuccess) {
    std::cerr << " CUDA ERROR " << error << " at line " << __LINE__ << std::endl;
    error = cudaSuccess;
  }
  if (dst.edge_data != nullptr && src.edge_data != nullptr)
    error = cudaMemcpy(dst.edge_data, src.edge_data, sizeof(weight_t) * static_cast<int64_t>(dst.number_of_edges), dir);
  if (error != cudaSuccess) {
    std::cerr << " CUDA ERROR " << error << " at line " << __LINE__ << std::endl;
    error = cudaSuccess;
  }
}

typedef enum {
  noChoice = 0,
  isForced = 1,
  ec_coarse = 2,
  vc_coarse = 4,
  undefined = -1
} implSelect;

implSelect selectImplementation() {
  implSelect retVal = noChoice;
  char *force_ec = std::getenv("JACCARD_FORCE_EDGE_CENTRIC");
  if (force_ec != nullptr) {
    //std::cerr << "FORCE Edge-Centric Implementation" << std::endl;
    retVal = (implSelect)(ec_coarse | isForced);
  }
  char *force_vc = std::getenv("JACCARD_FORCE_VERTEX_CENTRIC");
  if (force_vc != nullptr) {
    //std::cerr << "FORCE Vertex-Centric Implementation" << std::endl;
    retVal = (implSelect)(vc_coarse | isForced);
  }
  return retVal;
}

int main(int argc, char *argv[]) {

  // Open the specified file for reading
  // TODO arg bounds safety
  std::ifstream fileIn;
  std::ofstream fileOut;
  graphFileType inType, outType, working;
  setupInFile(argv[1], fileIn, inType);
  setupOutFile(argv[2], fileOut, outType);
  bool keepReverseEdges = true;
  bool isWeighted = false, isDirected = false, hasReverseEdges = false, isZeroIndexed = false;
  GraphCSRView<int32_t, int32_t, WEIGHT_TYPE> *graph;
  std::set<std::tuple<int32_t, int32_t, WEIGHT_TYPE>> *mtx_graph;
  if (inType == mtx) { // IF extension is mtx, use the string r/w
    working = mtx;
    mtx_graph = fileToMTXSet<int32_t, int32_t, WEIGHT_TYPE>(fileIn, &isWeighted, &isDirected);
  } else if (inType == csr) { // IF extension is csr, use binary r/w
    working = csr;
    CSRFileHeader header;
    graph = static_cast<GraphCSRView<int32_t, int32_t, WEIGHT_TYPE> *>(FileToCSR(fileIn, &header));
    isWeighted = header.flags.isWeighted;
    isDirected = header.flags.isDirected;
    hasReverseEdges = header.flags.hasReverseEdges;
    isZeroIndexed = header.flags.isZeroIndexed;
    if (header.flags.isVertexT64 || header.flags.isEdgeT64 ||
        (header.flags.isWeighted &&
         ((header.flags.isWeightT64 && !std::is_same<WEIGHT_TYPE, double>::value) ||
          (!header.flags.isWeightT64 && !std::is_same<WEIGHT_TYPE, float>::value)))) {
      std::cerr << "Binary CSR Input Header does not match required data types" << std::endl;
      exit(3);
    }
  } else {
    std::cerr << "InputGraphType is" << inType << std::endl;
  }
  fileIn.close();
  // MTX needs to have the reverse edges generated
  if (!isDirected && !hasReverseEdges) {
    if (working == csr) {
      // Switch it to MTX to reverse them
      mtx_graph = CSRToMtx(*graph, isZeroIndexed, isWeighted);
      working = mtx;
      delete graph; // Don't need to maintain it as a CSR, as a new one will be generated later
    } else if (working != mtx) {
      // Future formats
    }
    std::set<std::tuple<int32_t, int32_t, WEIGHT_TYPE>> *reverse = invertDirection(*mtx_graph);
    mtx_graph->insert(reverse->begin(), reverse->end());
    hasReverseEdges = true;
    delete reverse;
  }
  // Convert it to a CSR
  if (working == mtx) {
    graph = mtxSetToCSR(*mtx_graph);
    working = csr;
    delete mtx_graph;
  } else if (working != csr) {
    // Future formats
  }

  // Add an environment variable to dump CSR for both input and output as a sideeffect of an
  // MTX-file run
  char *dump_csr = std::getenv("JACCARD_IN_CSR_DUMP_FILEPATH");
  if (dump_csr != nullptr) {
#ifdef DEBUG
    std::cerr << "Requested CSR Dump of input file \"" << argv[1] << "\" to \"" << dump_csr << "\""
              << std::endl;
#endif
    std::ofstream csrDumpFile(dump_csr,
                              std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);
    CSRToFile(csrDumpFile, (*graph), false, isWeighted);
    csrDumpFile.close();
    dump_csr = nullptr; // Reset it incase there is no output dump
  }
  // We can't override weighting until here, or else the MTX will get confused about tokens per line
  // if the file and override disagree on the presence of weight values. Undef=defer to file,
  // 1=Weighted, 0=Unweighted
  char *weighted_override = std::getenv("JACCARD_FORCE_WEIGHTED");
  if (weighted_override != NULL) {
#ifdef DEBUG
    std::cerr << "Force Override of Weighted computation, current value is: " << isWeighted
              << " Override set to: " << weighted_override << std::endl;
#endif
    if (std::strcmp(weighted_override, "1") == 0) isWeighted = true;
    if (std::strcmp(weighted_override, "0") == 0) isWeighted = false;
    // If the graph has null weights vector, we have to provide it something if it's being forced on
    if (isWeighted && graph->edge_data == nullptr) {
      std::vector<WEIGHT_TYPE> *forcedWeights =
          new std::vector<WEIGHT_TYPE>(graph->number_of_edges, WEIGHT_TYPE{1.0});
      graph->edge_data = forcedWeights->data();
    }
  }

  if (argc >= 4) {
    cudaSetDevice(atoi(argv[3]));
  }
  // Sanity check the MTX --> CSR --> MTX conversion
  // std::set<std::tuple<int32_t, int32_t, WEIGHT_TYPE>> * sanity_mtx = CSRToMtx(*graph);
  // for (std::tuple<int32_t, int32_t, WEIGHT_TYPE> edge : *sanity_mtx) {
  // std::cout << "Source, Destination, JS-Score: " << std::get<0>(edge) << " " << std::get<1>(edge)
  // << " " << std::get<2>(edge) << std::endl;
  //}
  // Miake a GPU graph
  struct timeval gpu_region_time;
  double gpu_region_start = gettimer(&gpu_region_time);
  cudaProfilerStart();
  int32_t *gpu_offsets, *gpu_columns;
  WEIGHT_TYPE *gpu_weights = nullptr;
  cudaError_t error = cudaSuccess;
  error = cudaMalloc(&gpu_offsets, sizeof(int32_t) * (static_cast<int64_t>(graph->number_of_vertices) + 1));
  if (error != cudaSuccess) {
    std::cerr << " CUDA ERROR " << error << " at line " << __LINE__ << std::endl;
    error = cudaSuccess;
  }
  error = cudaMalloc(&gpu_columns, sizeof(int32_t) * static_cast<int64_t>(graph->number_of_edges));
  if (error != cudaSuccess) {
    std::cerr << " CUDA ERROR " << error << " at line " << __LINE__ << std::endl;
    error = cudaSuccess;
  }
  if (isWeighted) {
    error = cudaMalloc(&gpu_weights, sizeof(WEIGHT_TYPE) * static_cast<int64_t>(graph->number_of_edges));
    if (error != cudaSuccess) {
      std::cerr << " CUDA ERROR " << error << " at line " << __LINE__ << std::endl;
      error = cudaSuccess;
    }
  }
  GraphCSRView<int32_t, int32_t, WEIGHT_TYPE> gpu_graph(
      gpu_offsets, gpu_columns, gpu_weights, graph->number_of_vertices, graph->number_of_edges);
  // Copy data to it
  cudaMemcpyCSR<int32_t, int32_t, WEIGHT_TYPE>(gpu_graph, *graph, cudaMemcpyHostToDevice);

  // Run the CPU implementation
  // TODO

  // Run the GPU implementation
  // Results buffer
  WEIGHT_TYPE *gpu_results, *gpu_results_d;
  gpu_results = (WEIGHT_TYPE *)malloc(sizeof(WEIGHT_TYPE *) * static_cast<int64_t>(gpu_graph.number_of_edges));
  error = cudaMalloc(&gpu_results_d, static_cast<int64_t>(gpu_graph.number_of_edges) * sizeof(WEIGHT_TYPE));
  if (error != cudaSuccess) {
    std::cerr << " CUDA ERROR " << error << " at line " << __LINE__ << std::endl;
    error = cudaSuccess;
  }
  // Pick an implementation to use
  // TODO: Automatic selection based on graph properties
  implSelect implementation = selectImplementation();
  if (implementation & ec_coarse) {
    cugraph::jaccard<true, int32_t, int32_t, WEIGHT_TYPE>(gpu_graph, nullptr, gpu_results_d);
  } else if (implementation & vc_coarse) {
    if (isWeighted) {
      // Preprocess edge weights into vertex weights (simply sum them for now) IFF weighted
      // Create and populate the pseudo csrInd buffer
      int32_t *presumInd;
      cudaMalloc(&presumInd, static_cast<int64_t>(graph->number_of_edges) * sizeof(int32_t));
      dim3 nthreads, nblocks;
      nthreads.x = 32;
      nthreads.y = 1;
      nthreads.z = 1;
      nblocks.x = std::min((size_t)(graph->number_of_edges + nthreads.x - 1) / nthreads.x,
                           size_t{CUDA_MAX_BLOCKS});
      nblocks.y = 1;
      nblocks.z = 1;
      presum_kernel<int32_t><<<nblocks, nthreads>>>(presumInd, graph->number_of_edges);
#ifdef DEBUG_2
      int32_t *debug_pi = new int32_t[graph->number_of_edges];
      cudaMemcpy(debug_pi, presumInd, static_cast<int64_t>(graph->number_of_edges) * sizeof(int32_t),
                 cudaMemcpyDeviceToHost);
      std::cout << "DEBUG: Post-PreSum weight index vector of " << graph->number_of_edges
                << " elements" << std::endl;
      for (int i = 0; i < graph->number_of_edges; i++) {
        std::cout << debug_pi[i] << std::endl;
      }
      delete debug_pi;
#endif // DEBUG_2

      // launch kernel
      // csrPtr should be reusable, that's just the start and end indicies for each row
      // csrInd is not going to be reusable. it needs to be an index into the weight structure,
      // which should effectively just be [0, num_edges) in order Work should be a new buffer of
      // length num_verts
      WEIGHT_TYPE *vertWeights;
      cudaMalloc(&vertWeights, sizeof(WEIGHT_TYPE) * static_cast<int64_t>(graph->number_of_vertices));
      // setup launch configuration
      nthreads.x = 32;
      nthreads.y = 1;
      nthreads.z = 1;
      nblocks.x = 1;
      nblocks.y = std::min((size_t)(graph->number_of_vertices + nthreads.y - 1) / nthreads.y,
                           size_t{CUDA_MAX_BLOCKS});
      nblocks.z = 1;
      cugraph::detail::jaccard_row_sum<true, int32_t, int32_t, WEIGHT_TYPE>
          <<<nblocks, nthreads>>>(gpu_graph.number_of_vertices, gpu_graph.offsets, presumInd,
                                  gpu_graph.edge_data, vertWeights);
#ifdef DEBUG_2
      WEIGHT_TYPE *debug_vw = new WEIGHT_TYPE[graph->number_of_vertices];
      cudaMemcpy(debug_vw, vertWeights, static_cast<int64_t>(graph->number_of_vertices) * sizeof(WEIGHT_TYPE),
                 cudaMemcpyDeviceToHost);
      std::cout << "DEBUG: Post-VertSum weight vector of " << graph->number_of_vertices
                << " elements" << std::endl;
      for (int i = 0; i < graph->number_of_vertices; i++) {
        std::cout << debug_vw[i] << std::endl;
      }
      delete debug_vw;
#endif // DEBUG_2
      // This assume the graph's pointers are in GPU memory
      cugraph::jaccard<false, int32_t, int32_t, WEIGHT_TYPE>(gpu_graph, vertWeights, gpu_results_d);
    } else {
      cugraph::jaccard<false, int32_t, int32_t, WEIGHT_TYPE>(gpu_graph, nullptr, gpu_results_d);
    }
  }
    exit(0);
  error = cudaPeekAtLastError();
  if (error != cudaSuccess) {
    std::cerr << " CUDA ERROR " << error << " at line " << __LINE__ << std::endl;
    error = cudaSuccess;
  }

  error = cudaMemcpy(gpu_results, gpu_results_d, sizeof(WEIGHT_TYPE) * static_cast<int64_t>(gpu_graph.number_of_edges),
                     cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    std::cerr << " CUDA ERROR " << error << " at line " << __LINE__ << std::endl;
    error = cudaSuccess;
  }
  // Release GPU buffers
  error = cudaFree(gpu_offsets);
  if (error != cudaSuccess) {
    std::cerr << " CUDA ERROR " << error << " at line " << __LINE__ << std::endl;
    error = cudaSuccess;
  }
  error = cudaFree(gpu_columns);
  if (error != cudaSuccess) {
    std::cerr << " CUDA ERROR " << error << " at line " << __LINE__ << std::endl;
    error = cudaSuccess;
  }
  error = cudaFree(gpu_weights);
  if (error != cudaSuccess) {
    std::cerr << " CUDA ERROR " << error << " at line " << __LINE__ << std::endl;
    error = cudaSuccess;
  }
  error = cudaFree(gpu_results_d);
  if (error != cudaSuccess) {
    std::cerr << " CUDA ERROR " << error << " at line " << __LINE__ << std::endl;
    error = cudaSuccess;
  }
  cudaProfilerStop();
  double gpu_region_stop = gettimer(&gpu_region_time);
  if (implementation & ec_coarse) {
    std::cout <<"EC_GPU_Region Elapsed (s): " << gpu_region_stop-gpu_region_start << std::endl;
  } else if (implementation & vc_coarse) {
    std::cout <<"VC_GPU_Region Elapsed (s): " << gpu_region_stop-gpu_region_start << std::endl;
  }
  // Create a new results graph view, on the host side
  // Create a results graph (in which the weights are the jaccard similarity)
  GraphCSRView<int32_t, int32_t, WEIGHT_TYPE> gpu_graph_results(
      graph->offsets, graph->indices, gpu_results, graph->number_of_vertices,
      graph->number_of_edges);
  // Don't need the inputs anymore
  if (isWeighted) {
    delete graph->edge_data;
  }
  delete graph;
  // Set isWeighted to true to retain the scors
  isWeighted = true;
  graph = nullptr;
  // Only remove edges if the formats disagree
  std::set<std::tuple<int32_t, int32_t, WEIGHT_TYPE>> *gpu_results_mtx = nullptr;
  if (hasReverseEdges && (!keepReverseEdges || (outType == mtx && !isDirected))) {
    gpu_results_mtx = CSRToMtx(gpu_graph_results, true, isWeighted);
    removeReverseEdges(*gpu_results_mtx);
    hasReverseEdges = false;
    working = mtx;
  }
  // Add an environment variable to dump CSR for output as a sideeffect of an MTX-file run
  dump_csr = std::getenv("JACCARD_OUT_CSR_DUMP_FILEPATH");
  if (dump_csr != nullptr) {
    if (working == mtx) {
      // The only reason it would be MTX at this point is if we had to delete reverse edges
      graph = mtxSetToCSR(*gpu_results_mtx, true, false);
      gpu_graph_results = *graph;
      delete graph; // Just holds pointers, don't need them anymore
      working = csr;
    }
#ifdef DEBUG
    std::cerr << "Requested CSR Dump of output file \"" << argv[2] << "\" to \"" << dump_csr << "\""
              << std::endl;
#endif
    std::ofstream csrDumpFile(dump_csr,
                              std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);
    CSRToFile(csrDumpFile, gpu_graph_results, isZeroIndexed, isWeighted, isDirected,
              keepReverseEdges);
    csrDumpFile.close();
  }
  if (outType == mtx) { // IF extension is mtx, use the string r/w
    // Print formatted output data (i.e convert back to MTX)
    if (working == csr &&
        gpu_results_mtx ==
            nullptr) { // We have not had to generate the MTX yet (there were no reverse edges)
      gpu_results_mtx =
          CSRToMtx<int32_t, int32_t, WEIGHT_TYPE>(gpu_graph_results, true, isWeighted);
    }
    mtxSetToFile(fileOut, *gpu_results_mtx, gpu_graph_results.number_of_vertices,
                 gpu_graph_results.number_of_edges, isWeighted, isDirected);
  } else if (outType == csr) { // IF extension is csr, use binary r/w
    if (working == mtx) { // We had to delete some reverse edges and haven't flipped back to CSR yet
      graph = mtxSetToCSR(*gpu_results_mtx);
      gpu_graph_results = *graph;
      delete graph; // Just holds pointers, don't need them anymore
      working = csr;
    }
    CSRToFile(fileOut, gpu_graph_results, isZeroIndexed, isWeighted, isDirected, hasReverseEdges);
  } // No else, but extensible if we need different outputs eventually
  fileOut.close();
  // Cleanup outputs. CSR is canonical form, so only delete pointers from there
  if (gpu_results_mtx != nullptr) {
    delete gpu_results_mtx;
    gpu_results_mtx = nullptr;
  }
  delete gpu_graph_results.offsets;
  delete gpu_graph_results.indices;
  if (gpu_graph_results.edge_data != nullptr) {
    delete gpu_graph_results.edge_data;
  }
}
