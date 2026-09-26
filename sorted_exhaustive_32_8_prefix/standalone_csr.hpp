#ifndef __STANDALONE_CSR_HPP__
#define __STANDALONE_CSR_HPP__

// Handwritten
#ifndef __CUDACC__
  #define CUGRAPH_EXPECTS(cond, str)                                                               \
    {                                                                                              \
      if (!((cond))) {                                                                             \
        fprintf(stderr, str);                                                                      \
        exit((cond));                                                                              \
      }                                                                                            \
    }
#else
  #define CUGRAPH_EXPECTS(cond, str)                                                               \
    {}
#endif
// From utilities/graph_utils.cuh
#define CUDA_MAX_BLOCKS 65535
//#define CUDA_MAX_BLOCKS 33554432
#define CUDA_MAX_KERNEL_THREADS 256
#define US
#ifndef WEIGHT_TYPE
  #ifndef DISABLE_DP_WEIGHT
    #define WEIGHT_TYPE double
  #else
    #define WEIGHT_TYPE float
  #endif
#endif
enum class PropType { PROP_UNDEF, PROP_FALSE, PROP_TRUE };

struct GraphProperties {
  bool directed{false};
  bool weighted{false};
  bool multigraph{false};
  bool bipartite{false};
  bool tree{false};
  PropType has_negative_edges{PropType::PROP_UNDEF};
  GraphProperties() = default;
};

enum class DegreeDirection {
  IN_PLUS_OUT = 0, ///> Compute sum of in and out degree
  IN,              ///> Compute in degree
  OUT,             ///> Compute out degree
  DEGREE_DIRECTION_COUNT
};
/**
 * @brief       A graph stored in CSR (Compressed Sparse Row) format.
 *
 * @tparam vertex_t   Type of vertex id
 * @tparam edge_t   Type of edge id
 * @tparam weight_t   Type of weight
 */
template <typename vertex_t, typename edge_t, typename weight_t>
class GraphCSRView {
public:
  using vertex_type = vertex_t;
  using edge_type = edge_t;
  using weight_type = weight_t;

  void *handle;
  weight_t *edge_data; ///< edge weight

  GraphProperties prop;

  vertex_t number_of_vertices;
  edge_t number_of_edges;
  edge_t *offsets{nullptr}; ///< CSR offsets

  vertex_t *local_vertices;
  edge_t *local_edges;
  vertex_t *local_offsets;
  vertex_t *indices{nullptr}; ///< CSR indices

  /**
   * @brief      Fill the identifiers array with the vertex identifiers.
   *
   * @param[out]    identifiers      Pointer to device memory to store the vertex
   * identifiers
   */
  void get_vertex_identifiers(vertex_t *identifiers) const;

  void set_local_data(vertex_t *vertices, edge_t *edges, vertex_t *offsets) {
    local_vertices = vertices;
    local_edges = edges;
    local_offsets = offsets;
  }

  void set_handle(void *handle_in) {
    handle = handle_in;
  }

  /**
   * @brief      Default constructor
   */
  GraphCSRView() : GraphCSRView<vertex_t, edge_t, weight_t>(nullptr, nullptr, nullptr, 0, 0) {
  }
  GraphCSRView(weight_t *edge_data, vertex_t number_of_vertices, edge_t number_of_edges)
      : handle(nullptr), edge_data(edge_data), number_of_vertices(number_of_vertices),
        number_of_edges(number_of_edges), local_vertices(nullptr), local_edges(nullptr),
        local_offsets(nullptr) {
  }
  /**
   * @brief      Wrap existing arrays representing adjacency lists in a Graph.
   *             GraphCSRView does not own the memory used to represent this
   * graph. This
   *             function does not allocate memory.
   *
   * @param  offsets               This array of size V+1 (V is number of
   * vertices) contains the
   * offset of adjacency lists of every vertex. Offsets must be in the range [0,
   * E] (number of
   * edges).
   * @param  indices               This array of size E contains the index of
   * the destination for
   * each edge. Indices must be in the range [0, V-1].
   * @param  edge_data             This array of size E (number of edges)
   * contains the weight for
   * each edge.  This array can be null in which case the graph is considered
   * unweighted.
   * @param  number_of_vertices    The number of vertices in the graph
   * @param  number_of_edges       The number of edges in the graph
   */
  GraphCSRView(edge_t *offsets, vertex_t *indices, weight_t *edge_data, vertex_t number_of_vertices,
               edge_t number_of_edges)
      : handle(nullptr), offsets{offsets}, indices{indices}, edge_data(edge_data),
        number_of_vertices(number_of_vertices), number_of_edges(number_of_edges),
        local_vertices(nullptr), local_edges(nullptr), local_offsets(nullptr) {
  }

  bool has_data(void) const {
    return edge_data != nullptr;
  }
  /**
   * @brief      Fill the identifiers in the array with the source vertex
   * identifiers
   *
   * @param[out]    src_indices      Pointer to device memory to store the
   * source vertex identifiers
   */
  void get_source_indices(vertex_t *src_indices) const;

  /**
   * @brief     Computes degree(in, out, in+out) of all the nodes of a Graph
   *
   * @throws     cugraph::logic_error when an error occurs.
   *
   * @param[out] degree         Device array of size V (V is number of
   * vertices) initialized
   * to zeros. Will contain the computed degree of every vertex.
   * @param[in]  direction      Integer value indicating type of degree
   * calculation
   *                                      0 : in+out degree
   *                                      1 : in-degree
   *                                      2 : out-degree
   */
  void degree(edge_t *degree, DegreeDirection direction) const;
};

#endif
