#include "standalone_csr.hpp"

#ifndef __STANDALONE_ALGORITHMS_HPP__
  #define __STANDALONE_ALGORITHMS_HPP__

#include <sys/time.h>
double gettimer(struct timeval *timer);

namespace cugraph {

/**
 * @brief     Compute jaccard similarity coefficient for all vertices
 *
 * Computes the Jaccard similarity coefficient for every pair of vertices in the graph
 * which are connected by an edge.
 *
 * @throws                 cugraph::logic_error when an error occurs.
 *
 * @tparam VT              Type of vertex identifiers. Supported value : int (signed, 32-bit)
 * @tparam ET              Type of edge identifiers. Supported value : int (signed, 32-bit)
 * @tparam WT              Type of edge weights. Supported value : float or double.
 *
 * @param[in] graph        The input graph object
 * @param[in] weights      device pointer to input vertex weights for weighted Jaccard, may be NULL
 * for unweighted Jaccard.
 * @param[out] result      Device pointer to result values, memory needs to be pre-allocated by
 * caller
 */
template <bool edge_centric, typename VT, typename ET, typename WT>
void jaccard(GraphCSRView<VT, ET, WT> const &graph, WT const *weights, WT *result);

/**
 * @brief     Compute jaccard similarity coefficient for selected vertex pairs
 *
 * Computes the Jaccard similarity coefficient for each pair of specified vertices.
 * Vertices are specified as pairs where pair[n] = (first[n], second[n])
 *
 * @throws                 cugraph::logic_error when an error occurs.
 *
 * @tparam VT              Type of vertex identifiers. Supported value : int (signed, 32-bit)
 * @tparam ET              Type of edge identifiers. Supported value : int (signed, 32-bit)
 * @tparam WT              Type of edge weights. Supported value : float or double.
 *
 * @param[in] graph        The input graph object
 * @param[in] weights      The input vertex weights for weighted Jaccard, may be NULL for
 *                         unweighted Jaccard.
 * @param[in] num_pairs    The number of vertex ID pairs specified
 * @param[in] first        Device pointer to first vertex ID of each pair
 * @param[in] second       Device pointer to second vertex ID of each pair
 * @param[out] result      Device pointer to result values, memory needs to be pre-allocated by
 * caller
 */
template <typename VT, typename ET, typename WT>
void jaccard_list(GraphCSRView<VT, ET, WT> const &graph, WT const *weights, ET num_pairs,
                  VT const *first, VT const *second, WT *result);

namespace detail {
template <bool weighted, typename vertex_t, typename edge_t, typename weight_t>
__global__ void jaccard_row_sum(vertex_t n, edge_t const *csrPtr, vertex_t const *csrInd,
                                weight_t const *v, weight_t *work);
} // namespace detail
} // namespace cugraph

#endif
