#include "standalone_csr.hpp"
#include <fstream>
#include <set>
#include <tuple>
#define CSR_BINARY_FORMAT_VERSION 2

template <typename ET, typename VT, typename WT>
std::tuple<ET, VT, WT> readCoord(std::ifstream &fileIn, bool isWeighted = true, bool dropWeights = false);
template <typename ET, typename VT, typename WT>
std::set<std::tuple<ET, VT, WT>> *fileToMTXSet(std::ifstream &fileIn, bool *hasWeights,
                                               bool *isDirected, VT *numVerts = nullptr,
                                               ET *numEdges = nullptr, bool dropWeights = true);
template <typename ET, typename VT, typename WT>
std::set<std::tuple<ET, VT, WT>> *invertDirection(std::set<std::tuple<ET, VT, WT>> &mtx);
template <typename ET, typename VT, typename WT>
void removeReverseEdges(std::set<std::tuple<ET, VT, WT>> &mtx);
template <typename ET, typename VT, typename WT>
GraphCSRView<VT, ET, WT> *mtxSetToCSR(std::set<std::tuple<ET, VT, WT>> &mtx, bool ignoreSelf = true,
                                      bool isZeroIndexed = false);
template <typename ET, typename VT, typename WT>
std::set<std::tuple<ET, VT, WT>> *CSRToMtx(GraphCSRView<VT, ET, WT> &csr,
                                           bool isZeroIndexed = false, bool isWeighted = false);
template <typename ET, typename VT, typename WT>
void mtxSetToFile(std::ofstream &fileOut, std::set<std::tuple<ET, VT, WT>> &mtx, int64_t numVerts,
                  int64_t numEdges, bool isWeighted = false, bool isDirected = false);

typedef struct {
  int64_t binaryFormatVersion = CSR_BINARY_FORMAT_VERSION;
  int64_t numVerts;
  int64_t numEdges;
  struct alignas(alignof(int64_t)) {
    bool isWeighted : 1;      // Whether an edge-weight vector is present
    bool isZeroIndexed : 1;   // Whether the vertex indices start at 0 (true) or false (1)
    bool isDirected : 1;      // Whether the graph was original read as general (true) or symmetric
                              // (false)
    bool hasReverseEdges : 1; // Only used if !isDirected to indicate whether the file contains just
                              // one direction for each bidirectional edge (and thus needs reverse
                              // edges to be generated) or whether the reverse edges are already
                              // included
    bool isVertexT64 : 1; // Whether the vertex type is int64_t (8 bytes wide) or int32_t (4 bytes
                          // wide
    bool isEdgeT64 : 1;   // Whether the edge type is int64_t (8 bytes wide) or int32_t (4 bytes)
    bool isWeightT64 : 1; // whether the weight type is double (8 bytes wide) or float (4 bytes
                          // wide)
#ifdef NO_CXX20
    long long zeroPad : 57; //Rely on the user to zero-initialize
#else
    long long zeroPad : 57 = 0; //Make sure the remaining bits are initialized to zero (C++20)
#endif
  } flags;
} CSRFileHeader;

template <typename ET, typename VT, typename WT>
void CSRToFile(std::ofstream &fileOut, GraphCSRView<VT, ET, WT> &csr, bool isZeroIndexed = false,
               bool isWeighted = false, bool isDirected = false, bool hasReverseEdges = true);

void *FileToCSR(std::ifstream &fileIn, CSRFileHeader *header);
