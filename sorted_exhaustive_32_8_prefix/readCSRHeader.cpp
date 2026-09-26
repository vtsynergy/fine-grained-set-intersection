#include "readMtxToCSR.hpp"
#include <iostream>

#ifndef WEIGHT_TYPE
  #ifndef DISABLE_DP_WEIGHT
    #define WEIGHT_TYPE double
  #else
    #define WEIGHT_TYPE float
  #endif
#endif

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "Error, incorrect number of args, usage is:\n.readCSRHeader <input.csr>"
              << std::endl;
  }
  std::ifstream fileIn(argv[1]);
  // Header information comes from the file
  CSRFileHeader header;
  fileIn.read(reinterpret_cast<char *>(&header), sizeof(CSRFileHeader));
  std::cout << "==================================" << std::endl;
  std::cout << "Header info from " << argv[1] << std::endl;
  std::cout << "Format: " << header.binaryFormatVersion << " Intepreted as: " << CSR_BINARY_FORMAT_VERSION << std::endl;
  std::cout << "==================================" << std::endl;
  std::cout << "isWeighted " << header.flags.isWeighted << std::endl;
  std::cout << "isZeroIndexed " << header.flags.isZeroIndexed << std::endl;
  std::cout << "isDirected " << header.flags.isDirected << std::endl;
  std::cout << "hasReverseEdges " << header.flags.hasReverseEdges << std::endl;
  std::cout << "isVertexT64 " << header.flags.isVertexT64 << std::endl;
  std::cout << "isEdgeT64 " << header.flags.isEdgeT64 << std::endl;
  std::cout << "isWeightT64 " << header.flags.isWeightT64 << std::endl;
  std::cout << "zeroPad " << header.flags.zeroPad << std::endl;
  std::cout << "Vertices " << header.numVerts << std::endl;
  std::cout << "Edges " << header.numEdges << std::endl;
  fileIn.close();
}
