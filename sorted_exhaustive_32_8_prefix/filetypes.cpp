#if __GNUC__ == 7
  #include <experimental/filesystem>
namespace std {
namespace filesystem = experimental::filesystem;
}
#else
  #include <filesystem>
#endif
#include "filetypes.hpp"

void setupInFile(char *inFile, std::ifstream &retIFS, graphFileType &inType) {
  std::filesystem::path inPath(inFile);
  if (inPath.extension() == ".mtx") {
    inType = mtx;
    retIFS = std::ifstream(inPath, std::ios_base::in);
  } else if (inPath.extension() == ".csr") {
    inType = csr;
    retIFS = std::ifstream(inPath, std::ios_base::in | std::ios_base::binary);
  } else {
    std::cerr << "Input File " << inPath
              << "has illegal extension, must be \".mtx\" (text) or \".csr\" (binary)" << std::endl;
    exit(1);
  }
}

void setupOutFile(char *outFile, std::ofstream &retOFS, graphFileType &outType) {
  std::filesystem::path outPath(outFile);
  if (outPath.extension() == ".mtx") {
    outType = mtx;
    retOFS = std::ofstream(outPath, std::ios_base::out | std::ios_base::trunc);
  } else if (outPath.extension() == ".csr") {
    outType = csr;
    retOFS =
        std::ofstream(outPath, std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);
  } else {
    std::cerr << "Output File " << outPath
              << "has illegal extension, must be \".mtx\" (text) or \".csr\" (binary)" << std::endl;
    exit(2);
  }
}
