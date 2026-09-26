#include <fstream>
#include <iostream>
typedef enum { mtx, csr, other = -1 } graphFileType;
void setupInFile(char *inFile, std::ifstream &retIFS, graphFileType &inType);
void setupOutFile(char *outFile, std::ofstream &retOFS, graphFileType &outType);
