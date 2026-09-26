ifeq ($(DEBUG), 1)
  OPTS=-g -DDEBUG -O0
  CUDA_C_FLAGS=-G
else ifeq ($(DEBUG), 2)
  OPTS=-g -DDEBUG -O0 -DDEBUG_2
  CUDA_C_FLAGS=-G
else
  OPTS=-O3
endif
GCCVERSION := $(shell g++ -dumpversion)
ifeq ($(shell test $(GCCVERSION) -gt 9; echo $$?),0)
STD_CFLAGS := --std=c++20
STD_LDFLAGS := --std=c++20 -lstdc++fs
else
STD_CFLAGS := --std=c++2a
STD_LDFLAGS := --std=c++2a -lstdc++fs
endif
#Override standard until CUDA supports it
CFLAGS := $(CFLAGS) --std=c++17 -DNO_CXX20
LDFLAGS := $(LDFLAGS) --std=c++17

CFLAGS := $(CFLAGS) -DDISABLE_DP_WEIGHT -DDISABLE_LIST -DDISABLE_DP_INDEX
LDFLAGS := $(LDFLAGS) -lstdc++fs
CUDA_C_FLAGS := $(CUDA_C_FLAGS) $(CFLAGS)
CUDA_LD_FLAGS := $(CUDA_LD_FLAGS) $(LDFLAGS)
NVCC=nvcc

.PHONY: all
all: jaccardCUDA compareCoords fileConvert readCSRHeader

compareCoords: compareCoords.o readMtxToCSR.o filetypes.o
	g++ -o compareCoords compareCoords.o readMtxToCSR.o filetypes.o $(LDFLAGS)

fileConvert: fileConvert.o filetypes.o readMtxToCSR.o
	g++ -o fileConvert fileConvert.o filetypes.o readMtxToCSR.o $(LDFLAGS)

jaccardCUDA: jaccardCUDA.o readMtxToCSR.o main.o filetypes.o
	$(NVCC) -o jaccardCUDA jaccardCUDA.o readMtxToCSR.o main.o filetypes.o $(CUDA_LD_FLAGS)

readCSRHeader: readCSRHeader.o readMtxToCSR.o
	g++ -o readCSRHeader readCSRHeader.o readMtxToCSR.o $(LDFLAGS)

compareCoords.o: compareCoords.cpp readMtxToCSR.hpp standalone_csr.hpp
	g++ -o compareCoords.o -c compareCoords.cpp $(CFLAGS) $(OPTS)

fileConvert.o: fileConvert.cpp
	g++ -o fileConvert.o -c fileConvert.cpp $(CFLAGS) $(OPTS)

filetypes.o: filetypes.cpp
	g++ -o filetypes.o -c filetypes.cpp $(CFLAGS) $(OPTS)

jaccardCUDA.o: jaccard.cu standalone_csr.hpp
	$(NVCC) jaccard.cu -o jaccardCUDA.o -D STANDALONE --system-include ./ -c $(CUDA_C_FLAGS) $(OPTS)

main.o: main.cu
	$(NVCC) -o main.o -c main.cu $(CUDA_C_FLAGS) $(OPTS)

readCSRHeader.o: readCSRHeader.cpp
	g++ -o readCSRHeader.o -c readCSRHeader.cpp $(CFLAGS) $(OPTS)

readMtxToCSR.o: readMtxToCSR.cpp readMtxToCSR.hpp standalone_csr.hpp
	g++ -o readMtxToCSR.o -c readMtxToCSR.cpp $(CFLAGS) $(OPTS)

.PHONY: clean
clean:
	rm jaccardCUDA compareCoords fileConvert *.o
