FINE-GRAINED SET INTERSECTION
=============================

Overview
--------

This repository contains a fine-grained GPU set-intersection approach for
computing graph metrics such as edge-connected Jaccard similarity and triangle
counts. It estimates the intersection cost of each edge, sorts the input edge
list by that cost, and divides the sorted work into two parts:

* The sparse, lower-cost end is sent to a thread-per-edge (1D) kernel, where
  one unique edge is processed by one GPU thread.
* The denser, higher-cost end is sent to a thread-collaborative (2D) kernel,
  where multiple GPU threads collaboratively perform each set intersection.

The division is selected so that the cumulative estimated cost assigned to
the two kernels is as close to equal as possible.


How to compile
--------------

Insert the appropriate module-load commands for your system to load a
compatible CUDA/GCC environment, or make sure the appropriate CUDA and GCC
tools are available in your PATH. Then enter the implementation directory and
run make:

  module load <appropriate CUDA module> <appropriate GCC module>
  cd sorted_exhaustive_32_8_prefix
  make
  cd ..

The build produces, among other files:

* jaccardCUDA: the standalone CUDA Jaccard executable.
* fileConvert: the graph-format conversion executable.


Pull input graphs
-----------------

This work uses graphs from the SuiteSparse Matrix Collection:

  https://sparse.tamu.edu/

The graphs are downloaded in Matrix Market (.mtx) format. From the repository
root, run:

  ./pull_graphs.sh

By default, the script downloads roadNet-CA, extracts the downloaded archive,
finds the roadNet-CA.mtx file, and copies it into dataset_mtx/. The temporary
archive and extraction directory are removed automatically afterward. The
dataset_mtx/ directory is created automatically if it does not already exist.

Additional graphs can be enabled by adding or uncommenting their wget URLs in
the URLS table in pull_graphs.sh and adding their graph names to the default
GRAPHS list. Named graphs can also be supplied as command-line arguments once
their URLs are present in that table:

  ./pull_graphs.sh roadNet-CA another_graph


Convert Matrix Market graphs to CSR
-----------------------------------

Running make generates fileConvert. This executable converts a Matrix Market
(.mtx) graph into the binary Compressed Sparse Row (CSR) format used to load
graphs for the benchmark. The generated binary files use the .csr extension.

Convert the default graph with:

  ./convert_to_csr.sh

The script reads dataset_mtx/roadNet-CA.mtx and writes
dataset_csr/roadNet-CA.csr. It drops input weights, generates reverse edges,
and configures the conversion for the edge-centric implementation. The
dataset_csr/ directory is created automatically if it does not already exist.
Multiple configured graphs may be converted by passing their names:

  ./convert_to_csr.sh roadNet-CA another_graph


How to run
----------

Run the default benchmark and collect its results with:

  ./run_graphs.sh

The script runs the edge-centric implementation on each requested .csr graph
and writes the combined results to fine_grained_results.csv. This file is
created automatically and overwritten each time run_graphs.sh is executed.
Multiple graphs may be named on the command line:

  ./run_graphs.sh roadNet-CA another_graph

To run the standalone executable directly on a graph named a.csr, use:

  export JACCARD_FORCE_EDGE_CENTRIC=1
  ./sorted_exhaustive_32_8_prefix/jaccardCUDA \
    a.csr sorted_exhaustive_32_8_prefix/out.csr

The first argument is the input graph. The second argument is the output CSR
file containing the computed Jaccard values. An optional third argument is the
CUDA device number.


Reported output
---------------

The run script adds the graph name and produces CSV output with this header:

  graph,avg_kernel_runtime,sort_overhead,prefix_overhead,total_edges,total_estimated_cost,transition_cost,split_fraction_1d,e1d,e2d,total_nonzero_jaccard_pairs,num_runs

Each field means:

* graph: input graph name.
* avg_kernel_runtime: average runtime in seconds of the Jaccard intersection
  kernels across all measured runs.
* sort_overhead: time in seconds used to initialize work IDs, estimate edge
  costs, and radix-sort edges by estimated cost.
* prefix_overhead: time in seconds used to compute the inclusive prefix sum of
  the sorted estimated costs.
* total_edges: total number of input edges processed as intersection work
  items.
* total_estimated_cost: sum of the estimated intersection costs for all input
  edges. This is an abstract work estimate, not elapsed time.
* transition_cost: estimated cost of the first edge assigned to the 2D
  thread-collaborative partition; it is the cost at the 1D/2D boundary.
* split_fraction_1d: fraction of input edges assigned to the 1D
  thread-per-edge kernel.
* e1d: number of edges assigned to the 1D thread-per-edge kernel.
* e2d: number of edges assigned to the 2D thread-collaborative kernel.
* total_nonzero_jaccard_pairs: number of output edge pairs whose computed
  Jaccard value is greater than 0.00001 (the code's nonzero threshold).
* num_runs: number of measured kernel runs used to calculate the average.
