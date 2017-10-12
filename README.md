## A C++ Implementation of Delta Boosting Machines

Features:

- Boost random forests in a parallel way using OpenMP.
- Support different base learners in an automated manner, including trees, linear regressors, DPC_stairs, Kmeans2d and Splines.
- Directly use deltas, instead of gradients, to do the boosting.

Build:

- cmake CMakeLists.txt
- make -j4

APIs:

- Python & R

Remark:

- Details refer to the document.
- Set MKLROOT to the root of Intel MKL in the environment manually before building.
