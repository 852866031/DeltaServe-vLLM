export CUDA_HOME=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
export LIBRARY_PATH=$CONDA_PREFIX/lib/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}
hash -r                                  # clear bash's cached path to any old nvcc
which nvcc && nvcc --version             # MUST show $CONDA_PREFIX/bin/nvcc, release 13.0