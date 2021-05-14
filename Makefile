all:
	/usr/local/cuda-10.2/bin/nvcc dpll_cuda.cpp -O3 -m64 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_70,code=compute_70 -x cu -o dpll_cuda
