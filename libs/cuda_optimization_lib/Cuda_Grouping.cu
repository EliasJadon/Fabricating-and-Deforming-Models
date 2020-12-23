#include "Cuda_Grouping.cuh"
#include "Cuda_Minimizer.cuh"

namespace Utils_Cuda_Grouping {
	__device__ double3 sub(const double3 a, const double3 b)
	{
		return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
	}
	__device__ double dot(const double3 a, const double3 b)
	{
		return a.x * b.x + a.y * b.y + a.z * b.z;
	}
	__device__ double squared_norm(const double3 a)
	{
		return dot(a, a);
	}
	template<typename T>
	__global__ void setZeroKernel(T* vec)
	{
		vec[blockIdx.x] = 0;
	}
	template <unsigned int blockSize, typename T>
	__device__ void warpReduce(volatile T* sdata, unsigned int tid) {
		if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
		if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
		if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
		if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
		if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
		if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
	}
	__device__ double atomicAdd(double* address, double val, int flag)
	{
		unsigned long long int* address_as_ull =
			(unsigned long long int*)address;
		unsigned long long int old = *address_as_ull, assumed;
		do {
			assumed = old;
			old = atomicCAS(address_as_ull, assumed,
				__double_as_longlong(val +
					__longlong_as_double(assumed)));
			// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
		} while (assumed != old);
		return __longlong_as_double(old);
	}

	

	__global__ void gradientKernel(
		double* grad,
		const double* X,
		const unsigned int startX,
		const unsigned int startY,
		const unsigned int startZ,
		const int* Group_Ind,
		const double3* Const_Pos,
		const unsigned int size)
	{
		int i = blockIdx.x;
		if (i < size) {
			if (threadIdx.x == 0)
				grad[Group_Ind[i] + startX] = 2 * (X[Group_Ind[i] + startX] - Const_Pos[i].x);
			if (threadIdx.x == 1)
				grad[Group_Ind[i] + startY] = 2 * (X[Group_Ind[i] + startY] - Const_Pos[i].y);
			if (threadIdx.x == 2)
				grad[Group_Ind[i] + startZ] = 2 * (X[Group_Ind[i] + startZ] - Const_Pos[i].z);
		}
	}
}


double Cuda_Grouping::value(Cuda::Array<double>& curr_x) 
{
	return 18;
}
		
void Cuda_Grouping::gradient(Cuda::Array<double>& X)
{
}

Cuda_Grouping::Cuda_Grouping(
	const unsigned int numF,
	const unsigned int numV,
	const ConstraintsType const_Type)
{
	Cuda::initIndices(mesh_indices, numF, numV, 0);
	Cuda::AllocateMemory(grad, (3 * numV) + (7 * numF));
	Cuda::AllocateMemory(EnergyAtomic, 1);
	Cuda::AllocateMemory(Group_Ind, 0);
	//Choose the kind of constraints
	if (const_Type == ConstraintsType::VERTICES) { 
		std::cout << "Cuda_Grouping class Error! Invalid ConstraintsType.\n";
		exit(1);
	}
	if (const_Type == ConstraintsType::NORMALS) { 
		startX = mesh_indices.startNx;
		startY = mesh_indices.startNy;
		startZ = mesh_indices.startNz;
	}
	if (const_Type == ConstraintsType::SPHERES) { 
		startX = mesh_indices.startCx;
		startY = mesh_indices.startCy;
		startZ = mesh_indices.startCz;
	}
	
	//init host buffers...
	for (int i = 0; i < grad.size; i++) {
		grad.host_arr[i] = 0;
	}
	// Copy input vectors from host memory to GPU buffers.
	Cuda::MemCpyHostToDevice(grad);
}

Cuda_Grouping::~Cuda_Grouping() {
	cudaGetErrorString(cudaGetLastError());
	FreeMemory(grad);
	FreeMemory(EnergyAtomic);
	FreeMemory(Group_Ind);
}
