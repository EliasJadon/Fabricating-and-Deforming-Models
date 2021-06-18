#include "Cuda_FixChosenConstraints.cuh"
#include "Cuda_Minimizer.cuh"

namespace Utils_Cuda_FixChosenConstraints {
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

	template<unsigned int blockSize>
	__global__ void EnergyKernel(
		double* resAtomic,
		const double* curr_x,
		const unsigned int size,
		const int* Const_Ind,
		const double3* Const_Pos,
		const unsigned int startX,
		const unsigned int startY,
		const unsigned int startZ)
	{
		//init data
		extern __shared__ double energy_value[blockSize];
		unsigned int tid = threadIdx.x;
		unsigned int Global_idx = blockIdx.x * blockSize + tid;
		energy_value[tid] = 0;
		
		if (Global_idx < size) {
			double3 Vi = make_double3(
				curr_x[Const_Ind[Global_idx] + startX],
				curr_x[Const_Ind[Global_idx] + startY],
				curr_x[Const_Ind[Global_idx] + startZ]
			);
			energy_value[tid] = squared_norm(sub(Vi, Const_Pos[Global_idx]));
		}

		__syncthreads();

		if (blockSize >= 1024) { if (tid < 512) { energy_value[tid] += energy_value[tid + 512]; } __syncthreads(); }
		if (blockSize >= 512) { if (tid < 256) { energy_value[tid] += energy_value[tid + 256]; } __syncthreads(); }
		if (blockSize >= 256) { if (tid < 128) { energy_value[tid] += energy_value[tid + 128]; } __syncthreads(); }
		if (blockSize >= 128) { if (tid < 64) { energy_value[tid] += energy_value[tid + 64]; } __syncthreads(); }
		if (tid < 32) warpReduce<blockSize, double>(energy_value, tid);
		if (tid == 0) atomicAdd(resAtomic, energy_value[0], 0);
	}

	__global__ void gradientKernel(
		double* grad,
		const double* X,
		const unsigned int startX,
		const unsigned int startY,
		const unsigned int startZ,
		const int* Const_Ind,
		const double3* Const_Pos,
		const unsigned int size)
	{
		int i = blockIdx.x;
		if (i < size) {
			if (threadIdx.x == 0)
				grad[Const_Ind[i] + startX] = 2 * (X[Const_Ind[i] + startX] - Const_Pos[i].x);
			if (threadIdx.x == 1)
				grad[Const_Ind[i] + startY] = 2 * (X[Const_Ind[i] + startY] - Const_Pos[i].y);
			if (threadIdx.x == 2)
				grad[Const_Ind[i] + startZ] = 2 * (X[Const_Ind[i] + startZ] - Const_Pos[i].z);
		}
	}
}


void Cuda_FixChosenConstraints::value(Cuda::Array<double>& curr_x) {
	Utils_Cuda_FixChosenConstraints::setZeroKernel << <1, 1>> > (EnergyAtomic.cuda_arr);
	const unsigned int s = Const_Ind.size;
	Utils_Cuda_FixChosenConstraints::EnergyKernel<1024> << <ceil(s / (double)1024), 1024>> > (
		EnergyAtomic.cuda_arr,
		curr_x.cuda_arr,
		Const_Ind.size,
		Const_Ind.cuda_arr,
		Const_Pos.cuda_arr,
		startX, startY, startZ);
}
		
void Cuda_FixChosenConstraints::gradient(Cuda::Array<double>& X)
{
	Utils_Cuda_FixChosenConstraints::setZeroKernel << <grad.size, 1, 0, stream_gradient >> > (grad.cuda_arr);
	Utils_Cuda_FixChosenConstraints::gradientKernel << <Const_Ind.size, 3, 0, stream_gradient >> > (
		grad.cuda_arr,
		X.cuda_arr,
		startX, startY, startZ,
		Const_Ind.cuda_arr,
		Const_Pos.cuda_arr,
		Const_Ind.size);
}

Cuda_FixChosenConstraints::Cuda_FixChosenConstraints(const unsigned int numF, const unsigned int numV)
{
	cudaStreamCreate(&stream_value);
	cudaStreamCreate(&stream_gradient);
	Cuda::initIndices(mesh_indices, numF, numV, 0);
	Cuda::AllocateMemory(grad, (3 * numV) + (10 * numF));
	Cuda::AllocateMemory(EnergyAtomic, 1);
	Cuda::AllocateMemory(Const_Ind, 0);
	Cuda::AllocateMemory(Const_Pos, 0);
	//Choose the kind of constraints
	startX = mesh_indices.startVx;
	startY = mesh_indices.startVy;
	startZ = mesh_indices.startVz;
	
	//init host buffers...
	for (int i = 0; i < grad.size; i++) {
		grad.host_arr[i] = 0;
	}
	// Copy input vectors from host memory to GPU buffers.
	Cuda::MemCpyHostToDevice(grad);
}

Cuda_FixChosenConstraints::~Cuda_FixChosenConstraints() {
	cudaStreamDestroy(stream_value);
	cudaStreamDestroy(stream_gradient);
	cudaGetErrorString(cudaGetLastError());
	FreeMemory(grad);
	FreeMemory(EnergyAtomic);
	FreeMemory(Const_Ind);
	FreeMemory(Const_Pos);
}
