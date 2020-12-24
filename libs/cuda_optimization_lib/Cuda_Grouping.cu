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
	template <unsigned int blockSize>
	__global__ void sumOfArray(double* g_idata, unsigned int n) {
		extern __shared__ double sdata[blockSize];
		unsigned int tid = threadIdx.x;
		unsigned int i = blockIdx.x * (blockSize * 2) + tid;
		unsigned int gridSize = blockSize * 2 * gridDim.x;
		sdata[tid] = 0;
		while (i < n) { sdata[tid] += g_idata[i] + g_idata[i + blockSize]; i += gridSize; }
		__syncthreads();

		if (blockSize >= 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }
		if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
		if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
		if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
		if (tid < 32) warpReduce<blockSize, double>(sdata, tid);
		if (tid == 0) atomicAdd(g_idata, sdata[0], 0);
	}

	__global__ void valueKernel(
		double* resAtomic,
		const double* curr_x,
		const unsigned int startX,
		const unsigned int startY,
		const unsigned int startZ,
		const int* Group_Ind,
		const unsigned int num_clusters,
		const unsigned int max_face_per_cluster)
	{
		int f1 = blockIdx.x;
		int f2 = blockIdx.y;
		int ci = blockIdx.z;
		int globalIndex = f1 + f2 * max_face_per_cluster + ci * max_face_per_cluster* max_face_per_cluster;
		if ((f1 > f2) &&
			(ci < num_clusters) &&
			(f1 < max_face_per_cluster) &&
			(f2 < max_face_per_cluster))
		{
			const unsigned int indexF1 = Group_Ind[ci * max_face_per_cluster + f1];
			const unsigned int indexF2 = Group_Ind[ci * max_face_per_cluster + f2];
			if (indexF1 != -1 && indexF2 != -1) {
				double3 NormalPos1 = make_double3(
					curr_x[indexF1 + startX],	//X-coordinate
					curr_x[indexF1 + startY],	//Y-coordinate
					curr_x[indexF1 + startZ]		//Z-coordinate
				);
				double3 NormalPos2 = make_double3(
					curr_x[indexF2 + startX],	//X-coordinate
					curr_x[indexF2 + startY],	//Y-coordinate
					curr_x[indexF2 + startZ]		//Z-coordinate
				);
				resAtomic[globalIndex] = Utils_Cuda_Grouping::squared_norm(Utils_Cuda_Grouping::sub(NormalPos1, NormalPos2));
			}
			else {
				resAtomic[globalIndex] = 0;
			}
		}
		else {
			resAtomic[globalIndex] = 0;
		}
	}

	__global__ void gradientKernel(
		double* grad,
		const double* X,
		const unsigned int startX,
		const unsigned int startY,
		const unsigned int startZ,
		const int* Group_Ind,
		const unsigned int num_clusters,
		const unsigned int max_face_per_cluster)
	{
		int f1 = blockIdx.x;
		int f2 = blockIdx.y;
		int ci = blockIdx.z;
		
		if ((f1 > f2) &&
			(ci < num_clusters) &&
			(f1 < max_face_per_cluster) &&
			(f2 < max_face_per_cluster)) 
		{
			const unsigned int indexF1 = Group_Ind[ci * max_face_per_cluster + f1];
			const unsigned int indexF2 = Group_Ind[ci * max_face_per_cluster + f2];
			if (indexF1 != -1 && indexF2 != -1) {
				if (threadIdx.x == 0)
					atomicAdd(&grad[indexF1 + startX], 2 * (X[indexF1 + startX] - X[indexF2 + startX]), 0);
				if (threadIdx.x == 1)
					atomicAdd(&grad[indexF2 + startX], -2 * (X[indexF1 + startX] - X[indexF2 + startX]), 0);
				if (threadIdx.x == 2)
					atomicAdd(&grad[indexF1 + startY], 2 * (X[indexF1 + startY] - X[indexF2 + startY]), 0);
				if (threadIdx.x == 3)
					atomicAdd(&grad[indexF2 + startY], -2 * (X[indexF1 + startY] - X[indexF2 + startY]), 0);
				if (threadIdx.x == 4)
					atomicAdd(&grad[indexF1 + startZ], 2 * (X[indexF1 + startZ] - X[indexF2 + startZ]), 0);
				if (threadIdx.x == 5)
					atomicAdd(&grad[indexF2 + startZ], -2 * (X[indexF1 + startZ] - X[indexF2 + startZ]), 0);
			}
		}
	}
}


double Cuda_Grouping::value(Cuda::Array<double>& curr_x) 
{
	/*Utils_Cuda_Grouping::setZeroKernel << <EnergyAtomic.size, 1 >> > (EnergyAtomic.cuda_arr);
	Cuda::CheckErr(cudaDeviceSynchronize());
	Utils_Cuda_Grouping::valueKernel
		<< <dim3(max_face_per_cluster, max_face_per_cluster, num_clusters), 1 >> > (
			EnergyAtomic.cuda_arr,
			curr_x.cuda_arr,
			startX,
			startY,
			startZ,
			Group_Ind.cuda_arr,
			num_clusters,
			max_face_per_cluster);
	Cuda::CheckErr(cudaDeviceSynchronize());
	
	Utils_Cuda_Grouping::sumOfArray<1024> << <ceil(EnergyAtomic.size / (double)1024), 1024 >> >
		(EnergyAtomic.cuda_arr, EnergyAtomic.size);

	Cuda::CheckErr(cudaDeviceSynchronize());

	MemCpyDeviceToHost(EnergyAtomic,1);

	return EnergyAtomic.host_arr[0];*/
	return 0;
}
		
Cuda::Array<double>* Cuda_Grouping::gradient(Cuda::Array<double>& X)
{
	Utils_Cuda_Grouping::setZeroKernel << <grad.size, 1 >> > (grad.cuda_arr);
	Cuda::CheckErr(cudaDeviceSynchronize());
	Utils_Cuda_Grouping::gradientKernel
		<< <dim3(max_face_per_cluster, max_face_per_cluster, num_clusters), 6 >> > (
			grad.cuda_arr,
			X.cuda_arr,
			startX,
			startY,
			startZ,
			Group_Ind.cuda_arr,
			num_clusters,
			max_face_per_cluster
			);
	Cuda::CheckErr(cudaDeviceSynchronize());
	return &grad;
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
