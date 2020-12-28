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

	template<unsigned int blockSize>
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
		extern __shared__ double energy_value[blockSize];
		unsigned int tid = threadIdx.x;
		unsigned int Global_idx = blockIdx.x * blockSize + tid;
		energy_value[tid] = 0;
		const int F = max_face_per_cluster;
		const int F2 = F * F;
		const int size = F2 * num_clusters;

		if (Global_idx < size) {
			int ci = (int)Global_idx / (int)F2;
			int f2 = (int)(Global_idx - (ci * F2)) / (int)F;
			int f1 = Global_idx - (ci * F2) - (f2 * F);

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
					energy_value[tid] = squared_norm(sub(NormalPos1, NormalPos2));
				}
			}
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
		const int* Group_Ind,
		const unsigned int num_clusters,
		const unsigned int max_face_per_cluster)
	{
		int f1 = blockIdx.x;
		int ci = blockIdx.y;
		int start;
		if (threadIdx.x == 0)
			start = startX;
		else if (threadIdx.x == 1)
			start = startY;
		else if (threadIdx.x == 2)
			start = startZ;
		else return;
		if (!((ci < num_clusters) && (f1 < max_face_per_cluster)))
			return;
		const unsigned int indexF1 = Group_Ind[ci * max_face_per_cluster + f1];
		if (indexF1 == -1)
			return;

		double X_value = X[indexF1 + start];
		double grad_value = 0;
		for (int f2 = 0; f2 < max_face_per_cluster; f2++) {
			const unsigned int indexF2 = Group_Ind[ci * max_face_per_cluster + f2];
			if ((f1 != f2) && (indexF2 != -1)) {
				grad_value += 2 * (X_value - X[indexF2 + start]);
			}
		}
		grad[indexF1 + start] = grad_value;
	}
}


double Cuda_Grouping::value(Cuda::Array<double>& curr_x) 
{
	Utils_Cuda_Grouping::setZeroKernel << <1, 1 >> > (EnergyAtomic.cuda_arr);
	Cuda::CheckErr(cudaDeviceSynchronize());


	const unsigned int s = max_face_per_cluster * max_face_per_cluster * num_clusters;
	Utils_Cuda_Grouping::valueKernel<1024> << <ceil(s / (double)1024), 1024 >> > (
			EnergyAtomic.cuda_arr,
			curr_x.cuda_arr,
			startX,
			startY,
			startZ,
			Group_Ind.cuda_arr,
			num_clusters,
			max_face_per_cluster);
	Cuda::CheckErr(cudaDeviceSynchronize());
	MemCpyDeviceToHost(EnergyAtomic);
	return EnergyAtomic.host_arr[0];
}
		
Cuda::Array<double>* Cuda_Grouping::gradient(Cuda::Array<double>& X)
{
	Utils_Cuda_Grouping::setZeroKernel << <grad.size, 1 >> > (grad.cuda_arr);
	Cuda::CheckErr(cudaDeviceSynchronize());
	Utils_Cuda_Grouping::gradientKernel
		<< <dim3(max_face_per_cluster, num_clusters,1), 3 >> > (
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
	num_clusters = 0;
	max_face_per_cluster = 0;
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
