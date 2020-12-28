#include "Cuda_FixAllVertices.cuh"
#include "Cuda_Minimizer.cuh"

namespace Utils_Cuda_FixAllVertices {
	template<typename T>
	__global__ void setZeroKernel(T* vec)
	{
		vec[blockIdx.x] = 0;
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

	template <unsigned int blockSize, typename T>
	__device__ void warpReduce(volatile T* sdata, unsigned int tid) {
		if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
		if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
		if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
		if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
		if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
		if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
	}

	template<unsigned int blockSize>
	__global__ void EnergyKernel(
		double* resAtomic,
		const double* curr_x,
		const double3* restShapeV,
		const unsigned int num_vertices)
	{
		//init data
		extern __shared__ double energy_value[blockSize];
		unsigned int tid = threadIdx.x;
		unsigned int Global_idx = blockIdx.x * blockSize + tid;
		energy_value[tid] = 0;

		if (Global_idx < num_vertices) {
			double diff_x = curr_x[Global_idx] - restShapeV[Global_idx].x;
			energy_value[tid] = diff_x * diff_x;
		}
		else if (Global_idx < 2 * num_vertices) {
			unsigned int V_index = Global_idx - num_vertices;
			double diff_y = curr_x[Global_idx] - restShapeV[V_index].y;
			energy_value[tid] = diff_y * diff_y;
		}
		else if (Global_idx < 3 * num_vertices) {
			unsigned int V_index = Global_idx - 2 * num_vertices;
			double diff_z = curr_x[Global_idx] - restShapeV[V_index].z;
			energy_value[tid] = diff_z * diff_z;
		}

		__syncthreads();

		if (blockSize >= 1024) { if (tid < 512) { energy_value[tid] += energy_value[tid + 512]; } __syncthreads(); }
		if (blockSize >= 512) { if (tid < 256) { energy_value[tid] += energy_value[tid + 256]; } __syncthreads(); }
		if (blockSize >= 256) { if (tid < 128) { energy_value[tid] += energy_value[tid + 128]; } __syncthreads(); }
		if (blockSize >= 128) { if (tid < 64) { energy_value[tid] += energy_value[tid + 64]; } __syncthreads(); }
		if (tid < 32) warpReduce<blockSize, double>(energy_value, tid);
		if (tid == 0) atomicAdd(resAtomic, energy_value[0], 0);
	}

	template<unsigned int blockSize>
	__global__ void gradientKernel(
		double* grad,
		const double* X,
		const double3* restShapeV,
		const unsigned int num_vertices,
		const unsigned int size)
	{
		unsigned int tid = threadIdx.x;
		unsigned int Global_idx = blockIdx.x * blockSize + tid;
		if (Global_idx < num_vertices) {
			double diff_x = X[Global_idx] - restShapeV[Global_idx].x;
			grad[Global_idx] = 2 * diff_x; //X-coordinate
		}
		else if (Global_idx < 2 * num_vertices) {
			unsigned int V_index = Global_idx - num_vertices;
			double diff_y = X[Global_idx] - restShapeV[V_index].y;
			grad[Global_idx] = 2 * diff_y; //Y-coordinate
		}
		else if (Global_idx < 3 * num_vertices) {
			unsigned int V_index = Global_idx - 2 * num_vertices;
			double diff_z = X[Global_idx] - restShapeV[V_index].z;
			grad[Global_idx] = 2 * diff_z; //Z-coordinate
		}
		else if (Global_idx < size) {
			grad[Global_idx] = 0;
		}
	}
}
	
void Cuda_FixAllVertices::value(Cuda::Array<double>& curr_x) {
	Utils_Cuda_FixAllVertices::setZeroKernel << <1, 1>> > (EnergyAtomic.cuda_arr);
	unsigned int s = 3 * num_vertices;
	Utils_Cuda_FixAllVertices::EnergyKernel<1024> << <ceil(s / (double)1024), 1024>> > (
		EnergyAtomic.cuda_arr,
		curr_x.cuda_arr,
		restShapeV.cuda_arr,
		num_vertices);
}

void Cuda_FixAllVertices::gradient(Cuda::Array<double>& X)
{
	unsigned int s = grad.size;
	Utils_Cuda_FixAllVertices::gradientKernel<1024> << <ceil(s / (double)1024), 1024, 0, stream_gradient >> > (
		grad.cuda_arr,
		X.cuda_arr,
		restShapeV.cuda_arr,
		num_vertices,
		grad.size);
}

Cuda_FixAllVertices::Cuda_FixAllVertices(){
	cudaStreamCreate(&stream_value);
	cudaStreamCreate(&stream_gradient);
}

Cuda_FixAllVertices::~Cuda_FixAllVertices() {
	cudaStreamDestroy(stream_value);
	cudaStreamDestroy(stream_gradient);
	cudaGetErrorString(cudaGetLastError());
	FreeMemory(restShapeV);
	FreeMemory(grad);
	FreeMemory(EnergyAtomic);
}

