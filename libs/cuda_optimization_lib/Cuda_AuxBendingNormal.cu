#include "Cuda_AuxBendingNormal.cuh"
#include "Cuda_Minimizer.cuh"
//#include "Cuda_OptimizationUtils.cuh"

namespace Utils_Cuda_AuxBendingNormal {
	template<typename T>
	__global__ void setZeroKernel(T* vec)
	{
		vec[blockIdx.x] = 0;
	}
	__device__ double Phi(
		const double x,
		const double planarParameter,
		const FunctionType functionType,
		const double weight)
	{
		if (functionType == FunctionType::SIGMOID) {
			double x2 = pow(x / weight, 2);
			return x2 / (x2 + planarParameter);
		}
		if (functionType == FunctionType::QUADRATIC)
			return pow(x, 2);
		if (functionType == FunctionType::EXPONENTIAL)
			return exp(x * x);
	}
	__device__ double3 sub(const double3 a, const double3 b)
	{
		return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
	}
	__device__ double3 add(double3 a, double3 b)
	{
		return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
	}
	__device__ double dot(const double3 a, const double3 b)
	{
		return a.x * b.x + a.y * b.y + a.z * b.z;
	}
	__device__ double3 mul(const double a, const double3 b)
	{
		return make_double3(a * b.x, a * b.y, a * b.z);
	}
	__device__ double squared_norm(const double3 a)
	{
		return dot(a, a);
	}
	__device__ double norm(const double3 a)
	{
		return sqrt(squared_norm(a));
	}
	__device__ double3 normalize(const double3 a)
	{
		return mul(1.0f / norm(a), a);
	}
	__device__ double3 cross(const double3 a, const double3 b)
	{
		return make_double3(
			a.y * b.z - a.z * b.y,
			a.z * b.x - a.x * b.z,
			a.x * b.y - a.y * b.x
		);
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
	__device__ double dPhi_dm(
		const double x,
		const double planarParameter,
		const FunctionType functionType,
		const double weight)
	{
		const double w2 = pow(weight, 2);
		if (functionType == FunctionType::SIGMOID)
			return (2 * x * w2 * planarParameter) / pow(x * x + planarParameter * w2, 2);
		if (functionType == FunctionType::QUADRATIC)
			return 2 * x;
		if (functionType == FunctionType::EXPONENTIAL)
			return 2 * x * exp(x * x);
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
	template <unsigned int blockSize, typename T>
	__global__ void sumOfArray(T* g_idata, T* g_odata, unsigned int n) {
		extern __shared__ T sdata[blockSize];
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
		if (tid < 32) warpReduce<blockSize, T>(sdata, tid);
		if (tid == 0) g_odata[blockIdx.x] = sdata[0];
	}

	__device__ double Energy1Kernel(
		const double w1,
		const double* curr_x,
		const Cuda::hinge* hinges_faceIndex,
		const double* restAreaPerHinge,
		const double* weightPerHinge,
		const double planarParameter,
		const FunctionType functionType,
		const int hi,
		const Cuda::indices I)
	{
		int f0 = hinges_faceIndex[hi].f0;
		int f1 = hinges_faceIndex[hi].f1;
		if ((f0 >= I.num_faces) || (f1 >= I.num_faces))
			return;
		double3 N0 = make_double3(
			curr_x[f0 + I.startNx],
			curr_x[f0 + I.startNy],
			curr_x[f0 + I.startNz]
		);
		double3 N1 = make_double3(
			curr_x[f1 + I.startNx],
			curr_x[f1 + I.startNy],
			curr_x[f1 + I.startNz]
		);
		double3 diff = sub(N1, N0);
		double d_normals = squared_norm(diff);
		return w1 * restAreaPerHinge[hi] * weightPerHinge[hi] *
			Phi(d_normals, planarParameter, functionType, weightPerHinge[hi]);
	}
	__device__ double Energy2Kernel(
		const double w2,
		const double* curr_x,
		const int fi,
		const Cuda::indices I)
	{
		if (fi >= I.num_faces)
			return;
		double3 N = make_double3(
			curr_x[fi + I.startNx],
			curr_x[fi + I.startNy],
			curr_x[fi + I.startNz]
		);
		return pow(squared_norm(N) - 1, 2) * w2;
	}
	__device__ double Energy3Kernel(
		const double w3,
		const int3* restShapeF,
		const double* curr_x,
		const int fi,
		const Cuda::indices I)
	{
		// (N^T*(x1-x0))^2 + (N^T*(x2-x1))^2 + (N^T*(x0-x2))^2
		if (fi >= I.num_faces)
			return;
		int x0 = restShapeF[fi].x;
		int x1 = restShapeF[fi].y;
		int x2 = restShapeF[fi].z;
		double3 V0 = make_double3(
			curr_x[x0 + I.startVx],
			curr_x[x0 + I.startVy],
			curr_x[x0 + I.startVz]
		);
		double3 V1 = make_double3(
			curr_x[x1 + I.startVx],
			curr_x[x1 + I.startVy],
			curr_x[x1 + I.startVz]
		);
		double3 V2 = make_double3(
			curr_x[x2 + I.startVx],
			curr_x[x2 + I.startVy],
			curr_x[x2 + I.startVz]
		);
		double3 N = make_double3(
			curr_x[fi + I.startNx],
			curr_x[fi + I.startNy],
			curr_x[fi + I.startNz]
		);
		double3 e21 = sub(V2, V1);
		double3 e10 = sub(V1, V0);
		double3 e02 = sub(V0, V2);
		return w3 * (pow(dot(N, e21), 2) + pow(dot(N, e10), 2) + pow(dot(N, e02), 2));
	}

	template<unsigned int blockSize>
	__global__ void EnergyKernel(
		double* resAtomic,
		const double w1,
		const double w2,
		const double w3,
		const double* curr_x,
		const int3* restShapeF,
		const double* restAreaPerHinge,
		const double* weightPerHinge,
		const Cuda::hinge* hinges_faceIndex,
		const double planarParameter,
		const FunctionType functionType,
		const Cuda::indices mesh_indices)
	{
		extern __shared__ double energy_value[blockSize];
		unsigned int tid = threadIdx.x;
		unsigned int Global_idx = blockIdx.x * blockSize + tid;
		energy_value[tid] = 0;
		
		//0	,..., F-1,		==> Call Energy(3)
		//F	,..., 2F-1,		==> Call Energy(2)
		//2F,..., 2F+h-1	==> Call Energy(1)
		if (Global_idx < mesh_indices.num_faces) {
			energy_value[tid] = Energy3Kernel(
				w3,
				restShapeF,
				curr_x,
				Global_idx,
				mesh_indices);
		}
		else if (Global_idx < (2 * mesh_indices.num_faces)) {
			energy_value[tid] = Energy2Kernel(
				w2,
				curr_x,
				Global_idx - mesh_indices.num_faces,
				mesh_indices);
		}
		else if (Global_idx < ((2 * mesh_indices.num_faces) + mesh_indices.num_hinges)) {
			energy_value[tid] = Energy1Kernel(
				w1,
				curr_x,
				hinges_faceIndex,
				restAreaPerHinge,
				weightPerHinge,
				planarParameter,
				functionType,
				Global_idx - (2 * mesh_indices.num_faces),
				mesh_indices);
		}

		__syncthreads();

		if (blockSize >= 1024) { if (tid < 512) { energy_value[tid] += energy_value[tid + 512]; } __syncthreads(); }
		if (blockSize >= 512) { if (tid < 256) { energy_value[tid] += energy_value[tid + 256]; } __syncthreads(); }
		if (blockSize >= 256) { if (tid < 128) { energy_value[tid] += energy_value[tid + 128]; } __syncthreads(); }
		if (blockSize >= 128) { if (tid < 64) { energy_value[tid] += energy_value[tid + 64]; } __syncthreads(); }
		if (tid < 32) warpReduce<blockSize, double>(energy_value, tid);
		if (tid == 0) atomicAdd(resAtomic, energy_value[0], 0);
	}


	__device__ void gradient1Kernel(
		double* grad,
		const double* X,
		const Cuda::hinge* hinges_faceIndex,
		const double* restAreaPerHinge,
		const double* weightPerHinge,
		const double planarParameter,
		const FunctionType functionType,
		const double w1,
		const int hi,
		const int thread,
		const Cuda::indices I)
	{
		int f0 = hinges_faceIndex[hi].f0;
		int f1 = hinges_faceIndex[hi].f1;
		if ((f0 >= I.num_faces) || (f1 >= I.num_faces))
			return;
		double3 N0 = make_double3(
			X[f0 + I.startNx],
			X[f0 + I.startNy],
			X[f0 + I.startNz]
		);
		double3 N1 = make_double3(
			X[f1 + I.startNx],
			X[f1 + I.startNy],
			X[f1 + I.startNz]
		);
		double3 diff = sub(N1, N0);
		double d_normals = squared_norm(diff);

		double coeff = w1 * restAreaPerHinge[hi]* weightPerHinge[hi] * 
			dPhi_dm(d_normals, planarParameter, functionType, weightPerHinge[hi]);

		if (thread == 0) //n0.x;
			atomicAdd(&grad[f0 + I.startNx], coeff * 2 * (N0.x - N1.x), 0);
		else if (thread == 1) //n1.x
			atomicAdd(&grad[f1 + I.startNx], coeff * 2 * (N1.x - N0.x), 0);
		else if (thread == 2) //n0.y
			atomicAdd(&grad[f0 + I.startNy], coeff * 2 * (N0.y - N1.y), 0);
		else if (thread == 3) //n1.y
			atomicAdd(&grad[f1 + I.startNy], coeff * 2 * (N1.y - N0.y), 0);
		else if (thread == 4) //n0.z
			atomicAdd(&grad[f0 + I.startNz], coeff * 2 * (N0.z - N1.z), 0);
		else if (thread == 5) //n1.z
			atomicAdd(&grad[f1 + I.startNz], coeff * 2 * (N1.z - N0.z), 0);
	}
	__device__ void gradient2Kernel(
		double* grad,
		const double* X,
		const int thread,
		const double w2,
		const unsigned int fi,
		const Cuda::indices I)
	{
		if (fi >= I.num_faces)
			return;
		double3 N = make_double3(
			X[fi + I.startNx],
			X[fi + I.startNy],
			X[fi + I.startNz]
		);
		double coeff = w2 * 4 * (squared_norm(N) - 1);
		if (thread == 0) //N.x
			atomicAdd(&grad[fi + I.startNx], coeff * N.x, 0);
		else if (thread == 1) //N.y
			atomicAdd(&grad[fi + I.startNy], coeff * N.y, 0);
		else if (thread == 2) //N.z
			atomicAdd(&grad[fi + I.startNz], coeff * N.z, 0);
	}
	__device__ void gradient3Kernel(
		double* grad,
		const int3* restShapeF,
		const double* X,
		const unsigned int fi,
		const int thread,
		const double w3,
		const Cuda::indices I)
	{
		if (fi >= I.num_faces)
			return;
		const unsigned int x0 = restShapeF[fi].x;
		const unsigned int x1 = restShapeF[fi].y;
		const unsigned int x2 = restShapeF[fi].z;
		double3 V0 = make_double3(
			X[x0 + I.startVx],
			X[x0 + I.startVy],
			X[x0 + I.startVz]
		);
		double3 V1 = make_double3(
			X[x1 + I.startVx],
			X[x1 + I.startVy],
			X[x1 + I.startVz]
		);
		double3 V2 = make_double3(
			X[x2 + I.startVx],
			X[x2 + I.startVy],
			X[x2 + I.startVz]
		);
		double3 N = make_double3(
			X[fi + I.startNx],
			X[fi + I.startNy],
			X[fi + I.startNz]
		);
		double3 e21 = sub(V2, V1);
		double3 e10 = sub(V1, V0);
		double3 e02 = sub(V0, V2);
		double N02 = dot(N, e02);
		double N10 = dot(N, e10);
		double N21 = dot(N, e21);
		double coeff = 2 * w3;

		if (thread == 0) //x0
			atomicAdd(&grad[x0 + I.startVx], coeff * N.x * (N02 - N10), 0);
		else if (thread == 1) //y0
			atomicAdd(&grad[x0 + I.startVy], coeff * N.y * (N02 - N10), 0);
		else if (thread == 2) //z0
			atomicAdd(&grad[x0 + I.startVz], coeff * N.z * (N02 - N10), 0);
		else if (thread == 3) //x1
			atomicAdd(&grad[x1 + I.startVx], coeff * N.x * (N10 - N21), 0);
		else if (thread == 4) //y1
			atomicAdd(&grad[x1 + I.startVy], coeff * N.y * (N10 - N21), 0);
		else if (thread == 5) //z1
			atomicAdd(&grad[x1 + I.startVz], coeff * N.z * (N10 - N21), 0);
		else if (thread == 6) //x2
			atomicAdd(&grad[x2 + I.startVx], coeff * N.x * (N21 - N02), 0);
		else if (thread == 7) //y2
			atomicAdd(&grad[x2 + I.startVy], coeff * N.y * (N21 - N02), 0);
		else if (thread == 8) //z2
			atomicAdd(&grad[x2 + I.startVz], coeff * N.z * (N21 - N02), 0);
		else if (thread == 9) //Nx
			atomicAdd(&grad[fi + I.startNx], coeff * (N10 * e10.x + N21 * e21.x + N02 * e02.x), 0);
		else if (thread == 10) //Ny
			atomicAdd(&grad[fi + I.startNy], coeff * (N10 * e10.y + N21 * e21.y + N02 * e02.y), 0);
		else if (thread == 11) //Nz
			atomicAdd(&grad[fi + I.startNz], coeff * (N10 * e10.z + N21 * e21.z + N02 * e02.z), 0);
	}

	__global__ void gradientKernel(
		double* grad,
		const double* X,
		const Cuda::hinge* hinges_faceIndex,
		const int3* restShapeF,
		const double* restAreaPerHinge,
		const double* weightPerHinge,
		const double planarParameter,
		const FunctionType functionType,
		const double w1,
		const double w2,
		const double w3,
		const Cuda::indices mesh_indices)
	{
		int Bl_index = blockIdx.x;
		int Th_Index = threadIdx.x;
		//0	,..., F-1,		==> Call Energy(3)
		//F	,..., 2F-1,		==> Call Energy(2)
		//2F,..., 2F+h-1	==> Call Energy(1)
		if (Bl_index < mesh_indices.num_faces) {
			gradient3Kernel(
				grad,
				restShapeF,
				X,
				Bl_index,
				Th_Index,
				w3,
				mesh_indices);
		}
		else if (Bl_index < (2 * mesh_indices.num_faces)) {
			gradient2Kernel(
				grad,
				X,
				Th_Index,
				w2,
				(Bl_index - mesh_indices.num_faces),
				mesh_indices);
		}
		else if (Bl_index < (2 * mesh_indices.num_faces + mesh_indices.num_hinges)) {
			gradient1Kernel(
				grad,
				X,
				hinges_faceIndex,
				restAreaPerHinge,
				weightPerHinge,
				planarParameter,
				functionType,
				w1,
				Bl_index - (2 * mesh_indices.num_faces),
				Th_Index,
				mesh_indices);
		}
	}
}

void Cuda_AuxBendingNormal::value(Cuda::Array<double>& curr_x) {
	const unsigned int s = mesh_indices.num_hinges + 2 * mesh_indices.num_faces;
	Utils_Cuda_AuxBendingNormal::setZeroKernel << <1, 1>> > (EnergyAtomic.cuda_arr);
	Utils_Cuda_AuxBendingNormal::EnergyKernel<1024> << <ceil(s / (double)1024), 1024>> > (
		EnergyAtomic.cuda_arr,
		w1, w2, w3,
		curr_x.cuda_arr,
		restShapeF.cuda_arr,
		restAreaPerHinge.cuda_arr,
		weightPerHinge.cuda_arr,
		hinges_faceIndex.cuda_arr,
		planarParameter,
		functionType,
		mesh_indices);
}

void Cuda_AuxBendingNormal::gradient(Cuda::Array<double>& X)
{
	Utils_Cuda_AuxBendingNormal::setZeroKernel << <grad.size, 1,0,stream_gradient >> > (grad.cuda_arr);
	Utils_Cuda_AuxBendingNormal::gradientKernel << <mesh_indices.num_hinges + 2 * mesh_indices.num_faces, 12, 0, stream_gradient >> > (
		grad.cuda_arr,
		X.cuda_arr,
		hinges_faceIndex.cuda_arr,
		restShapeF.cuda_arr,
		restAreaPerHinge.cuda_arr,
		weightPerHinge.cuda_arr,
		planarParameter, functionType,
		w1, w2, w3, mesh_indices);
}

Cuda_AuxBendingNormal::Cuda_AuxBendingNormal() {
	cudaStreamCreate(&stream_value);
	cudaStreamCreate(&stream_gradient);
}

Cuda_AuxBendingNormal::~Cuda_AuxBendingNormal() {
	cudaStreamDestroy(stream_value);
	cudaStreamDestroy(stream_gradient);
	cudaGetErrorString(cudaGetLastError());
	FreeMemory(restShapeF);
	FreeMemory(grad);
	FreeMemory(restAreaPerFace);
	FreeMemory(restAreaPerHinge);
	FreeMemory(weightPerHinge);
	FreeMemory(EnergyAtomic);
	FreeMemory(hinges_faceIndex);
	FreeMemory(x0_GlobInd);
	FreeMemory(x1_GlobInd);
	FreeMemory(x2_GlobInd);
	FreeMemory(x3_GlobInd);
	FreeMemory(x0_LocInd);
	FreeMemory(x1_LocInd);
	FreeMemory(x2_LocInd);
	FreeMemory(x3_LocInd);
}

