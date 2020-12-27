#include "Cuda_STVK.cuh"
#include "Cuda_Minimizer.cuh"


namespace Utils_Cuda_STVK {
	__device__ double3 sub(const double3 a, const double3 b)
	{
		return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
	}
	template<typename T>
	__global__ void setZeroKernel(T* vec)
	{
		vec[blockIdx.x] = 0;
	}
	template<int R1, int C1_R2, int C2> 
	__device__ void multiply(
		double mat1[R1][C1_R2],
		double mat2[C1_R2][C2],
		double res[R1][C2])
	{
		int i, j, k;
		for (i = 0; i < R1; i++) {
			for (j = 0; j < C2; j++) {
				res[i][j] = 0;
				for (k = 0; k < C1_R2; k++)
					res[i][j] += mat1[i][k] * mat2[k][j];
			}
		}
	}
	template<int R1, int C1_R2, int C2> 
	__device__ void multiplyTranspose(
		double mat1[C1_R2][R1],
		double mat2[C1_R2][C2],
		double res[R1][C2])
	{
		int i, j, k;
		for (i = 0; i < R1; i++) {
			for (j = 0; j < C2; j++) {
				res[i][j] = 0;
				for (k = 0; k < C1_R2; k++)
					res[i][j] += mat1[k][i] * mat2[k][j];
			}
		}
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
		*resAtomic = 0;

		__syncthreads();

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
		else {
			energy_value[tid] = 0;
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
		const int3* restShapeF,
		const double* restShapeArea,
		const double4* dXInv,
		const Cuda::indices I,
		const double shearModulus,
		const double bulkModulus,
		const unsigned int size)
	{
		unsigned int fi = blockIdx.x * blockSize + threadIdx.x;
		if (fi >= I.num_faces)
			return;
		const double4 dxinv = dXInv[fi];
		const int startX = I.startVx;
		const int startY = I.startVy;
		const int startZ = I.startVz;
		const double Area = restShapeArea[fi];
		const unsigned int v0i = restShapeF[fi].x;
		const unsigned int v1i = restShapeF[fi].y;
		const unsigned int v2i = restShapeF[fi].z;
		double3 V0 = make_double3(
			X[v0i + startX],
			X[v0i + startY],
			X[v0i + startZ]
		);
		double3 V1 = make_double3(
			X[v1i + startX],
			X[v1i + startY],
			X[v1i + startZ]
		);
		double3 V2 = make_double3(
			X[v2i + startX],
			X[v2i + startY],
			X[v2i + startZ]
		);

		double3 e10 = sub(V1, V0);
		double3 e20 = sub(V2, V0);
		double dx[3][2];
		dx[0][0] = e10.x; dx[0][1] = e20.x;
		dx[1][0] = e10.y; dx[1][1] = e20.y;
		dx[2][0] = e10.z; dx[2][1] = e20.z;

		double F[3][2];
		double dxInv[2][2];
		dxInv[0][0] = dxinv.x;
		dxInv[0][1] = dxinv.y;
		dxInv[1][0] = dxinv.z;
		dxInv[1][1] = dxinv.w;
		multiply<3, 2, 2>(dx, dxInv, F);

		//compute the Green Strain = 1/2 * (F'F-I)
		double strain[2][2];
		multiplyTranspose<2, 3, 2>(F, F, strain);
		strain[0][0] -= 1; strain[1][1] -= 1;
		strain[0][0] *= 0.5;
		strain[0][1] *= 0.5;
		strain[1][0] *= 0.5;
		strain[1][1] *= 0.5;

		double dF_dX[6][9] = { 0 };
		dF_dX[0][0] = -dxinv.x - dxinv.z;
		dF_dX[0][1] = dxinv.x;
		dF_dX[0][2] = dxinv.z;

		dF_dX[1][0] = -dxinv.y - dxinv.w;
		dF_dX[1][1] = dxinv.y;
		dF_dX[1][2] = dxinv.w;

		dF_dX[2][3] = -dxinv.x - dxinv.z;
		dF_dX[2][4] = dxinv.x;
		dF_dX[2][5] = dxinv.z;

		dF_dX[3][3] = -dxinv.y - dxinv.w;
		dF_dX[3][4] = dxinv.y;
		dF_dX[3][5] = dxinv.w;

		dF_dX[4][6] = -dxinv.x - dxinv.z;
		dF_dX[4][7] = dxinv.x;
		dF_dX[4][8] = dxinv.z;

		dF_dX[5][6] = -dxinv.y - dxinv.w;
		dF_dX[5][7] = dxinv.y;
		dF_dX[5][8] = dxinv.w;

		double dstrain_dF[4][6] = { 0 };
		dstrain_dF[0][0] = F[0][0];
		dstrain_dF[0][2] = F[1][0];
		dstrain_dF[0][4] = F[2][0];

		dstrain_dF[1][0] = 0.5 * F[0][1];
		dstrain_dF[1][1] = 0.5 * F[0][0];
		dstrain_dF[1][2] = 0.5 * F[1][1];
		dstrain_dF[1][3] = 0.5 * F[1][0];
		dstrain_dF[1][4] = 0.5 * F[2][1];
		dstrain_dF[1][5] = 0.5 * F[2][0];

		dstrain_dF[2][0] = 0.5 * F[0][1];
		dstrain_dF[2][1] = 0.5 * F[0][0];
		dstrain_dF[2][2] = 0.5 * F[1][1];
		dstrain_dF[2][3] = 0.5 * F[1][0];
		dstrain_dF[2][4] = 0.5 * F[2][1];
		dstrain_dF[2][5] = 0.5 * F[2][0];

		dstrain_dF[3][1] = F[0][1];
		dstrain_dF[3][3] = F[1][1];
		dstrain_dF[3][5] = F[2][1];

		double dE_dJ[1][4];
		dE_dJ[0][0] = Area * (2 * shearModulus * strain[0][0] + bulkModulus * (strain[0][0] + strain[1][1]));
		dE_dJ[0][1] = Area * (2 * shearModulus * strain[0][1]);
		dE_dJ[0][2] = Area * (2 * shearModulus * strain[1][0]);
		dE_dJ[0][3] = Area * (2 * shearModulus * strain[1][1] + bulkModulus * (strain[0][0] + strain[1][1]));

		double dE_dX[1][9];
		double temp[1][6];
		multiply<1, 4, 6>(dE_dJ, dstrain_dF, temp);
		multiply<1, 6, 9>(temp, dF_dX, dE_dX);

		atomicAdd(&(grad[v0i + startX]), dE_dX[0][0], 0);
		atomicAdd(&(grad[v1i + startX]), dE_dX[0][1], 0);
		atomicAdd(&(grad[v2i + startX]), dE_dX[0][2], 0);
		atomicAdd(&(grad[v0i + startY]), dE_dX[0][3], 0);
		atomicAdd(&(grad[v1i + startY]), dE_dX[0][4], 0);
		atomicAdd(&(grad[v2i + startY]), dE_dX[0][5], 0);
		atomicAdd(&(grad[v0i + startZ]), dE_dX[0][6], 0);
		atomicAdd(&(grad[v1i + startZ]), dE_dX[0][7], 0);
		atomicAdd(&(grad[v2i + startZ]), dE_dX[0][8], 0);
	}
}
	
double Cuda_STVK::value(Cuda::Array<double>& curr_x) {
	/*unsigned int s = 3 * num_vertices;
	Utils_Cuda_STVK::EnergyKernel<1024> << <ceil(s / (double)1024), 1024 >> > (
		EnergyAtomic.cuda_arr,
		curr_x.cuda_arr,
		restShapeV.cuda_arr,
		num_vertices);
	Cuda::CheckErr(cudaDeviceSynchronize());
	MemCpyDeviceToHost(EnergyAtomic);*/
	return EnergyAtomic.host_arr[0];
}

		

Cuda::Array<double>* Cuda_STVK::gradient(Cuda::Array<double>& X)
{
	Utils_Cuda_STVK::setZeroKernel << <grad.size, 1 >> > (grad.cuda_arr);
	Cuda::CheckErr(cudaDeviceSynchronize());

	unsigned int s = mesh_indices.num_faces;
	Utils_Cuda_STVK::gradientKernel<1024> << <ceil(s / (double)1024), 1024 >> > (
		grad.cuda_arr,
		X.cuda_arr,
		restShapeF.cuda_arr,
		restShapeArea.cuda_arr,
		dXInv.cuda_arr,
		mesh_indices,
		shearModulus,
		bulkModulus,
		grad.size);
	Cuda::CheckErr(cudaDeviceSynchronize());
	return &grad;
}

Cuda_STVK::Cuda_STVK(){

}

Cuda_STVK::~Cuda_STVK() {
	cudaGetErrorString(cudaGetLastError());
	FreeMemory(restShapeArea);
	FreeMemory(grad);
	FreeMemory(restShapeF);
	FreeMemory(EnergyAtomic);
	FreeMemory(dXInv);
}

