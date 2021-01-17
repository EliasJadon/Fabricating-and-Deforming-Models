#include "Cuda_SDenergy.cuh"
#include "Cuda_Minimizer.cuh"

namespace Utils_Cuda_SDenergy {
	__device__ double3 sub(const double3 a, const double3 b)
	{
		return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
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
	template<int N> __device__ void multiply(
		double3 mat1,
		double mat2[3][N],
		double res[N])
	{
		for (int i = 0; i < N; i++) {
			res[i] = mat1.x * mat2[0][i] + mat1.y * mat2[1][i] + mat1.z * mat2[2][i];
		}
	}
	template<int N> __device__ void multiply(
		double4 mat1,
		double mat2[4][N],
		double res[N])
	{
		for (int i = 0; i < N; i++) {
			res[i] =
				mat1.x * mat2[0][i] +
				mat1.y * mat2[1][i] +
				mat1.z * mat2[2][i] +
				mat1.w * mat2[3][i];
		}
	}
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
		double* Energy,
		const double* curr_x,
		const int3* restShapeF,
		const double* restShapeArea,
		const double3* D1d,
		const double3* D2d,
		const Cuda::indices I)
	{
		//init data
		extern __shared__ double energy_value[blockSize];
		unsigned int tid = threadIdx.x;
		unsigned int fi = blockIdx.x * blockSize + tid;
		energy_value[tid] = 0;

		if (fi < I.num_faces) {
			int v0_index = restShapeF[fi].x;
			int v1_index = restShapeF[fi].y;
			int v2_index = restShapeF[fi].z;
			double3 V0 = make_double3(
				curr_x[v0_index + I.startVx],
				curr_x[v0_index + I.startVy],
				curr_x[v0_index + I.startVz]
			);
			double3 V1 = make_double3(
				curr_x[v1_index + I.startVx],
				curr_x[v1_index + I.startVy],
				curr_x[v1_index + I.startVz]
			);
			double3 V2 = make_double3(
				curr_x[v2_index + I.startVx],
				curr_x[v2_index + I.startVy],
				curr_x[v2_index + I.startVz]
			);
			double3 e10 = sub(V1, V0);
			double3 e20 = sub(V2, V0);
			double3 B1 = normalize(e10);
			double3 B2 = normalize(cross(cross(B1, e20), B1));
			double3 Xi = make_double3(dot(V0, B1), dot(V1, B1), dot(V2, B1));
			double3 Yi = make_double3(dot(V0, B2), dot(V1, B2), dot(V2, B2));
			//prepare jacobian		
			const double a = dot(D1d[fi], Xi);
			const double b = dot(D1d[fi], Yi);
			const double c = dot(D2d[fi], Xi);
			const double d = dot(D2d[fi], Yi);
			const double detJ = a * d - b * c;
			const double detJ2 = detJ * detJ;
			const double a2 = a * a;
			const double b2 = b * b;
			const double c2 = c * c;
			const double d2 = d * d;
			Energy[fi] = 0.5 * restShapeArea[fi] * (1 + 1 / detJ2) * (a2 + b2 + c2 + d2);
			energy_value[tid] = Energy[fi];
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
		const double bulkModulus)
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
		//multiply<3, 2, 2>(dx, dxInv, F);

		//compute the Green Strain = 1/2 * (F'F-I)
		double strain[2][2];
		//multiplyTranspose<2, 3, 2>(F, F, strain);
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
		//multiply<1, 4, 6>(dE_dJ, dstrain_dF, temp);
		//multiply<1, 6, 9>(temp, dF_dX, dE_dX);

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
	
void Cuda_SDenergy::value(Cuda::Array<double>& curr_x)
{
	Utils_Cuda_SDenergy::setZeroKernel << <1, 1>> > (EnergyAtomic.cuda_arr);
	unsigned int s = mesh_indices.num_faces;
	Utils_Cuda_SDenergy::EnergyKernel<1024> << <ceil(s / (double)1024), 1024 >> > (
		EnergyAtomic.cuda_arr,
		Energy.cuda_arr,
		curr_x.cuda_arr,
		restShapeF.cuda_arr,
		restShapeArea.cuda_arr,
		D1d.cuda_arr,
		D2d.cuda_arr,
		mesh_indices);
}

		

void Cuda_SDenergy::gradient(Cuda::Array<double>& X)
{
	/*Utils_Cuda_SDenergy::setZeroKernel << <grad.size, 1, 0, stream_gradient >> > (grad.cuda_arr);
	unsigned int s = mesh_indices.num_faces;
	Utils_Cuda_SDenergy::gradientKernel<1024> << <ceil(s / (double)1024), 1024, 0, stream_gradient >> > (
		grad.cuda_arr,
		X.cuda_arr,
		restShapeF.cuda_arr,
		restShapeArea.cuda_arr,
		dXInv.cuda_arr,
		mesh_indices,
		shearModulus,
		bulkModulus);*/
}

Cuda_SDenergy::Cuda_SDenergy(const int F, const int V) {
	cudaStreamCreate(&stream_value);
	cudaStreamCreate(&stream_gradient);
	Cuda::AllocateMemory(Energy			, F);
	Cuda::AllocateMemory(D1d			, F);
	Cuda::AllocateMemory(D2d			, F);
	Cuda::AllocateMemory(restShapeF		, F);
	Cuda::AllocateMemory(restShapeArea	, F);
	Cuda::initIndices(mesh_indices		, F, V, 0);
	Cuda::AllocateMemory(grad			, 3 * V + 10 * F);
	Cuda::AllocateMemory(EnergyAtomic	, 1);
}

Cuda_SDenergy::~Cuda_SDenergy() {
	cudaStreamDestroy(stream_value);
	cudaStreamDestroy(stream_gradient);
	cudaGetErrorString(cudaGetLastError());
	FreeMemory(restShapeArea);
	FreeMemory(Energy);
	FreeMemory(D1d);
	FreeMemory(D2d);
	FreeMemory(grad);
	FreeMemory(restShapeF);
	FreeMemory(EnergyAtomic);
}
