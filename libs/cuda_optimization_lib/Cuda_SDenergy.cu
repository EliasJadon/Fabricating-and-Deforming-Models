#include "Cuda_SDenergy.cuh"
#include "Cuda_Minimizer.cuh"

namespace Utils_Cuda_SDenergy {
	__device__ double3 sub(const double3 a, const double3 b)
	{
		return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
	}
	__device__ double dot4(const double4 a, const double4 b)
	{
		return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
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
			Energy[fi] = 0.5 * (1 + 1 / detJ2) * (a2 + b2 + c2 + d2);
			energy_value[tid] = restShapeArea[fi] * Energy[fi];
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
		const double3* D1d,
		const double3* D2d,
		const Cuda::indices I)
	{
		unsigned int fi = blockIdx.x;// *blockSize + threadIdx.x;
		if (fi >= I.num_faces)
			return;
		const int v0_index = restShapeF[fi].x;
		const int v1_index = restShapeF[fi].y;
		const int v2_index = restShapeF[fi].z;
		const int startX = I.startVx;
		const int startY = I.startVy;
		const int startZ = I.startVz;
		double3 V0 = make_double3(
			X[v0_index + startX],
			X[v0_index + startY],
			X[v0_index + startZ]
		);
		double3 V1 = make_double3(
			X[v1_index + startX],
			X[v1_index + startY],
			X[v1_index + startZ]
		);
		double3 V2 = make_double3(
			X[v2_index + startX],
			X[v2_index + startY],
			X[v2_index + startZ]
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
		const double det2 = pow(detJ, 2);
		const double a2 = pow(a, 2);
		const double b2 = pow(b, 2);
		const double c2 = pow(c, 2);
		const double d2 = pow(d, 2);
		const double det3 = pow(detJ, 3);
		const double Fnorm = a2 + b2 + c2 + d2;

		double4 de_dJ = make_double4(
			restShapeArea[fi] * (a + a / det2 - d * Fnorm / det3),
			restShapeArea[fi] * (b + b / det2 + c * Fnorm / det3),
			restShapeArea[fi] * (c + c / det2 + b * Fnorm / det3),
			restShapeArea[fi] * (d + d / det2 - a * Fnorm / det3)
		);
		double Norm_e10_3 = pow(norm(e10), 3);
		double3 B2_b2 = cross(cross(e10, e20), e10);
		double Norm_B2 = norm(B2_b2);
		double Norm_B2_2 = pow(Norm_B2, 2);
		double3 B2_dxyz0, B2_dxyz1;
		double B2_dnorm0, B2_dnorm1;
		double3 db1_dX, db2_dX, XX, YY;
		double4 dj_dx;



		if (threadIdx.x == 0) {
			B2_dxyz0 = make_double3(-e10.y * e20.y - e10.z * e20.z, 2 * e10.x * e20.y - e10.y * e20.x, -e10.z * e20.x + 2 * e10.x * e20.z);
			B2_dxyz1 = make_double3(pow(e10.y, 2) + pow(e10.z, 2), -e10.x * e10.y, -e10.x * e10.z);
			B2_dnorm0 = dot(B2_b2, B2_dxyz0) / Norm_B2;
			B2_dnorm1 = dot(B2_b2, B2_dxyz1) / Norm_B2;
			db2_dX = make_double3(
				-((B2_dxyz0.x * Norm_B2 - B2_b2.x * B2_dnorm0) / Norm_B2_2) - ((B2_dxyz1.x * Norm_B2 - B2_b2.x * B2_dnorm1) / Norm_B2_2),
				-((B2_dxyz0.y * Norm_B2 - B2_b2.y * B2_dnorm0) / Norm_B2_2) - ((B2_dxyz1.y * Norm_B2 - B2_b2.y * B2_dnorm1) / Norm_B2_2),
				-((B2_dxyz0.z * Norm_B2 - B2_b2.z * B2_dnorm0) / Norm_B2_2) - ((B2_dxyz1.z * Norm_B2 - B2_b2.z * B2_dnorm1) / Norm_B2_2)
			);
			db1_dX = make_double3((-(pow(e10.y, 2) + pow(e10.z, 2)) / Norm_e10_3), ((e10.y * e10.x) / Norm_e10_3), ((e10.z * e10.x) / Norm_e10_3));
			XX = make_double3(dot(V0, db1_dX) + B1.x, dot(V1, db1_dX), dot(V2, db1_dX));
			YY = make_double3(dot(V0, db2_dX) + B2.x, dot(V1, db2_dX), dot(V2, db2_dX));
			dj_dx = make_double4(
				dot(D1d[fi], XX),
				dot(D1d[fi], YY),
				dot(D2d[fi], XX),
				dot(D2d[fi], YY)
			);
			atomicAdd(&(grad[v0_index + I.startVx]), dot4(de_dJ, dj_dx), 0);
		}
		else if (threadIdx.x == 1) {
			B2_dxyz0 = make_double3(-e10.y * e20.y - e10.z * e20.z, 2 * e10.x * e20.y - e10.y * e20.x, -e10.z * e20.x + 2 * e10.x * e20.z);
			B2_dnorm0 = dot(B2_b2, B2_dxyz0) / Norm_B2;
			db2_dX = make_double3(
				(B2_dxyz0.x * Norm_B2 - B2_b2.x * B2_dnorm0) / Norm_B2_2,
				(B2_dxyz0.y * Norm_B2 - B2_b2.y * B2_dnorm0) / Norm_B2_2,
				(B2_dxyz0.z * Norm_B2 - B2_b2.z * B2_dnorm0) / Norm_B2_2
			);
			db1_dX = make_double3(-(-(pow(e10.y, 2) + pow(e10.z, 2)) / Norm_e10_3), -((e10.y * e10.x) / Norm_e10_3), -((e10.z * e10.x) / Norm_e10_3));
			XX = make_double3(dot(V0, db1_dX), dot(V1, db1_dX) + B1.x, dot(V2, db1_dX));
			YY = make_double3(dot(V0, db2_dX), dot(V1, db2_dX) + B2.x, dot(V2, db2_dX));
			dj_dx = make_double4(
				dot(D1d[fi], XX),
				dot(D1d[fi], YY),
				dot(D2d[fi], XX),
				dot(D2d[fi], YY)
			);
			atomicAdd(&(grad[v1_index + I.startVx]), dot4(de_dJ, dj_dx), 0);
		}
		else if (threadIdx.x == 2) {
			B2_dxyz0 = make_double3(pow(e10.y, 2) + pow(e10.z, 2), -e10.x * e10.y, -e10.x * e10.z);
			B2_dnorm0 = dot(B2_b2, B2_dxyz0) / Norm_B2;
			db2_dX = make_double3(
				(B2_dxyz0.x * Norm_B2 - B2_b2.x * B2_dnorm0) / Norm_B2_2,
				(B2_dxyz0.y * Norm_B2 - B2_b2.y * B2_dnorm0) / Norm_B2_2,
				(B2_dxyz0.z * Norm_B2 - B2_b2.z * B2_dnorm0) / Norm_B2_2
			);
			XX = make_double3(0, 0, B1.x);
			YY = make_double3(dot(V0, db2_dX), dot(V1, db2_dX), dot(V2, db2_dX) + B2.x);
			dj_dx = make_double4(
				dot(D1d[fi], XX),
				dot(D1d[fi], YY),
				dot(D2d[fi], XX),
				dot(D2d[fi], YY)
			);
			atomicAdd(&(grad[v2_index + I.startVx]), dot4(de_dJ, dj_dx), 0);
		}
		else if (threadIdx.x == 3) {
			B2_dxyz0 = make_double3(-e10.x * e20.y + 2 * e10.y * e20.x, -e10.z * e20.z - e20.x * e10.x, 2 * e10.y * e20.z - e10.z * e20.y);
			B2_dxyz1 = make_double3(-e10.y * e10.x, pow(e10.z, 2) + pow(e10.x, 2), -e10.z * e10.y);
			B2_dnorm0 = dot(B2_b2, B2_dxyz0) / Norm_B2;
			B2_dnorm1 = dot(B2_b2, B2_dxyz1) / Norm_B2;
			db2_dX = make_double3(
				-((B2_dxyz0.x * Norm_B2 - B2_b2.x * B2_dnorm0) / Norm_B2_2) - ((B2_dxyz1.x * Norm_B2 - B2_b2.x * B2_dnorm1) / Norm_B2_2),
				-((B2_dxyz0.y * Norm_B2 - B2_b2.y * B2_dnorm0) / Norm_B2_2) - ((B2_dxyz1.y * Norm_B2 - B2_b2.y * B2_dnorm1) / Norm_B2_2),
				-((B2_dxyz0.z * Norm_B2 - B2_b2.z * B2_dnorm0) / Norm_B2_2) - ((B2_dxyz1.z * Norm_B2 - B2_b2.z * B2_dnorm1) / Norm_B2_2)
			);
			db1_dX = make_double3(((e10.y * e10.x) / Norm_e10_3), (-(pow(e10.x, 2) + pow(e10.z, 2)) / Norm_e10_3), ((e10.z * e10.y) / Norm_e10_3));
			XX = make_double3(dot(V0, db1_dX) + B1.y, dot(V1, db1_dX), dot(V2, db1_dX));
			YY = make_double3(dot(V0, db2_dX) + B2.y, dot(V1, db2_dX), dot(V2, db2_dX));
			dj_dx = make_double4(
				dot(D1d[fi], XX),
				dot(D1d[fi], YY),
				dot(D2d[fi], XX),
				dot(D2d[fi], YY)
			);
			atomicAdd(&(grad[v0_index + I.startVy]), dot4(de_dJ, dj_dx), 0);
		}
		else if (threadIdx.x == 4) {
			B2_dxyz0 = make_double3(-e10.x * e20.y + 2 * e10.y * e20.x, -e10.z * e20.z - e20.x * e10.x, 2 * e10.y * e20.z - e10.z * e20.y);
			B2_dnorm0 = dot(B2_b2, B2_dxyz0) / Norm_B2;
			db2_dX = make_double3(
				(B2_dxyz0.x * Norm_B2 - B2_b2.x * B2_dnorm0) / Norm_B2_2,
				(B2_dxyz0.y * Norm_B2 - B2_b2.y * B2_dnorm0) / Norm_B2_2,
				(B2_dxyz0.z * Norm_B2 - B2_b2.z * B2_dnorm0) / Norm_B2_2
			);
			db1_dX = make_double3(-((e10.y * e10.x) / Norm_e10_3), -(-(pow(e10.x, 2) + pow(e10.z, 2)) / Norm_e10_3), -((e10.z * e10.y) / Norm_e10_3));
			XX = make_double3(dot(V0, db1_dX), dot(V1, db1_dX) + B1.y, dot(V2, db1_dX));
			YY = make_double3(dot(V0, db2_dX), dot(V1, db2_dX) + B2.y, dot(V2, db2_dX));
			dj_dx = make_double4(
				dot(D1d[fi], XX),
				dot(D1d[fi], YY),
				dot(D2d[fi], XX),
				dot(D2d[fi], YY)
			);
			atomicAdd(&(grad[v1_index + I.startVy]), dot4(de_dJ, dj_dx), 0);
		}
		else if (threadIdx.x == 5) {
			B2_dxyz0 = make_double3(-e10.y * e10.x, pow(e10.z, 2) + pow(e10.x, 2), -e10.z * e10.y);
			B2_dnorm0 = dot(B2_b2, B2_dxyz0) / Norm_B2;
			db2_dX = make_double3(
				(B2_dxyz0.x * Norm_B2 - B2_b2.x * B2_dnorm0) / Norm_B2_2,
				(B2_dxyz0.y * Norm_B2 - B2_b2.y * B2_dnorm0) / Norm_B2_2,
				(B2_dxyz0.z * Norm_B2 - B2_b2.z * B2_dnorm0) / Norm_B2_2
			);
			XX = make_double3(0, 0, B1.y);
			YY = make_double3(dot(V0, db2_dX), dot(V1, db2_dX), dot(V2, db2_dX) + B2.y);
			dj_dx = make_double4(
				dot(D1d[fi], XX),
				dot(D1d[fi], YY),
				dot(D2d[fi], XX),
				dot(D2d[fi], YY)
			);
			atomicAdd(&(grad[v2_index + I.startVy]), dot4(de_dJ, dj_dx), 0);
		}
		else if (threadIdx.x == 6) {
			B2_dxyz0 = make_double3(2 * e10.z * e20.x - e10.x * e20.z, -e10.y * e20.z + 2 * e10.z * e20.y, -e10.x * e20.x - e10.y * e20.y);
			B2_dxyz1 = make_double3(-e10.x * e10.z, -e10.z * e10.y, pow(e10.x, 2) + pow(e10.y, 2));
			B2_dnorm0 = dot(B2_b2, B2_dxyz0) / Norm_B2;
			B2_dnorm1 = dot(B2_b2, B2_dxyz1) / Norm_B2;
			db2_dX = make_double3(
				-((B2_dxyz0.x * Norm_B2 - B2_b2.x * B2_dnorm0) / Norm_B2_2) - ((B2_dxyz1.x * Norm_B2 - B2_b2.x * B2_dnorm1) / Norm_B2_2),
				-((B2_dxyz0.y * Norm_B2 - B2_b2.y * B2_dnorm0) / Norm_B2_2) - ((B2_dxyz1.y * Norm_B2 - B2_b2.y * B2_dnorm1) / Norm_B2_2),
				-((B2_dxyz0.z * Norm_B2 - B2_b2.z * B2_dnorm0) / Norm_B2_2) - ((B2_dxyz1.z * Norm_B2 - B2_b2.z * B2_dnorm1) / Norm_B2_2)
			);
			db1_dX = make_double3(((e10.z * e10.x) / Norm_e10_3), ((e10.z * e10.y) / Norm_e10_3), (-(pow(e10.x, 2) + pow(e10.y, 2)) / Norm_e10_3));
			XX = make_double3(dot(V0, db1_dX) + B1.z, dot(V1, db1_dX), dot(V2, db1_dX));
			YY = make_double3(dot(V0, db2_dX) + B2.z, dot(V1, db2_dX), dot(V2, db2_dX));
			dj_dx = make_double4(
				dot(D1d[fi], XX),
				dot(D1d[fi], YY),
				dot(D2d[fi], XX),
				dot(D2d[fi], YY)
			);
			atomicAdd(&(grad[v0_index + I.startVz]), dot4(de_dJ, dj_dx), 0);
		}
		else if (threadIdx.x == 7) {
			B2_dxyz0 = make_double3(2 * e10.z * e20.x - e10.x * e20.z, -e10.y * e20.z + 2 * e10.z * e20.y, -e10.x * e20.x - e10.y * e20.y);
			B2_dnorm0 = dot(B2_b2, B2_dxyz0) / Norm_B2;
			db2_dX = make_double3(
				(B2_dxyz0.x * Norm_B2 - B2_b2.x * B2_dnorm0) / Norm_B2_2,
				(B2_dxyz0.y * Norm_B2 - B2_b2.y * B2_dnorm0) / Norm_B2_2,
				(B2_dxyz0.z * Norm_B2 - B2_b2.z * B2_dnorm0) / Norm_B2_2
			);
			db1_dX = make_double3(-((e10.z * e10.x) / Norm_e10_3), -((e10.z * e10.y) / Norm_e10_3), -(-(pow(e10.x, 2) + pow(e10.y, 2)) / Norm_e10_3));
			XX = make_double3(dot(V0, db1_dX), dot(V1, db1_dX) + B1.z, dot(V2, db1_dX));
			YY = make_double3(dot(V0, db2_dX), dot(V1, db2_dX) + B2.z, dot(V2, db2_dX));
			dj_dx = make_double4(
				dot(D1d[fi], XX),
				dot(D1d[fi], YY),
				dot(D2d[fi], XX),
				dot(D2d[fi], YY)
			);
			atomicAdd(&(grad[v1_index + I.startVz]), dot4(de_dJ, dj_dx), 0);
		}
		else if (threadIdx.x == 8) {
			B2_dxyz0 = make_double3(-e10.x * e10.z, -e10.z * e10.y, pow(e10.x, 2) + pow(e10.y, 2));
			B2_dnorm0 = dot(B2_b2, B2_dxyz0) / Norm_B2;
			db2_dX = make_double3(
				(B2_dxyz0.x * Norm_B2 - B2_b2.x * B2_dnorm0) / Norm_B2_2,
				(B2_dxyz0.y * Norm_B2 - B2_b2.y * B2_dnorm0) / Norm_B2_2,
				(B2_dxyz0.z * Norm_B2 - B2_b2.z * B2_dnorm0) / Norm_B2_2
			);
			XX = make_double3(0, 0, B1.z);
			YY = make_double3(dot(V0, db2_dX), dot(V1, db2_dX), dot(V2, db2_dX) + B2.z);
			dj_dx = make_double4(
				dot(D1d[fi], XX),
				dot(D1d[fi], YY),
				dot(D2d[fi], XX),
				dot(D2d[fi], YY)
			);
			atomicAdd(&(grad[v2_index + I.startVz]), dot4(de_dJ, dj_dx), 0);
		}
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
	Utils_Cuda_SDenergy::setZeroKernel << <grad.size, 1>> > (grad.cuda_arr);
	Utils_Cuda_SDenergy::gradientKernel<1024> << <mesh_indices.num_faces, 9>> > (
		grad.cuda_arr,
		X.cuda_arr,
		restShapeF.cuda_arr,
		restShapeArea.cuda_arr,
		D1d.cuda_arr,
		D2d.cuda_arr,
		mesh_indices);
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
	Cuda::AllocateMemory(grad			, 3 * V + 7 * F);
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

