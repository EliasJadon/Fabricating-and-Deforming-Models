#include "Cuda_AuxCylinder.cuh"
#include "Cuda_Minimizer.cuh"

namespace Utils_Cuda_AuxCylinder {
	template<typename T>
	__global__ void setZeroKernel(T* vec)
	{
		vec[blockIdx.x] = 0;
	}
	__device__ double Phi(
		const double x,
		const double planarParameter,
		const FunctionType functionType)
	{
		if (functionType == FunctionType::SIGMOID) {
			double x2 = pow(x, 2);
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
		const FunctionType functionType)
	{
		if (functionType == FunctionType::SIGMOID)
			return (2 * x * planarParameter) / pow(x * x + planarParameter, 2);
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
		const double planarParameter,
		const FunctionType functionType,
		const int hi,
		const Cuda::indices I)
	{
		int f0 = hinges_faceIndex[hi].f0;
		int f1 = hinges_faceIndex[hi].f1;
		if ((f0 >= I.num_faces) || (f1 >= I.num_faces))
			return;
		double3 A0 = make_double3(
			curr_x[f0 + I.startAx],
			curr_x[f0 + I.startAy],
			curr_x[f0 + I.startAz]
		);
		double3 A1 = make_double3(
			curr_x[f1 + I.startAx],
			curr_x[f1 + I.startAy],
			curr_x[f1 + I.startAz]
		);
		double3 C0 = make_double3(
			curr_x[f0 + I.startCx],
			curr_x[f0 + I.startCy],
			curr_x[f0 + I.startCz]
		);
		double3 C1 = make_double3(
			curr_x[f1 + I.startCx],
			curr_x[f1 + I.startCy],
			curr_x[f1 + I.startCz]
		);
		double R0 = curr_x[f0 + I.startR];
		double R1 = curr_x[f1 + I.startR];

		double d_cylinder_dir = pow(pow(dot(A1, A0), 2) - 1, 2);
		double d_center0 = pow(pow(dot(normalize(sub(C1, C0)), A0), 2) - 1, 2);
		double d_center1 = pow(pow(dot(sub(C1, C0), A1), 2) - 1, 2);
		double d_radius = pow(R1 - R0, 2);
		return w1 * restAreaPerHinge[hi] *
			Phi(d_cylinder_dir + d_center0 + d_center1 + d_radius, planarParameter, functionType);
	}
	__device__ double Energy2Kernel(
		const double w2,
		const double* curr_x,
		const int fi,
		const Cuda::indices I)
	{
		if (fi >= I.num_faces)
			return;
		double3 A = make_double3(
			curr_x[fi + I.startAx],
			curr_x[fi + I.startAy],
			curr_x[fi + I.startAz]
		);
		return pow(squared_norm(A) - 1, 2) * w2;
	}
	__device__ double Energy3Kernel(
		const double w3,
		const int3* restShapeF,
		const double* curr_x,
		const int fi,
		const Cuda::indices I)
	{
		// (N*A)^2
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
		double3 C = make_double3(
			curr_x[fi + I.startCx],
			curr_x[fi + I.startCy],
			curr_x[fi + I.startCz]
		);
		double3 A = make_double3(
			curr_x[fi + I.startAx],
			curr_x[fi + I.startAy],
			curr_x[fi + I.startAz]
		);
		double R = curr_x[fi + I.startR];
		double E3 =
			pow(squared_norm(cross(sub(V0, C), A)) - pow(R, 2), 2) +
			pow(squared_norm(cross(sub(V1, C), A)) - pow(R, 2), 2) +
			pow(squared_norm(cross(sub(V2, C), A)) - pow(R, 2), 2);
		return w3 * E3;
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
		double3 A0 = make_double3(
			X[f0 + I.startAx],
			X[f0 + I.startAy],
			X[f0 + I.startAz]
		);
		double3 A1 = make_double3(
			X[f1 + I.startAx],
			X[f1 + I.startAy],
			X[f1 + I.startAz]
		);
		double3 C0 = make_double3(
			X[f0 + I.startCx],
			X[f0 + I.startCy],
			X[f0 + I.startCz]
		);
		double3 C1 = make_double3(
			X[f1 + I.startCx],
			X[f1 + I.startCy],
			X[f1 + I.startCz]
		);
		double R0 = X[f0 + I.startR];
		double R1 = X[f1 + I.startR];

		double d_cylinder_dir = pow(pow(dot(A1, A0), 2) - 1, 2);
		double d_center0 = pow(pow(dot(normalize(sub(C1, C0)), A0), 2) - 1, 2);
		double d_center1 = pow(pow(dot(sub(C1, C0), A1), 2) - 1, 2);
		double d_radius = pow(R1 - R0, 2);
		double coeff = w1 * restAreaPerHinge[hi] *
			dPhi_dm(d_cylinder_dir + d_center0 + d_center1 + d_radius, planarParameter, functionType);
		
		double cylinder_coeff = 4 * coeff * (pow(dot(A1, A0), 2) - 1) * dot(A1, A0);
		double center0_coeff = 4 * coeff * (pow(dot(normalize(sub(C1, C0)), A0), 2) - 1) * (dot(normalize(sub(C1, C0)), A0));
		double center1_coeff = 4 * coeff * (pow(dot(sub(C1, C0), A1), 2) - 1) * (dot(sub(C1, C0), A1));

		double3 diffC = sub(C1, C0);
		double norm1 = norm(diffC);
		double norm2 = norm1 * norm1;
		double norm3 = norm2 * norm1;
		double3 C1C0 = normalize(diffC);

		if (thread == 0) //A0.x;
			atomicAdd(&grad[f0 + I.startAx],
				cylinder_coeff * A1.x +
				center0_coeff * C1C0.x,
				0);
		else if (thread == 1) //A1.x
			atomicAdd(&grad[f1 + I.startAx],
				cylinder_coeff * A0.x +
				center1_coeff * (C1.x - C0.x),
				0);
		else if (thread == 2) //A0.y
			atomicAdd(&grad[f0 + I.startAy],
				cylinder_coeff * A1.y +
				center0_coeff * C1C0.y,
				0);
		else if (thread == 3) //A1.y
			atomicAdd(&grad[f1 + I.startAy],
				cylinder_coeff * A0.y +
				center1_coeff * (C1.y - C0.y),
				0);
		else if (thread == 4) //A0.z
			atomicAdd(&grad[f0 + I.startAz],
				cylinder_coeff * A1.z +
				center0_coeff * C1C0.z,
				0);
		else if (thread == 5) //A1.z
			atomicAdd(&grad[f1 + I.startAz],
				cylinder_coeff * A0.z +
				center1_coeff * (C1.z - C0.z),
				0);
		else if (thread == 6) //R0
			atomicAdd(&grad[f0 + I.startR], coeff * 2 * (R0 - R1), 0);
		else if (thread == 7) //R1
			atomicAdd(&grad[f1 + I.startR], coeff * 2 * (R1 - R0), 0);

		else if (thread == 8) //C0.x
			atomicAdd(&grad[f0 + I.startCx],
				-(center1_coeff * A1.x)
				+ center0_coeff *
				(
					-(A0.x / norm1) + ((diffC.x * diffC.x * A0.x) / norm3)
					+ ((diffC.x * diffC.y * A0.y) / norm3)
					+ ((diffC.x * diffC.z * A0.z) / norm3)
					),
				0);
		else if (thread == 9) //C1.x
			atomicAdd(&grad[f1 + I.startCx],  
				center1_coeff * A1.x
				- center0_coeff *
				(
					-(A0.x / norm1) + ((diffC.x * diffC.x * A0.x) / norm3)
					+ ((diffC.x * diffC.y * A0.y) / norm3)
					+ ((diffC.x * diffC.z * A0.z) / norm3)
					),
				0);
		else if (thread == 10) //C0.y
			atomicAdd(&grad[f0 + I.startCy],  
				-center1_coeff * A1.y
				+ center0_coeff *
				(
					-(A0.y / norm1) + ((diffC.y * diffC.y * A0.y) / norm3)
					+ ((diffC.y * diffC.x * A0.x) / norm3)
					+ ((diffC.y * diffC.z * A0.z) / norm3)
					),
				0);
		else if (thread == 11) //C1.y
			atomicAdd(&grad[f1 + I.startCy], 
				center1_coeff * A1.y
				- center0_coeff *
				(
					-(A0.y / norm1) + ((diffC.y * diffC.y * A0.y) / norm3)
					+ ((diffC.y * diffC.x * A0.x) / norm3)
					+ ((diffC.y * diffC.z * A0.z) / norm3)
					),
				0);
		else if (thread == 12) //C0.z
			atomicAdd(&grad[f0 + I.startCz], 
				-center1_coeff * A1.z
				+ center0_coeff *
				(
					-(A0.z / norm1) + ((diffC.z * diffC.z * A0.z) / norm3)
					+ ((diffC.z * diffC.x * A0.x) / norm3)
					+ ((diffC.z * diffC.y * A0.y) / norm3)
					),
				0);
		else if (thread == 13) //C1.z
			atomicAdd(&grad[f1 + I.startCz], 
				center1_coeff * A1.z
				- center0_coeff *
				(
					-(A0.z / norm1) + ((diffC.z * diffC.z * A0.z) / norm3)
					+ ((diffC.z * diffC.x * A0.x) / norm3)
					+ ((diffC.z * diffC.y * A0.y) / norm3)
					),
				0);
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
		double3 A = make_double3(
			X[fi + I.startAx],
			X[fi + I.startAy],
			X[fi + I.startAz]
		);
		double coeff = w2 * 4 * (squared_norm(A) - 1);
		if (thread == 0) //N.x
			atomicAdd(&grad[fi + I.startAx], coeff * A.x, 0);
		else if (thread == 1) //N.y
			atomicAdd(&grad[fi + I.startAy], coeff * A.y, 0);
		else if (thread == 2) //N.z
			atomicAdd(&grad[fi + I.startAz], coeff * A.z, 0);
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
		int x0 = restShapeF[fi].x;
		int x1 = restShapeF[fi].y;
		int x2 = restShapeF[fi].z;
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
		double3 C = make_double3(
			X[fi + I.startCx],
			X[fi + I.startCy],
			X[fi + I.startCz]
		);
		double3 A = make_double3(
			X[fi + I.startAx],
			X[fi + I.startAy],
			X[fi + I.startAz]
		);
		double R = X[fi + I.startR];
		double3 crossV0 = cross(sub(V0, C), A);
		double3 crossV1 = cross(sub(V1, C), A);
		double3 crossV2 = cross(sub(V2, C), A);
		double dV0 = w3 * 4 * (squared_norm(crossV0) - pow(R, 2));
		double dV1 = w3 * 4 * (squared_norm(crossV1) - pow(R, 2));
		double dV2 = w3 * 4 * (squared_norm(crossV2) - pow(R, 2));
			
		if (thread == 0) //V0x
			atomicAdd(&grad[x0 + I.startVx], dV0* (A.y* crossV0.z - A.z * crossV0.y), 0);
		else if (thread == 1) //V0y
			atomicAdd(&grad[x0 + I.startVy], dV0* (A.z* crossV0.x - A.x * crossV0.z), 0);
		else if (thread == 2) //V0z
			atomicAdd(&grad[x0 + I.startVz], dV0* (A.x* crossV0.y - A.y * crossV0.x), 0);

		else if (thread == 3) //V1x
			atomicAdd(&grad[x1 + I.startVx], dV1* (A.y* crossV1.z - A.z * crossV1.y), 0);
		else if (thread == 4) //V1y
			atomicAdd(&grad[x1 + I.startVy], dV1* (A.z* crossV1.x - A.x * crossV1.z), 0);
		else if (thread == 5) //V1z
			atomicAdd(&grad[x1 + I.startVz], dV1* (A.x* crossV1.y - A.y * crossV1.x), 0);

		else if (thread == 6) //V2x
			atomicAdd(&grad[x2 + I.startVx], dV2* (A.y* crossV2.z - A.z * crossV2.y), 0);
		else if (thread == 7) //V2y
			atomicAdd(&grad[x2 + I.startVy], dV2* (A.z* crossV2.x - A.x * crossV2.z), 0);
		else if (thread == 8) //V2z
			atomicAdd(&grad[x2 + I.startVz], dV2* (A.x* crossV2.y - A.y * crossV2.x), 0);

		else if (thread == 9) //Ax
			atomicAdd(&grad[fi + I.startAx],
				dV0* (crossV0.y* (V0.z - C.z) - crossV0.z * (V0.y - C.y)) +
				dV1 * (crossV1.y * (V1.z - C.z) - crossV1.z * (V1.y - C.y)) +
				dV2 * (crossV2.y * (V2.z - C.z) - crossV2.z * (V2.y - C.y)),
				0);
		else if (thread == 10) //Ay
			atomicAdd(&grad[fi + I.startAy],
				dV0* (crossV0.z* (V0.x - C.x) - crossV0.x * (V0.z - C.z)) +
				dV1 * (crossV1.z * (V1.x - C.x) - crossV1.x * (V1.z - C.z)) +
				dV2 * (crossV2.z * (V2.x - C.x) - crossV2.x * (V2.z - C.z)),
				0);
		else if (thread == 11) //Az
			atomicAdd(&grad[fi + I.startAz],
				dV0* (crossV0.x* (V0.y - C.y) - crossV0.y * (V0.x - C.x)) +
				dV1 * (crossV1.x * (V1.y - C.y) - crossV1.y * (V1.x - C.x)) +
				dV2 * (crossV2.x * (V2.y - C.y) - crossV2.y * (V2.x - C.x)),
				0);

		else if (thread == 12) //Cx
			atomicAdd(&grad[fi + I.startCx],
				dV0* (crossV0.y* A.z - crossV0.z * A.y) +
				dV1 * (crossV1.y * A.z - crossV1.z * A.y) +
				dV2 * (crossV2.y * A.z - crossV2.z * A.y),
				0);
		else if (thread == 13) //Cy
			atomicAdd(&grad[fi + I.startCy],
				dV0* (crossV0.z* A.x - crossV0.x * A.z) +
				dV1 * (crossV1.z * A.x - crossV1.x * A.z) +
				dV2 * (crossV2.z * A.x - crossV2.x * A.z),
				0);
		else if (thread == 14) //Cz
			atomicAdd(&grad[fi + I.startCz],
				dV0* (crossV0.x* A.y - crossV0.y * A.x) +
				dV1 * (crossV1.x * A.y - crossV1.y * A.x) +
				dV2 * (crossV2.x * A.y - crossV2.y * A.x),
				0);

		else if (thread == 15) //R
			atomicAdd(&grad[fi + I.startR], -R * (dV0 + dV1 + dV2), 0);
	}

	__global__ void gradientKernel(
		double* grad,
		const double* X,
		const Cuda::hinge* hinges_faceIndex,
		const int3* restShapeF,
		const double* restAreaPerHinge,
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
				planarParameter,
				functionType,
				w1,
				Bl_index - (2 * mesh_indices.num_faces),
				Th_Index,
				mesh_indices);
		}
	}
}

void Cuda_AuxCylinder::value(Cuda::Array<double>& curr_x) {
	const unsigned int s = mesh_indices.num_hinges + 2 * mesh_indices.num_faces;
	Utils_Cuda_AuxCylinder::setZeroKernel << <1, 1>> > (EnergyAtomic.cuda_arr);
	Utils_Cuda_AuxCylinder::EnergyKernel<1024> << <ceil(s / (double)1024), 1024>> > (
		EnergyAtomic.cuda_arr,
		w1, w2, w3,
		curr_x.cuda_arr,
		restShapeF.cuda_arr,
		restAreaPerHinge.cuda_arr,
		hinges_faceIndex.cuda_arr,
		planarParameter,
		functionType,
		mesh_indices);
}

void Cuda_AuxCylinder::gradient(Cuda::Array<double>& X)
{
	Utils_Cuda_AuxCylinder::setZeroKernel << <grad.size, 1,0,stream_gradient >> > (grad.cuda_arr);
	Utils_Cuda_AuxCylinder::gradientKernel << <mesh_indices.num_hinges + 2 * mesh_indices.num_faces, 16, 0, stream_gradient >> > (
		grad.cuda_arr,
		X.cuda_arr,
		hinges_faceIndex.cuda_arr,
		restShapeF.cuda_arr,
		restAreaPerHinge.cuda_arr,
		planarParameter, functionType,
		w1, w2, w3, mesh_indices);
}

Cuda_AuxCylinder::Cuda_AuxCylinder() {
	cudaStreamCreate(&stream_value);
	cudaStreamCreate(&stream_gradient);
}

Cuda_AuxCylinder::~Cuda_AuxCylinder() {
	cudaStreamDestroy(stream_value);
	cudaStreamDestroy(stream_gradient);
	cudaGetErrorString(cudaGetLastError());
	FreeMemory(restShapeF);
	FreeMemory(grad);
	FreeMemory(restAreaPerFace);
	FreeMemory(restAreaPerHinge);
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

