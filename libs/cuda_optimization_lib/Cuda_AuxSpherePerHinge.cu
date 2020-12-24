#include "Cuda_AuxSpherePerHinge.cuh"
#include "Cuda_Minimizer.cuh"

namespace Utils_Cuda_AuxSpherePerHinge {
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
		double R0 = curr_x[f0 + I.startR];
		double R1 = curr_x[f1 + I.startR];
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
		double d_center = squared_norm(sub(C1, C0));
		double d_radius = pow(R1 - R0, 2);
		return w1 * restAreaPerHinge[hi] *
			Phi(d_center + d_radius, planarParameter, functionType);
	}
	__device__ double Energy2Kernel(
		const double w2,
		const int3* restShapeF,
		const double* curr_x,
		const int fi,
		const Cuda::indices I)
	{
		const unsigned int x0 = restShapeF[fi].x;
		const unsigned int x1 = restShapeF[fi].y;
		const unsigned int x2 = restShapeF[fi].z;
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
		double R = curr_x[fi + I.startR];
		double res =
			pow(squared_norm(sub(V0, C)) - pow(R, 2), 2) +
			pow(squared_norm(sub(V1, C)) - pow(R, 2), 2) +
			pow(squared_norm(sub(V2, C)) - pow(R, 2), 2);
		return w2 * res;
	}

	template<unsigned int blockSize>
	__global__ void EnergyKernel(
		double* resAtomic,
		const double w1,
		const double w2,
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
		*resAtomic = 0;
		__syncthreads();

		//0	,..., F-1,	==> Call Energy(2)
		//F ,..., F+h-1	==> Call Energy(1)

		if (Global_idx < mesh_indices.num_faces) {
			energy_value[tid] = Energy2Kernel(
				w2,
				restShapeF,
				curr_x,
				Global_idx, //fi
				mesh_indices);
		}
		else if (Global_idx < (mesh_indices.num_faces + mesh_indices.num_hinges)) {
			energy_value[tid] = Energy1Kernel(
				w1,
				curr_x,
				hinges_faceIndex,
				restAreaPerHinge,
				planarParameter,
				functionType,
				Global_idx - mesh_indices.num_faces, //hi
				mesh_indices);
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
		if (f0 >= I.num_faces || f1 >= I.num_faces)
			return;
		double R0 = X[f0 + I.startR];
		double R1 = X[f1 + I.startR];
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
		double d_center = squared_norm(sub(C1, C0));
		double d_radius = pow(R1 - R0, 2);
		double coeff = 2 * w1 * restAreaPerHinge[hi] * dPhi_dm(d_center + d_radius, planarParameter, functionType);

		if (thread == 0)
			atomicAdd(&grad[f0 + I.startCx], (C0.x - C1.x) * coeff, 0); //C0.x
		if (thread == 1)
			atomicAdd(&grad[f0 + I.startCy], (C0.y - C1.y) * coeff, 0);	//C0.y
		if (thread == 2)
			atomicAdd(&grad[f0 + I.startCz], (C0.z - C1.z) * coeff, 0);	//C0.z
		if (thread == 3)
			atomicAdd(&grad[f1 + I.startCx], (C1.x - C0.x) * coeff, 0);	//C1.x
		if (thread == 4)
			atomicAdd(&grad[f1 + I.startCy], (C1.y - C0.y) * coeff, 0);	//C1.y
		if (thread == 5)
			atomicAdd(&grad[f1 + I.startCz], (C1.z - C0.z) * coeff, 0);	//C1.z
		if (thread == 6)
			atomicAdd(&grad[f0 + I.startR], (R0 - R1) * coeff, 0);		//r0
		if (thread == 7)
			atomicAdd(&grad[f1 + I.startR], (R1 - R0) * coeff, 0);		//r1
	}
	__device__ void gradient2Kernel(
		double* grad,
		const int3* restShapeF,
		const double* X,
		const unsigned int fi,
		const int thread,
		const double w2,
		const Cuda::indices I)
	{
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
		double3 C = make_double3(
			X[fi + I.startCx],
			X[fi + I.startCy],
			X[fi + I.startCz]
		);
		double R = X[fi + I.startR];

		double coeff = w2 * 4;
		double E0 = coeff * (squared_norm(sub(V0, C)) - pow(R, 2));
		double E1 = coeff * (squared_norm(sub(V1, C)) - pow(R, 2));
		double E2 = coeff * (squared_norm(sub(V2, C)) - pow(R, 2));

		if (thread == 0)
			atomicAdd(&grad[x0 + I.startVx], E0 * (V0.x - C.x), 0); // V0x
		if (thread == 1)
			atomicAdd(&grad[x0 + I.startVy], E0 * (V0.y - C.y), 0); // V0y
		if (thread == 2)
			atomicAdd(&grad[x0 + I.startVz], E0 * (V0.z - C.z), 0); // V0z
		if (thread == 3)
			atomicAdd(&grad[x1 + I.startVx], E1 * (V1.x - C.x), 0); // V1x
		if (thread == 4)
			atomicAdd(&grad[x1 + I.startVy], E1 * (V1.y - C.y), 0); // V1y
		if (thread == 5)
			atomicAdd(&grad[x1 + I.startVz], E1 * (V1.z - C.z), 0); // V1z
		if (thread == 6)
			atomicAdd(&grad[x2 + I.startVx], E2 * (V2.x - C.x), 0); // V2x
		if (thread == 7)
			atomicAdd(&grad[x2 + I.startVy], E2 * (V2.y - C.y), 0); // V2y
		if (thread == 8)
			atomicAdd(&grad[x2 + I.startVz], E2 * (V2.z - C.z), 0); // V2z
		if (thread == 9)
			atomicAdd(&grad[fi + I.startCx],
			(E0 * (C.x - V0.x)) +
				(E1 * (C.x - V1.x)) +
				(E2 * (C.x - V2.x)), 0); // Cx
		if (thread == 10)
			atomicAdd(&grad[fi + I.startCy],
			(E0 * (C.y - V0.y)) +
				(E1 * (C.y - V1.y)) +
				(E2 * (C.y - V2.y)), 0); // Cy
		if (thread == 11)
			atomicAdd(&grad[fi + I.startCz],
			(E0 * (C.z - V0.z)) +
				(E1 * (C.z - V1.z)) +
				(E2 * (C.z - V2.z)), 0); // Cz
		if (thread == 12)
			atomicAdd(&grad[fi + I.startR],
			(E0 * (-1) * R) +
				(E1 * (-1) * R) +
				(E2 * (-1) * R), 0); //r
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
		const Cuda::indices mesh_indices)
	{
		//0	,..., F-1,	==> Call Energy(2)
		//F ,..., F+h-1	==> Call Energy(1)

		if (blockIdx.x < mesh_indices.num_faces)
			gradient2Kernel(
				grad,
				restShapeF,
				X,
				blockIdx.x, //fi
				threadIdx.x,
				w2,
				mesh_indices);
		if (blockIdx.x < (mesh_indices.num_faces + mesh_indices.num_hinges))
			gradient1Kernel(
				grad,
				X,
				hinges_faceIndex,
				restAreaPerHinge,
				planarParameter,
				functionType,
				w1,
				blockIdx.x - mesh_indices.num_faces, //hi
				threadIdx.x,
				mesh_indices);
	}

}


namespace Cuda {
	namespace AuxSpherePerHinge {
		//dynamic variables
		double w1 = 1, w2 = 100;
		FunctionType functionType;
		double planarParameter;
		Array<double> grad;
		//help variables - dynamic
		Array<double> EnergyAtomic;

		//Static variables
		Array<int3> restShapeF;
		indices mesh_indices;
		Array<double> restAreaPerFace, restAreaPerHinge; //Eigen::VectorXd
		Array<hinge> hinges_faceIndex; //std::vector<Eigen::Vector2d> //num_hinges*2
		Array<int> x0_GlobInd, x1_GlobInd, x2_GlobInd, x3_GlobInd; //Eigen::VectorXi //num_hinges
		Array<hinge> x0_LocInd, x1_LocInd, x2_LocInd, x3_LocInd; //Eigen::MatrixXi //num_hinges*2
		
		

		

		double value(Cuda::Array<double>& curr_x) {
			const unsigned int s = mesh_indices.num_hinges + mesh_indices.num_faces;
			Utils_Cuda_AuxSpherePerHinge::EnergyKernel<1024> << <ceil(s / (double)1024), 1024 >> > (
				EnergyAtomic.cuda_arr,
				w1, w2,
				curr_x.cuda_arr,
				restShapeF.cuda_arr,
				restAreaPerHinge.cuda_arr,
				hinges_faceIndex.cuda_arr,
				planarParameter,
				functionType,
				mesh_indices);
			CheckErr(cudaDeviceSynchronize());
			MemCpyDeviceToHost(EnergyAtomic);
			return EnergyAtomic.host_arr[0];
		}
		
		Cuda::Array<double>* gradient(Cuda::Array<double>& X)
		{
			Utils_Cuda_AuxSpherePerHinge::setZeroKernel << <grad.size, 1 >> > (grad.cuda_arr);
			CheckErr(cudaDeviceSynchronize());
			Utils_Cuda_AuxSpherePerHinge::gradientKernel << <mesh_indices.num_hinges + mesh_indices.num_faces, 20 >> > (
				grad.cuda_arr,
				X.cuda_arr,
				hinges_faceIndex.cuda_arr,
				restShapeF.cuda_arr,
				restAreaPerHinge.cuda_arr,
				planarParameter, functionType,
				w1, w2, mesh_indices);
			CheckErr(cudaDeviceSynchronize());
			/*MemCpyDeviceToHost(grad);
			for (int i = 0; i < grad.size; i++) {
				std::cout << i << ":\t" << grad.host_arr[i] << "\n";
			}*/
			return &grad;
		}

		void FreeAllVariables() {
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
		
	}
}