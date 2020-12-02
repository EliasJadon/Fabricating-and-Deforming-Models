#include "Cuda_AuxBendingNormal.cuh"
#include "Cuda_Minimizer.cuh"


namespace Cuda {
	namespace AuxBendingNormal {
		//dynamic variables
		double w1 = 1, w2 = 100, w3 = 100;
		FunctionType functionType;
		double planarParameter;
		Array<double> grad;
		//help variables - dynamic
		Array<double> EnergyAtomic;
		
		//Static variables
		Array<int3> restShapeF;
		Array<double> restAreaPerFace, restAreaPerHinge; //Eigen::VectorXd
		int num_hinges, num_faces, num_vertices;
		Array<hinge> hinges_faceIndex; //std::vector<Eigen::Vector2d> //num_hinges*2
		Array<int> x0_GlobInd, x1_GlobInd, x2_GlobInd, x3_GlobInd; //Eigen::VectorXi //num_hinges
		Array<hinge> x0_LocInd, x1_LocInd, x2_LocInd, x3_LocInd; //Eigen::MatrixXi //num_hinges*2
		
		
		__device__ double3 addVectors(double3 a, double3 b)
		{
			double3 result;
			result.x = a.x + b.x;
			result.y = a.y + b.y;
			result.z = a.z + b.z;
			return result;
		}
		__device__ double3 subVectors(const double3 a, const double3 b)
		{
			double3 result;
			result.x = a.x - b.x;
			result.y = a.y - b.y;
			result.z = a.z - b.z;
			return result;
		}
		__device__ double mulVectors(const double3 a, const double3 b)
		{
			return
				a.x * b.x +
				a.y * b.y +
				a.z * b.z;
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
			const hinge* hinges_faceIndex,
			const double* restAreaPerHinge,
			const double planarParameter,
			const FunctionType functionType,
			const int hi,
			const int num_faces,
			const int startN)
		{
			int f0 = hinges_faceIndex[hi].f0;
			int f1 = hinges_faceIndex[hi].f1;
			int startNy = startN + num_faces;
			int startNz = startNy + num_faces;
			double3 N0;
			N0.x = curr_x[f0 + startN];
			N0.y = curr_x[f0 + startNy];
			N0.z = curr_x[f0 + startNz];
			double3 N1;
			N1.x = curr_x[f1 + startN];
			N1.y = curr_x[f1 + startNy];
			N1.z = curr_x[f1 + startNz];
			double3 diff = subVectors(N1, N0);
			double d_normals = mulVectors(diff, diff);
			
			double res;
			if (functionType == FunctionType::SIGMOID) {
				double x2 = d_normals * d_normals;
				res = x2 / (x2 + planarParameter);
			}
			else if (functionType == FunctionType::QUADRATIC)
				res = d_normals * d_normals;
			else if (functionType == FunctionType::EXPONENTIAL)
				res = 0; //TODO: add exponential option
			return res* restAreaPerHinge[hi];
		}

		__device__ double Energy2Kernel(
			const double w2,
			const double* curr_x,
			const int fi,
			const int num_faces,
			const int startN)
		{
			double Nx = curr_x[fi + startN];
			double Ny = curr_x[fi + startN + num_faces];
			double Nz = curr_x[fi + startN + 2 * num_faces];
			double x2 = Nx * Nx;
			double y2 = Ny * Ny;
			double z2 = Nz * Nz;
			double sqrN = x2 + y2 + z2 - 1;
			return sqrN * sqrN * w2;
		}
		__device__ double Energy3Kernel(
			const double w3,
			const int3* restShapeF,
			const double* curr_x,
			const int fi,
			const int num_faces,
			const int num_vertices,
			const int startN)
		{
			// (N^T*(x1-x0))^2 + (N^T*(x2-x1))^2 + (N^T*(x0-x2))^2
			int x0 = restShapeF[fi].x;
			int x1 = restShapeF[fi].y;
			int x2 = restShapeF[fi].z;

			int num_vertices2 = 2 * num_vertices;
			double3 V0,V1,V2;
			V0.x = curr_x[x0];
			V0.y = curr_x[x0 + num_vertices];
			V0.z = curr_x[x0 + num_vertices2];
			V1.x = curr_x[x1];
			V1.y = curr_x[x1 + num_vertices];
			V1.z = curr_x[x1 + num_vertices2];
			V2.x = curr_x[x2];
			V2.y = curr_x[x2 + num_vertices];
			V2.z = curr_x[x2 + num_vertices2];

			double3 N;
			int startNi = fi + startN;
			N.x = curr_x[startNi];
			N.y = curr_x[startNi + num_faces];
			N.z = curr_x[startNi + 2 * num_faces];

			double3 e21 = subVectors(V2, V1);
			double3 e10 = subVectors(V1, V0);
			double3 e02 = subVectors(V0, V2);
			double d1 = mulVectors(N, e21);
			double d2 = mulVectors(N, e10);
			double d3 = mulVectors(N, e02);
			double res = d1 * d1 + d2 * d2 + d3 * d3;
			res *= w3;
			return res;
		}

		template<unsigned int blockSize>
		__global__ void EnergyKernel(
			double* resAtomic,
			const double w1,
			const double w2,
			const double w3,
			const double* curr_x,
			const int3 * restShapeF,
			const double * restAreaPerHinge,
			const hinge* hinges_faceIndex,
			const double planarParameter,
			const FunctionType functionType,
			const int num_hinges,
			const int num_faces,
			const int num_vertices,
			const int startN) 
		{
			extern __shared__ double energy_value[blockSize];
			unsigned int tid = threadIdx.x;
			unsigned int Global_idx = blockIdx.x * blockSize + tid;
			*resAtomic = 0;
			__syncthreads();

			//0	,..., F-1,		==> Call Energy(3)
			//F	,..., 2F-1,		==> Call Energy(2)
			//2F,..., 2F+h-1	==> Call Energy(1)
			if (Global_idx < num_faces) {
				energy_value[tid] = Energy3Kernel(
					w3,
					restShapeF,
					curr_x,
					Global_idx,
					num_faces,
					num_vertices,
					startN);
			}
			else if (Global_idx < (2*num_faces)) {
				energy_value[tid] = Energy2Kernel(
					w2,
					curr_x,
					Global_idx - num_faces,
					num_faces,
					startN);
			}
			else if (Global_idx < ((2 * num_faces) + num_hinges)) {
				energy_value[tid] = Energy1Kernel(
					w1,
					curr_x,
					hinges_faceIndex,
					restAreaPerHinge,
					planarParameter,
					functionType,
					Global_idx - (2 * num_faces),
					num_faces,
					startN);
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
		
		double value() {
			unsigned int s = num_hinges + num_faces + num_faces;
			EnergyKernel<1024> << <ceil(s / (double)1024), 1024 >> > (
				EnergyAtomic.cuda_arr,
				w1, w2, w3,
				Cuda::Minimizer::curr_x.cuda_arr,
				restShapeF.cuda_arr,
				restAreaPerHinge.cuda_arr,
				hinges_faceIndex.cuda_arr,
				planarParameter,
				functionType,
				num_hinges,
				num_faces,
				num_vertices,
				3 * num_vertices);
			CheckErr(cudaDeviceSynchronize());
			MemCpyDeviceToHost(EnergyAtomic);
			return EnergyAtomic.host_arr[0];
		}

		__device__ double dPhi_dm(
			const double x, 
			const double planarParameter,
			const FunctionType functionType) 
		{
			if (functionType == FunctionType::SIGMOID)
				return (2 * x * planarParameter) / pow(x * x + planarParameter, 2);
			else if (functionType == FunctionType::QUADRATIC)
				return 2 * x;
			else if (functionType == FunctionType::EXPONENTIAL)
				return 0; //TODO: add exponential
		}

		__device__ void gradient1Kernel(
			double* grad,
			const double* curr_x,
			const hinge* hinges_faceIndex,
			const double* restAreaPerHinge,
			const double planarParameter,
			const FunctionType functionType,
			const double w1,
			const int hi,
			const int thread,
			const int num_faces,
			const int num_vertices,
			const int startN)
		{
			int f0 = hinges_faceIndex[hi].f0;
			int f1 = hinges_faceIndex[hi].f1;
			int startNy = startN + num_faces;
			int startNz = startNy + num_faces;

			int startN0x = f0 + startN;
			int startN0y = f0 + startNy;
			int startN0z = f0 + startNz;
			int startN1x = f1 + startN;
			int startN1y = f1 + startNy;
			int startN1z = f1 + startNz;
			double3 N0;
			N0.x = curr_x[startN0x];
			N0.y = curr_x[startN0y];
			N0.z = curr_x[startN0z];
			double3 N1;
			N1.x = curr_x[startN1x];
			N1.y = curr_x[startN1y];
			N1.z = curr_x[startN1z];
			double3 diff = subVectors(N1, N0);
			double d_normals = mulVectors(diff, diff);

			double coeff = w1 * restAreaPerHinge[hi] * dPhi_dm(d_normals, planarParameter, functionType);

			if (thread == 0) //n0.x;
				atomicAdd(&grad[startN0x], coeff * 2 * (N0.x - N1.x), 0);
			else if (thread == 1) //n1.x
				atomicAdd(&grad[startN1x], coeff * 2 * (N1.x - N0.x), 0);
			else if (thread == 2) //n0.y
				atomicAdd(&grad[startN0y], coeff * 2 * (N0.y - N1.y), 0);
			else if (thread == 3) //n1.y
				atomicAdd(&grad[startN1y], coeff * 2 * (N1.y - N0.y), 0);
			else if (thread == 4) //n0.z
				atomicAdd(&grad[startN0z], coeff * 2 * (N0.z - N1.z), 0);
			else if (thread == 5) //n1.z
				atomicAdd(&grad[startN1z], coeff * 2 * (N1.z - N0.z), 0);
		}
		__device__ void gradient2Kernel(
			double* grad,
			const double* curr_x,
			const int thread,
			const double w2,
			const int num_faces,
			const int startNi)
		{
			int startNiy = startNi + num_faces;
			int startNiz = startNiy + num_faces;
			double3 N;
			N.x = curr_x[startNi];
			N.y = curr_x[startNiy];
			N.z = curr_x[startNiz];

			double coeff = w2 * 4 * (mulVectors(N, N) - 1);
			if (thread == 0) //N.x
				atomicAdd(&grad[startNi], coeff * N.x, 0);
			else if (thread == 1) //N.y
				atomicAdd(&grad[startNiy], coeff * N.y, 0);
			else if (thread == 2) //N.z
				atomicAdd(&grad[startNiz], coeff * N.z, 0);
		}
		__device__ void gradient3Kernel(
			double* grad,
			const int3* restShapeF,
			const double* curr_x,
			const int fi,
			const int thread,
			const double w3,
			const int num_vertices,
			const int num_faces,
			const int startN)
		{
			int x0 = restShapeF[fi].x;
			int x1 = restShapeF[fi].y;
			int x2 = restShapeF[fi].z;
			int num_vertices2 = 2 * num_vertices;
			double3 V0, V1, V2;
			V0.x = curr_x[x0];
			V0.y = curr_x[x0 + num_vertices];
			V0.z = curr_x[x0 + num_vertices2];
			V1.x = curr_x[x1];
			V1.y = curr_x[x1 + num_vertices];
			V1.z = curr_x[x1 + num_vertices2];
			V2.x = curr_x[x2];
			V2.y = curr_x[x2 + num_vertices];
			V2.z = curr_x[x2 + num_vertices2];

			double3 N;
			int startNi = fi + startN;
			N.x = curr_x[startNi];
			N.y = curr_x[startNi + num_faces];
			N.z = curr_x[startNi + 2 * num_faces];

			double3 e21 = subVectors(V2, V1);
			double3 e10 = subVectors(V1, V0);
			double3 e02 = subVectors(V0, V2);
			double N02 = mulVectors(N, e02);
			double N10 = mulVectors(N, e10);
			double N21 = mulVectors(N, e21);
			double coeff = 2 * w3;
			int num_2verices = 2 * num_vertices;
			int num_3verices_fi = fi + num_2verices + num_vertices;

			switch (thread) {
			case 0: //x0
				atomicAdd(&grad[x0], coeff * N.x * (N02 - N10), 0);
				break;
			case 1: //y0
				atomicAdd(&grad[x0 + num_vertices], coeff * N.y * (N02 - N10), 0);
				break;
			case 2: //z0
				atomicAdd(&grad[x0 + num_2verices], coeff * N.z * (N02 - N10), 0);
				break;
			case 3: //x1
				atomicAdd(&grad[x1], coeff * N.x * (N10 - N21), 0);
				break;
			case 4: //y1
				atomicAdd(&grad[x1 + num_vertices], coeff * N.y * (N10 - N21), 0);
				break;
			case 5: //z1
				atomicAdd(&grad[x1 + num_2verices], coeff * N.z * (N10 - N21), 0);
				break;
			case 6: //x2
				atomicAdd(&grad[x2], coeff * N.x * (N21 - N02), 0);
				break;
			case 7: //y2
				atomicAdd(&grad[x2 + num_vertices], coeff * N.y * (N21 - N02), 0);
				break;
			case 8: //z2
				atomicAdd(&grad[x2 + num_2verices], coeff * N.z * (N21 - N02), 0);
				break;
			case 9: //Nx
				atomicAdd(&grad[num_3verices_fi], coeff * (N10 * e10.x + N21 * e21.x + N02 * e02.x), 0);
				break;
			case 10: //Ny
				atomicAdd(&grad[num_3verices_fi + num_faces], coeff * (N10 * e10.y + N21 * e21.y + N02 * e02.y), 0);
				break;
			case 11: //Nz
				atomicAdd(&grad[num_3verices_fi + (2 * num_faces)], coeff * (N10 * e10.z + N21 * e21.z + N02 * e02.z), 0);
				break;
			}
		}

		__global__ void gradientKernel(
			double* grad,
			const double* curr_x,
			const hinge* hinges_faceIndex,
			const int3* restShapeF,
			const double* restAreaPerHinge,
			const double planarParameter,
			const FunctionType functionType,
			const int num_hinges,
			const int num_faces,
			const int num_vertices,
			const double w1,
			const double w2,
			const double w3)
		{
			int Bl_index = blockIdx.x;
			int Th_Index = threadIdx.x;
			//0	,..., F-1,		==> Call Energy(3)
			//F	,..., 2F-1,		==> Call Energy(2)
			//2F,..., 2F+h-1	==> Call Energy(1)
			if (Bl_index < num_faces) {
				gradient3Kernel(
					grad,
					restShapeF,
					curr_x,
					Bl_index,
					Th_Index,
					w3,
					num_vertices,
					num_faces,
					3 * num_vertices);
			}
			else if (Bl_index < (2 * num_faces)) {
				gradient2Kernel(
					grad, 
					curr_x,  
					Th_Index,
					w2,
					num_faces,
					3 * num_vertices + (Bl_index - num_faces));
			}
			else {
				gradient1Kernel(
					grad,
					curr_x,
					hinges_faceIndex,
					restAreaPerHinge,
					planarParameter,
					functionType,
					w1,
					Bl_index - (2 * num_faces),
					Th_Index,
					num_faces,
					num_vertices,
					3 * num_vertices);
			}
		}

		template<typename T>
		__global__ void setZeroKernel(T* vec)
		{
			vec[blockIdx.x] = 0;
		}

		void gradient()
		{
			setZeroKernel << <grad.size, 1 >> > (grad.cuda_arr);
			CheckErr(cudaDeviceSynchronize());

			gradientKernel << <num_hinges + num_faces + num_faces, 12 >> > (
				grad.cuda_arr,
				Cuda::Minimizer::X.cuda_arr,
				hinges_faceIndex.cuda_arr,
				restShapeF.cuda_arr,
				restAreaPerHinge.cuda_arr,
				planarParameter, functionType,
				num_hinges, num_faces, num_vertices,
				w1, w2, w3);
			CheckErr(cudaDeviceSynchronize());
			//MemCpyDeviceToHost(grad);
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
