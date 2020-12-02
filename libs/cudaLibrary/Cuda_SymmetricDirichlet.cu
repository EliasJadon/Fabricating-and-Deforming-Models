//#include "Cuda_SymmetricDirichlet.cuh"
//#include "Cuda_Minimizer.cuh"
//
//namespace Cuda {
//	namespace SymmetricDirichlet {
//		Array<double> grad, EnergyAtomic;
//		Array<rowVector<double>> restShapeV;
//		unsigned int num_faces, num_vertices;
//		
//		
//		
//		
//		__device__ double atomicAdd(double* address, double val, int flag)
//		{
//			unsigned long long int* address_as_ull =
//				(unsigned long long int*)address;
//			unsigned long long int old = *address_as_ull, assumed;
//
//			do {
//				assumed = old;
//				old = atomicCAS(address_as_ull, assumed,
//					__double_as_longlong(val +
//						__longlong_as_double(assumed)));
//
//				// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
//			} while (assumed != old);
//
//			return __longlong_as_double(old);
//		}
//
//		template <unsigned int blockSize, typename T>
//		__device__ void warpReduce(volatile T* sdata, unsigned int tid) {
//			if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
//			if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
//			if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
//			if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
//			if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
//			if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
//		}
//
//		template<unsigned int blockSize>
//		__global__ void EnergyKernel(
//			double* resAtomic,
//			const double* curr_x,
//			const rowVector<double>* restShapeV,
//			const unsigned int num_vertices) 
//		{
//			//init data
//			extern __shared__ double energy_value[blockSize];
//			unsigned int tid = threadIdx.x;
//			unsigned int Global_idx = blockIdx.x * blockSize + tid;
//			*resAtomic = 0;
//
//			__syncthreads();
//
//			if (Global_idx < num_vertices) {
//				double diff_x = curr_x[Global_idx] - restShapeV[Global_idx].x;
//				energy_value[tid] = diff_x * diff_x;
//			}
//			else if (Global_idx < 2*num_vertices) {
//				unsigned int V_index = Global_idx - num_vertices;
//				double diff_y = curr_x[Global_idx] - restShapeV[V_index].y;
//				energy_value[tid] = diff_y * diff_y;
//			}
//			else if (Global_idx < 3*num_vertices) {
//				unsigned int V_index = Global_idx - 2*num_vertices;
//				double diff_z = curr_x[Global_idx] - restShapeV[V_index].z;
//				energy_value[tid] = diff_z * diff_z;
//			}
//			else {
//				energy_value[tid] = 0;
//			}
//			
//			__syncthreads();
//
//			if (blockSize >= 1024) { if (tid < 512) { energy_value[tid] += energy_value[tid + 512]; } __syncthreads(); }
//			if (blockSize >= 512) { if (tid < 256) { energy_value[tid] += energy_value[tid + 256]; } __syncthreads(); }
//			if (blockSize >= 256) { if (tid < 128) { energy_value[tid] += energy_value[tid + 128]; } __syncthreads(); }
//			if (blockSize >= 128) { if (tid < 64) { energy_value[tid] += energy_value[tid + 64]; } __syncthreads(); }
//			if (tid < 32) warpReduce<blockSize, double>(energy_value, tid);
//			if (tid == 0) atomicAdd(resAtomic, energy_value[0], 0);
//		}
//		
//		double value() {
//			unsigned int s = 3 * num_vertices;
//			EnergyKernel<1024> << <ceil(s / (double)1024), 1024 >> > (
//				EnergyAtomic.cuda_arr,
//				Cuda::Minimizer::curr_x.cuda_arr,
//				restShapeV.cuda_arr,
//				num_vertices);
//			CheckErr(cudaDeviceSynchronize());
//			MemCpyDeviceToHost(EnergyAtomic);
//			return EnergyAtomic.host_arr[0];
//		}
//
//		template<unsigned int blockSize>
//		__global__ void gradientKernel(
//			double* grad,
//			const double* X,
//			const rowVector<double>* restShapeV,
//			const unsigned int num_vertices,
//			const unsigned int size)
//		{
//			unsigned int tid = threadIdx.x;
//			unsigned int Global_idx = blockIdx.x * blockSize + tid;
//			if (Global_idx < num_vertices) {
//				double diff_x = X[Global_idx] - restShapeV[Global_idx].x;
//				grad[Global_idx] = 2 * diff_x; //X-coordinate
//			}
//			else if (Global_idx < 2 * num_vertices) {
//				unsigned int V_index = Global_idx - num_vertices;
//				double diff_y = X[Global_idx] - restShapeV[V_index].y;
//				grad[Global_idx] = 2 * diff_y; //Y-coordinate
//			}
//			else if (Global_idx < 3 * num_vertices) {
//				unsigned int V_index = Global_idx - 2 * num_vertices;
//				double diff_z = X[Global_idx] - restShapeV[V_index].z;
//				grad[Global_idx] = 2 * diff_z; //Z-coordinate
//			}
//			else if (Global_idx < size) {
//				grad[Global_idx] = 0;
//			}
//		}
//
//		template <typename DerivedV, typename DerivedF>
//		void local_basis(
//			const double* V,//MatrixX3d
//			const int* F,	//MatrixX3i***
//			double* B1,		//MatrixX3d
//			double* B2		//MatrixX3d
//		)
//		{
//			B1.resize(F.rows(), 3);
//			B2.resize(F.rows(), 3);
//			
//			for (unsigned i = 0; i < F.rows(); ++i)
//			{
//				double3 v1 = normalize(V.row(F(i, 1)) - V.row(F(i, 0)));
//				double3 t = V.row(F(i, 2)) - V.row(F(i, 0));
//				double3 v3 = normalize(cross(v1, t));
//				double3 v2 = normalize(cross(v1, v3));
//
//				B1.row(i) = v1;
//				B2.row(i) = -v2;
//			}
//		}
//
//		void gradient()
//		{
//			unsigned int s = grad.size;
//			gradientKernel<1024> << <ceil(s / (double)1024), 1024 >> > (
//				grad.cuda_arr,
//				Cuda::Minimizer::X.cuda_arr,
//				restShapeV.cuda_arr,
//				num_vertices,
//				grad.size);
//			CheckErr(cudaDeviceSynchronize());
//			//MemCpyDeviceToHost(grad);
//		}
//		
//		void FreeAllVariables() {
//			cudaGetErrorString(cudaGetLastError());
//			FreeMemory(restShapeV);
//			FreeMemory(grad);
//			FreeMemory(EnergyAtomic);
//		}
//	}
//}
