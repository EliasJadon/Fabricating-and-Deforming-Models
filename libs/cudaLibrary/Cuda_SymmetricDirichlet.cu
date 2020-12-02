#include "Cuda_SymmetricDirichlet.cuh"
#include "Cuda_Minimizer.cuh"

namespace Cuda {
	namespace SymmetricDirichlet {
		Array<double> grad, EnergyAtomic;
		Array<double3> restShapeV;
		unsigned int num_faces, num_vertices;
		
		
		
		__device__ double3 add(double3 a, double3 b)
		{
			return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
		}
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








		
		
		double value() {
			return 1;
		}

		
		void local_basis(
			const double3* V,	
			const int3* F,		
			double3* B1,		
			double3* B2,		
			const unsigned int num_faces
		)
		{
			int face_index = threadIdx.x + blockIdx.x * blockDim.x;
			if (face_index < num_faces)
			{
				unsigned int x0 = F[face_index].x;
				unsigned int x1 = F[face_index].y;
				unsigned int x2 = F[face_index].z;
				double3 v1 = normalize(sub(V[x1], V[x0]));
				double3 t = sub(V[x2], V[x0]);
				double3 v3 = normalize(cross(v1, t));
				double3 v2 = normalize(cross(v1, v3));
				B1[face_index] = v1;
				B2[face_index] = mul(-1, v2);
			}
		}

		void gradient()
		{
			
		}
		
		void FreeAllVariables() {
			cudaGetErrorString(cudaGetLastError());
			FreeMemory(restShapeV);
			FreeMemory(grad);
			FreeMemory(EnergyAtomic);
		}
	}
}
