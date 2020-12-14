#include "Cuda_SSSymmetricDirichlet.cuh"
#include "Cuda_Minimizer.cuh"

#define CUDA_EPSILON 0.000000000001 //1e-12
namespace Cuda {
	namespace SSSymmetricDirichlet {
		Array<double> grad, EnergyAtomic, EnergyVec, restShapeArea;
		Array<double3> D1d, D2d;
		Array<int3> restShapeF;
		unsigned int num_faces, num_vertices;

		
		
		__device__ double3 add(double3 a, double3 b)
		{
			return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
		}
		template<unsigned int size>
		__device__ void add(double res[size], const double a[size], const double b[size]) {
			for (int j = 0; j < size; ++j) {
				res[j] = a[j] + b[j];
			}
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
			double n = norm(a);
			if (n < CUDA_EPSILON)
				return a;
			return mul(1.0f / n, a);
		}
		__device__ double3 cross(const double3 a, const double3 b)
		{
			return make_double3(
				a.y * b.z - a.z * b.y,
				a.z * b.x - a.x * b.z,
				a.x * b.y - a.y * b.x
			);
		}

		template<unsigned int row1, unsigned int col1, unsigned int row2, unsigned int col2>
		__device__ void mulMatrix(double res[row1][col2], const double a[row1][col1], const double b[row2][col2]) {
			for (int i = 0; i < row1; ++i)
				for (int j = 0; j < col2; ++j) {
					res[i][j] = 0;
					for (int k = 0; k < col1; ++k)
						res[i][j] += a[i][k] * b[k][j];
				}	
		}
		template<unsigned int col>
		__device__ void mulMatrix(double res[col], const double3 a, const double b[3][col]) {
			// Multiplying matrix a and b and storing in array mult.
			for (int j = 0; j < col; ++j) {
				res[j] = 0;
				res[j] += a.x * b[0][j];
				res[j] += a.y * b[1][j];
				res[j] += a.z * b[2][j];
			}
		}
		template<unsigned int col>
		__device__ void mulMatrix(double res[col], const double4 a, const double b[4][col]) {
			// Multiplying matrix a and b and storing in array mult.
			for (int j = 0; j < col; ++j) {
				res[j] = 0;
				res[j] += a.x * b[0][j];
				res[j] += a.y * b[1][j];
				res[j] += a.z * b[2][j];
				res[j] += a.w * b[3][j];
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








		
		
		

		
		__device__ void local_basis(
			const double3 CurrV0,	
			const double3 CurrV1,	
			const double3 CurrV2,	
			double3& B1,		
			double3& B2)
		{
			double3 v1 = normalize(sub(CurrV1, CurrV0));
			double3 t = sub(CurrV2, CurrV0);
			double3 v3 = normalize(cross(v1, t));
			double3 v2 = normalize(cross(v1, v3));
			B1 = v1;
			B2 = mul(-1, v2);
		}
			   		 	  
		template<unsigned int blockSize>
		__global__ void EnergyKernel(
			double* resAtomic,
			double* Energy,
			const double* X,
			const int3* restShapeF,
			const double* restShapeArea,
			const double3* D1d,
			const double3* D2d,
			const unsigned int num_faces,
			const unsigned int num_vertices)
		{
			extern __shared__ double energy_value[blockSize];
			unsigned int tid = threadIdx.x;
			unsigned int face_index = blockIdx.x * blockSize + tid;
			*resAtomic = 0;
			__syncthreads();

			if (face_index < num_faces) {
				unsigned int x0 = restShapeF[face_index].x;
				unsigned int x1 = restShapeF[face_index].y;
				unsigned int x2 = restShapeF[face_index].z;
				double3 CurrV0 = make_double3(X[x0], X[x0 + num_vertices], X[x0 + 2 * num_vertices]);
				double3 CurrV1 = make_double3(X[x1], X[x1 + num_vertices], X[x1 + 2 * num_vertices]);
				double3 CurrV2 = make_double3(X[x2], X[x2 + num_vertices], X[x2 + 2 * num_vertices]);
				double3 Dx = D1d[face_index];
				double3 Dy = D2d[face_index];
				double area = restShapeArea[face_index];

				double3 B1, B2;
				local_basis(CurrV0, CurrV1, CurrV2, B1, B2);

				double3 Xi = make_double3(
					dot(CurrV0, B1),
					dot(CurrV1, B1),
					dot(CurrV2, B1)
				);
				double3 Yi = make_double3(
					dot(CurrV0, B2),
					dot(CurrV1, B2),
					dot(CurrV2, B2)
				);
				//prepare jacobian		
				double a = dot(Dx, Xi);
				double b = dot(Dx, Yi);
				double c = dot(Dy, Xi);
				double d = dot(Dy, Yi);
				double det = a * d - b * c;

				//till now is updateX
				double det2 = det * det;
				double a2 = a * a;
				double b2 = b * b;
				double c2 = c * c;
				double d2 = d * d;
				double Fnorm = a2 + b2 + c2 + d2;
				//now, value...
				double val = 0.5 * area * (1 + 1 / det2) * Fnorm;
				Energy[face_index] = val;
				energy_value[tid] = val;
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
			EnergyKernel<1024> << <ceil(num_faces / (double)1024), 1024 >> > (
				EnergyAtomic.cuda_arr,
				EnergyVec.cuda_arr,
				Cuda::Minimizer::curr_x.cuda_arr,
				restShapeF.cuda_arr,
				restShapeArea.cuda_arr,
				D1d.cuda_arr,
				D2d.cuda_arr,
				num_faces,
				num_vertices);
			CheckErr(cudaDeviceSynchronize());
			MemCpyDeviceToHost(EnergyAtomic);
			return EnergyAtomic.host_arr[0];
		}









		__device__ void dB1_dX(double g[3][9], const double3 vec10)
		{
			double Norm = norm(vec10);

			double Norm3 = Norm * Norm * Norm;
			if (Norm3 == 0)
				Norm3 = 1;

			double dB1x_dx0 = -(pow(vec10.y, 2) + pow(vec10.z, 2)) / Norm3;
			double dB1y_dy0 = -(pow(vec10.x, 2) + pow(vec10.z, 2)) / Norm3;
			double dB1z_dz0 = -(pow(vec10.x, 2) + pow(vec10.y, 2)) / Norm3;
			double dB1x_dy0 = (vec10.y * vec10.x) / Norm3;
			double dB1x_dz0 = (vec10.z * vec10.x) / Norm3;
			double dB1y_dz0 = (vec10.z * vec10.y) / Norm3;
			
			g[0][0] = dB1x_dx0; 
			g[0][1] = -dB1x_dx0; 
			g[0][2] = 0; 
			g[0][3] = dB1x_dy0;
			g[0][4] = -dB1x_dy0;
			g[0][5] = 0;
			g[0][6] = dB1x_dz0;
			g[0][7] = -dB1x_dz0;
			g[0][8] = 0;
					
			g[1][0] = dB1x_dy0;
			g[1][1] = -dB1x_dy0;
			g[1][2] = 0;
			g[1][3] = dB1y_dy0;
			g[1][4] = -dB1y_dy0; 
			g[1][5] = 0;
			g[1][6] = dB1y_dz0; 
			g[1][7] = -dB1y_dz0;
			g[1][8] = 0;
				
			g[2][0] = dB1x_dz0;
			g[2][1] = -dB1x_dz0;
			g[2][2] = 0;
			g[2][3] = dB1y_dz0;
			g[2][4] = -dB1y_dz0;
			g[2][5] = 0;
			g[2][6] = dB1z_dz0;
			g[2][7] = -dB1z_dz0;
			g[2][8] = 0;
		}

		__device__ void dB2_dX(
			double g[3][9],
			const double3 vec10, 
			const double3 vec20)
		{
			double3 b2 = mul(-1,(cross(vec10,cross(vec10,vec20))));
			double NormB2 = norm(b2);
			double NormB2_2 = pow(NormB2, 2);
			if (NormB2 == 0)
				NormB2 = 1;
			if (NormB2_2 == 0)
				NormB2_2 = 1;

			double3 dxyz[6];
			dxyz[0] = make_double3(
				-vec10.y * vec20.y - vec10.z * vec20.z,
				2 * vec10.x * vec20.y - vec10.y * vec20.x,
				-vec10.z * vec20.x + 2 * vec10.x * vec20.z
			);
			dxyz[1] = make_double3(
				-vec10.x * vec20.y + 2 * vec10.y * vec20.x,
				-vec10.z * vec20.z - vec20.x * vec10.x,
				2 * vec10.y * vec20.z - vec10.z * vec20.y
			);
			dxyz[2] = make_double3(
				2 * vec10.z * vec20.x - vec10.x * vec20.z,
				-vec10.y * vec20.z + 2 * vec10.z * vec20.y,
				-vec10.x * vec20.x - vec10.y * vec20.y
			);
			dxyz[3] = make_double3(
				pow(vec10.y, 2) + pow(vec10.z, 2),
				-vec10.x * vec10.y,
				-vec10.x * vec10.z
			);
			dxyz[4] = make_double3(
				-vec10.y * vec10.x,
				pow(vec10.z, 2) + pow(vec10.x, 2),
				-vec10.z * vec10.y
			);
			dxyz[5] = make_double3(
				-vec10.x * vec10.z,
				-vec10.z * vec10.y,
				pow(vec10.x, 2) + pow(vec10.y, 2)
			);

			double dnorm[6];
			dnorm[0] = dot(b2, dxyz[0]) / NormB2;
			dnorm[1] = dot(b2, dxyz[1]) / NormB2;
			dnorm[2] = dot(b2, dxyz[2]) / NormB2;
			dnorm[3] = dot(b2, dxyz[3]) / NormB2;
			dnorm[4] = dot(b2, dxyz[4]) / NormB2;
			dnorm[5] = dot(b2, dxyz[5]) / NormB2;

			g[0][1] = (dxyz[0].x * NormB2 - b2.x * dnorm[0]) / NormB2_2;
			g[0][2] = (dxyz[3].x * NormB2 - b2.x * dnorm[3]) / NormB2_2;
			g[0][0] = -g[0][1] - g[0][2];
			g[0][4] = (dxyz[1].x * NormB2 - b2.x * dnorm[1]) / NormB2_2;
			g[0][5] = (dxyz[4].x * NormB2 - b2.x * dnorm[4]) / NormB2_2;
			g[0][3] = -g[0][4] - g[0][5];
			g[0][7] = (dxyz[2].x * NormB2 - b2.x * dnorm[2]) / NormB2_2;
			g[0][8] = (dxyz[5].x * NormB2 - b2.x * dnorm[5]) / NormB2_2;
			g[0][6] = -g[0][7] - g[0][8];
			
			g[1][1] = (dxyz[0].y * NormB2 - b2.y * dnorm[0]) / NormB2_2;
			g[1][2] = (dxyz[3].y * NormB2 - b2.y * dnorm[3]) / NormB2_2;
			g[1][0] = -g[1][1] - g[1][2];
			g[1][4] = (dxyz[1].y * NormB2 - b2.y * dnorm[1]) / NormB2_2;
			g[1][5] = (dxyz[4].y * NormB2 - b2.y * dnorm[4]) / NormB2_2;
			g[1][3] = -g[1][4] - g[1][5];
			g[1][7] = (dxyz[2].y * NormB2 - b2.y * dnorm[2]) / NormB2_2;
			g[1][8] = (dxyz[5].y * NormB2 - b2.y * dnorm[5]) / NormB2_2;
			g[1][6] = -g[1][7] - g[1][8];

			g[2][1] = (dxyz[0].z * NormB2 - b2.z * dnorm[0]) / NormB2_2;
			g[2][2] = (dxyz[3].z * NormB2 - b2.z * dnorm[3]) / NormB2_2;
			g[2][0] = -g[2][1] - g[2][2];
			g[2][4] = (dxyz[1].z * NormB2 - b2.z * dnorm[1]) / NormB2_2;
			g[2][5] = (dxyz[4].z * NormB2 - b2.z * dnorm[4]) / NormB2_2;
			g[2][3] = -g[2][4] - g[2][5];
			g[2][7] = (dxyz[2].z * NormB2 - b2.z * dnorm[2]) / NormB2_2;
			g[2][8] = (dxyz[5].z * NormB2 - b2.z * dnorm[5]) / NormB2_2;
			g[2][6] = -g[2][7] - g[2][8];
		}

		__device__ void dJ_dX(
			double dJ[4][9],
			const double3 Dx,
			const double3 Dy,
			const double3 B1,
			const double3 B2,
			const double3 CurrV0,
			const double3 CurrV1,
			const double3 CurrV2)
		{
			double dV0_dX[3][9] = { 0 }, dV1_dX[3][9] = { 0 }, dV2_dX[3][9] = { 0 };
			dV0_dX[0][0] = 1; dV0_dX[1][3] = 1; dV0_dX[2][6] = 1;
			dV1_dX[0][1] = 1; dV1_dX[1][4] = 1; dV1_dX[2][7] = 1;
			dV2_dX[0][2] = 1; dV2_dX[1][5] = 1; dV2_dX[2][8] = 1;

			double YY[3][9], XX[3][9];
			double3 vec10 = sub(CurrV1, CurrV0);
			double3 vec20 = sub(CurrV2, CurrV0);
			double db1_dX[3][9]; dB1_dX(db1_dX, vec10);
			double db2_dX[3][9]; dB2_dX(db2_dX, vec10, vec20);

			double tmp1[9], tmp2[9];

			mulMatrix<9>(tmp1, CurrV0, db1_dX); mulMatrix<9>(tmp2, B1, dV0_dX);
			add<9>(&(XX[0][0]), tmp1, tmp2);
			//XX[0][0:8] = (CurrV0 * db1_dX + B1 * dV0_dX);

			mulMatrix<9>(tmp1, CurrV1, db1_dX); mulMatrix<9>(tmp2, B1, dV1_dX);
			add<9>(&(XX[1][0]), tmp1, tmp2);
			//XX[1][0:8] = (CurrV1 * db1_dX + B1 * dV1_dX);
			
			mulMatrix<9>(tmp1, CurrV2, db1_dX); mulMatrix<9>(tmp2, B1, dV2_dX);
			add<9>(&(XX[2][0]), tmp1, tmp2);
			//XX[2][0:8] = (CurrV2 * db1_dX + B1 * dV2_dX);
			
			mulMatrix<9>(tmp1, CurrV0, db2_dX); mulMatrix<9>(tmp2, B2, dV0_dX);
			add<9>(&(YY[0][0]), tmp1, tmp2);
			//YY[0][0:8] = (CurrV0 * db2_dX + B2 * dV0_dX);
			
			mulMatrix<9>(tmp1, CurrV1, db2_dX); mulMatrix<9>(tmp2, B2, dV1_dX);
			add<9>(&(YY[1][0]), tmp1, tmp2);
			//YY[1][0:8] = (CurrV1 * db2_dX + B2 * dV1_dX);
			
			mulMatrix<9>(tmp1, CurrV2, db2_dX); mulMatrix<9>(tmp2, B2, dV2_dX);
			add<9>(&(YY[2][0]), tmp1, tmp2);
			//YY[2][0:8] = (CurrV2 * db2_dX + B2 * dV2_dX);

			
			mulMatrix<9>(&(dJ[0][0]), Dx, XX);
			//dJ[0][0:8] = Dx * XX;
			mulMatrix<9>(&(dJ[1][0]), Dx, YY);
			//dJ[1][0:8] = Dx * YY;
			mulMatrix<9>(&(dJ[2][0]), Dy, XX);
			//dJ[2][0:8] = Dy * XX;
			mulMatrix<9>(&(dJ[3][0]), Dy, YY);
			//dJ[3][0:8] = Dy * YY;
		}


		template<unsigned int blockSize>
		__global__ void GradientKernel(
			double* grad,
			const double* X,
			const int3* restShapeF,
			const double* restShapeArea,
			const double3* D1d,
			const double3* D2d,
			const unsigned int num_faces,
			const unsigned int num_vertices)
		{
			unsigned int tid = threadIdx.x;
			unsigned int face_index = blockIdx.x * blockSize + tid;
			
			

			if (face_index  == 159/*< num_faces*/) {
				unsigned int x0 = restShapeF[face_index].x;
				unsigned int x1 = restShapeF[face_index].y;
				unsigned int x2 = restShapeF[face_index].z;
				double3 CurrV0 = make_double3(X[x0], X[x0 + num_vertices], X[x0 + 2 * num_vertices]);
				double3 CurrV1 = make_double3(X[x1], X[x1 + num_vertices], X[x1 + 2 * num_vertices]);
				double3 CurrV2 = make_double3(X[x2], X[x2 + num_vertices], X[x2 + 2 * num_vertices]);
				double3 Dx = D1d[face_index];
				double3 Dy = D2d[face_index];
				double area = restShapeArea[face_index];

				double3 B1, B2;
				local_basis(CurrV0, CurrV1, CurrV2, B1, B2);

				double3 Xi = make_double3(
					dot(CurrV0, B1),
					dot(CurrV1, B1),
					dot(CurrV2, B1)
				);
				double3 Yi = make_double3(
					dot(CurrV0, B2),
					dot(CurrV1, B2),
					dot(CurrV2, B2)
				);
				//prepare jacobian		
				double a = dot(Dx, Xi);
				double b = dot(Dx, Yi);
				double c = dot(Dy, Xi);
				double d = dot(Dy, Yi);
				double det = a * d - b * c;
				//till now is updateX
				double det2 = det * det;
				double det3 = det2 * det;
				double a2 = a * a;
				double b2 = b * b;
				double c2 = c * c;
				double d2 = d * d;
				double Fnorm = a2 + b2 + c2 + d2;

				//if (abs(det) < CUDA_EPSILON)
					//	det = 1;
				double4 de_dJ = make_double4(
					area * (a + a /*/ det2*/ - d * Fnorm /*/ det3*/),
					area * (b + b /*/ det2*/ + c * Fnorm /*/ det3*/),
					area * (c + c /*/ det2*/ + b * Fnorm /*/ det3*/),
					area * (d + d /*/ det2*/ - a * Fnorm /*/ det3*/)
				);

				//double dE_dX[9];
				//double dJ[4][9];
				//dJ_dX(dJ, Dx, Dy, B1, B2, CurrV0, CurrV1, CurrV2);

				//mulMatrix<9>(dE_dX, de_dJ, dJ);
				////dE_dX = de_dJ/*1,4*/ * dJ/*4,9*/;
				

				////////////////////////////////////////////////////////////////
				int debug_index = 0;
				grad[debug_index++] = x0;
				grad[debug_index++] = x1;
				grad[debug_index++] = x2;
				grad[debug_index++] = CurrV0.x;
				grad[debug_index++] = CurrV0.y;
				grad[debug_index++] = CurrV0.z;
				grad[debug_index++] = CurrV1.x;
				grad[debug_index++] = CurrV1.y;
				grad[debug_index++] = CurrV1.z;
				grad[debug_index++] = CurrV2.x;
				grad[debug_index++] = CurrV2.y;
				grad[debug_index++] = CurrV2.z;
				grad[debug_index++] = Dx.x;
				grad[debug_index++] = Dx.y;
				grad[debug_index++] = Dx.z;
				grad[debug_index++] = Dy.x;
				grad[debug_index++] = Dy.y;
				grad[debug_index++] = Dy.z;
				grad[debug_index++] = area;
				grad[debug_index++] = B1.x;
				grad[debug_index++] = B1.y;
				grad[debug_index++] = B1.z;
				grad[debug_index++] = B2.x;
				grad[debug_index++] = B2.y;
				
				grad[debug_index++] = B2.z;
				grad[debug_index++] = Xi.x;
				grad[debug_index++] = Xi.y;
				grad[debug_index++] = Xi.z;
				grad[debug_index++] = Yi.x;
				grad[debug_index++] = Yi.y;
				grad[debug_index++] = Yi.z;
				grad[debug_index++] = a;
				grad[debug_index++] = b;
				grad[debug_index++] = c;
				grad[debug_index++] = d;
				grad[debug_index++] = det;
				grad[debug_index++] = det2;
				grad[debug_index++] = det3;
				grad[debug_index++] = a2;
				grad[debug_index++] = b2;
				grad[debug_index++] = c2;
				grad[debug_index++] = d2;
				grad[debug_index++] = Fnorm;

				grad[debug_index++] = de_dJ.x;
				//grad[debug_index++] = de_dJ.y;
				//grad[debug_index++] = de_dJ.z;
				//grad[debug_index++] = de_dJ.w;

				//grad[debug_index++] = dE_dX[0];
				//grad[debug_index++] = dE_dX[1];
				//grad[debug_index++] = dE_dX[2];
				//grad[debug_index++] = dE_dX[3];
				//grad[debug_index++] = dE_dX[4];
				//grad[debug_index++] = dE_dX[5];
				//grad[debug_index++] = dE_dX[6];
				//grad[debug_index++] = dE_dX[7];
				//grad[debug_index++] = dE_dX[8];
				//
				//
				//grad[debug_index++] = dJ[0][0];
				//grad[debug_index++] = dJ[0][1];
				//grad[debug_index++] = dJ[0][2];
				//grad[debug_index++] = dJ[0][3];
				//grad[debug_index++] = dJ[0][4];
				//grad[debug_index++] = dJ[0][5];
				//grad[debug_index++] = dJ[0][6];
				//grad[debug_index++] = dJ[0][7];
				//grad[debug_index++] = dJ[0][8];
				//grad[debug_index++] = dJ[1][0];
				//grad[debug_index++] = dJ[1][1];
				//grad[debug_index++] = dJ[1][2];
				//grad[debug_index++] = dJ[1][3];
				//grad[debug_index++] = dJ[1][4];
				//grad[debug_index++] = dJ[1][5];
				//grad[debug_index++] = dJ[1][6];
				//grad[debug_index++] = dJ[1][7];
				//grad[debug_index++] = dJ[1][8];
				//grad[debug_index++] = dJ[2][0];
				//grad[debug_index++] = dJ[2][1];
				//grad[debug_index++] = dJ[2][2];
				//grad[debug_index++] = dJ[2][3];
				//grad[debug_index++] = dJ[2][4];
				//grad[debug_index++] = dJ[2][5];
				//grad[debug_index++] = dJ[2][6];
				//grad[debug_index++] = dJ[2][7];
				//grad[debug_index++] = dJ[2][8];
				//grad[debug_index++] = dJ[3][0];
				//grad[debug_index++] = dJ[3][1];
				//grad[debug_index++] = dJ[3][2];
				//grad[debug_index++] = dJ[3][3];
				//grad[debug_index++] = dJ[3][4];
				//grad[debug_index++] = dJ[3][5];
				//grad[debug_index++] = dJ[3][6];
				//grad[debug_index++] = dJ[3][7];
				//grad[debug_index++] = dJ[3][8];
				////////////////////////////////////////////////////////////////

				/*atomicAdd(&(grad[x0]), dE_dX[0], 0);
				atomicAdd(&(grad[x0 + num_vertices]), dE_dX[3], 0);
				atomicAdd(&(grad[x0 + (2 * num_vertices)]), dE_dX[6], 0);
				
				atomicAdd(&(grad[x1]), dE_dX[1], 0);
				atomicAdd(&(grad[x1 + num_vertices]), dE_dX[4], 0);
				atomicAdd(&(grad[x1 + (2 * num_vertices)]), dE_dX[7], 0);
				
				atomicAdd(&(grad[x2]), dE_dX[2], 0);
				atomicAdd(&(grad[x2 + num_vertices]), dE_dX[5], 0);
				atomicAdd(&(grad[x2 + (2 * num_vertices)]), dE_dX[8], 0);*/
			}
		}


		template<typename T>
		__global__ void setZeroKernel(T* vec)
		{
			vec[blockIdx.x] = 0;
		}

		void gradient() {
			setZeroKernel << <grad.size, 1 >> > (grad.cuda_arr);
			CheckErr(cudaDeviceSynchronize());
			GradientKernel<1024> << <ceil(num_faces / (double)1024), 1024 >> > (
				grad.cuda_arr,
				Cuda::Minimizer::X.cuda_arr,
				restShapeF.cuda_arr,
				restShapeArea.cuda_arr,
				D1d.cuda_arr,
				D2d.cuda_arr,
				num_faces,
				num_vertices);

			CheckErr(cudaDeviceSynchronize());
		}
		
		void FreeAllVariables() {
			cudaGetErrorString(cudaGetLastError());
			FreeMemory(restShapeF);
			FreeMemory(D1d);
			FreeMemory(D2d);
			FreeMemory(EnergyVec);
			FreeMemory(restShapeArea);
			FreeMemory(grad);
			FreeMemory(EnergyAtomic);
		}
	}
}
