#include "Cuda_AuxBendingNormal.cuh"
#include "Cuda_Minimizer.cuh"

namespace Cuda {
	namespace AuxBendingNormal {
		//dynamic variables
		double w1 = 1, w2 = 100, w3 = 100;
		FunctionType functionType;
		double planarParameter;
		Array<rowVector<double>> CurrV, CurrN; //Eigen::MatrixX3d
		Array<double> d_normals;
		Array<double> grad;
		//help variables - dynamic
		Array<double> EnergyAtomic;
		
		//Static variables
		Array<rowVector<int>> restShapeF;
		Array<double> restAreaPerFace, restAreaPerHinge; //Eigen::VectorXd
		int num_hinges, num_faces, num_vertices;
		Array<hinge> hinges_faceIndex; //std::vector<Eigen::Vector2d> //num_hinges*2
		Array<int> x0_GlobInd, x1_GlobInd, x2_GlobInd, x3_GlobInd; //Eigen::VectorXi //num_hinges
		Array<hinge> x0_LocInd, x1_LocInd, x2_LocInd, x3_LocInd; //Eigen::MatrixXi //num_hinges*2
		
		template<typename T> __device__ rowVector<T> addVectors(
			rowVector<T> a,
			rowVector<T> b)
		{
			rowVector<T> result;
			result.x = a.x + b.x;
			result.y = a.y + b.y;
			result.z = a.z + b.z;
			return result;
		}
		template<typename T> __device__ rowVector<T> subVectors(
			const rowVector<T> a,
			const rowVector<T> b)
		{
			rowVector<T> result;
			result.x = a.x - b.x;
			result.y = a.y - b.y;
			result.z = a.z - b.z;
			return result;
		}
		template<typename T> __device__ T mulVectors(rowVector<T> a, rowVector<T> b)
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

		__global__ void updateXKernel(
			double* d_normals, 
			const rowVector<double>* Normals, 
			const hinge* Hinges_Findex, 
			const int size)
		{
			int hi = blockIdx.x;
			if (hi < size)
			{
				int f0 = Hinges_Findex[hi].f0;
				int f1 = Hinges_Findex[hi].f1;
				rowVector<double> diff = subVectors(Normals[f1], Normals[f0]);
				d_normals[hi] = mulVectors(diff, diff);
			}
		}

		

		__device__ void Energy1Kernel(
			const double w1,
			double* resAtomic,
			const double* x,
			const double* area,
			const double planarParameter,
			const FunctionType functionType,
			const int hi,
			const int size)
		{
			if (hi < size)
			{
				double res;
				if (functionType == FunctionType::SIGMOID) {
					double x2 = x[hi] * x[hi];
					res = x2 / (x2 + planarParameter);
				}
				else if (functionType == FunctionType::QUADRATIC)
					res = x[hi] * x[hi];
				else if (functionType == FunctionType::EXPONENTIAL)
					res = 0;

				res *= area[hi];
				atomicAdd(resAtomic, res, 0);
			}
		}

		__device__ void Energy2Kernel(
			const double w2,
			double* resAtomic,
			const rowVector<double>* Normals,
			const int fi,
			const int size)
		{
			if (fi < size)
			{
				double res;
				double x2 = Normals[fi].x * Normals[fi].x;
				double y2 = Normals[fi].y * Normals[fi].y;
				double z2 = Normals[fi].z * Normals[fi].z;
				double sqrN = x2 + y2 + z2 - 1;
				res = sqrN * sqrN;
				res *= w2;
				atomicAdd(resAtomic, res, 0);
			}
		}
		__device__ void Energy3Kernel(
			const double w3,
			double* resAtomic, 
			const rowVector<int>* restShapeF,
			const rowVector<double>* Vertices,
			const rowVector<double>* Normals,
			const int fi,
			const int size)
		{
			if (fi < size)
			{
				double res;
				// (N^T*(x1-x0))^2 + (N^T*(x2-x1))^2 + (N^T*(x0-x2))^2
				int x0 = restShapeF[fi].x;
				int x1 = restShapeF[fi].y;
				int x2 = restShapeF[fi].z;

				rowVector<double> e21 = subVectors(Vertices[x2], Vertices[x1]);
				rowVector<double> e10 = subVectors(Vertices[x1], Vertices[x0]);
				rowVector<double> e02 = subVectors(Vertices[x0], Vertices[x2]);
				double d1 = mulVectors(Normals[fi], e21);
				double d2 = mulVectors(Normals[fi], e10);
				double d3 = mulVectors(Normals[fi], e02);
				res = d1 * d1 + d2 * d2 + d3 * d3;
				res *= w3;
				atomicAdd(resAtomic, res, 0);
			}
		}
		__global__ void EnergyKernel(
			double* resAtomic,
			const double w1,
			const double w2,
			const double w3,
			const double * d_normals,
			const rowVector<double> * CurrV,
			const rowVector<double> * CurrN,
			const rowVector<int> * restShapeF,
			const double * restAreaPerHinge,
			const double planarParameter,
			const FunctionType functionType,
			const int num_hinges,
			const int num_faces) 
		{
			int index = blockIdx.x;
			//0	,..., F-1,		==> Call Energy(3)
			//F	,..., 2F-1,		==> Call Energy(2)
			//2F,..., 2F+h-1	==> Call Energy(1)
			if (index < num_faces) {
				Energy3Kernel(
					w3,
					resAtomic,
					restShapeF,
					CurrV,
					CurrN,
					index,
					num_faces);
			}
			else if (index < (2*num_faces)) {
				Energy2Kernel(
					w2,
					resAtomic,
					CurrN,
					index - num_faces,
					num_faces);
			}
			else {
				Energy1Kernel(
					w1,
					resAtomic,
					d_normals,
					restAreaPerHinge,
					planarParameter,
					functionType,
					index - (2 * num_faces),
					num_hinges);
			}			
		}
		
		double value() {
			EnergyAtomic.host_arr[0] = 0;
			MemCpyHostToDevice(EnergyAtomic);
			EnergyKernel <<<num_hinges + num_faces + num_faces, 1 >> > (
					EnergyAtomic.cuda_arr,
					w1,w2,w3,
					d_normals.cuda_arr,
					CurrV.cuda_arr,
					CurrN.cuda_arr,
					restShapeF.cuda_arr,
					restAreaPerHinge.cuda_arr,
					planarParameter,
					functionType,
					num_hinges,
					num_faces);
			CheckErr(cudaDeviceSynchronize());
			MemCpyDeviceToHost(EnergyAtomic);
			return EnergyAtomic.host_arr[0];
		}



		__global__ void updateVN(
			rowVector<double>* CurrN,
			rowVector<double>* CurrV,
			double* curr_x,
			const int num_vertices, 
			const int num_faces) 
		{
			int index = blockIdx.x;
			int thread = threadIdx.x;
			if (index < num_faces) {
				int f = index;
				if (thread == 0)
					CurrN[f].x = curr_x[f + (3 * num_vertices)];
				else if (thread == 1)
					CurrN[f].y = curr_x[f + (3 * num_vertices + num_faces)];
				else if (thread == 2)
					CurrN[f].z = curr_x[f + (3 * num_vertices + 2 * num_faces)];
			}
			else {
				int v = index - num_faces;
				if (thread == 0)
					CurrV[v].x = curr_x[v];
				if (thread == 1)
					CurrV[v].y = curr_x[v + num_vertices];
				if (thread == 2)
					CurrV[v].z = curr_x[v + (2 * num_vertices)];
			}
		}

		void updateX() {
			updateVN<<<num_vertices+ num_faces,3>>>(
				CurrN.cuda_arr,
				CurrV.cuda_arr,
				Cuda::Minimizer::curr_x.cuda_arr,
				num_vertices,
				num_faces);

			CheckErr(cudaDeviceSynchronize());

			updateXKernel <<<num_hinges, 1>>> (
				d_normals.cuda_arr, 
				CurrN.cuda_arr, 
				hinges_faceIndex.cuda_arr, 
				num_hinges);
			CheckErr(cudaDeviceSynchronize());
			
			////For Debugging...
			//MemCpyDeviceToHost(d_normals);
			//for (int hi = 0; hi < num_hinges; hi++) {
			//	int f0 = hinges_faceIndex.host_arr[hi].f0;
			//	int f1 = hinges_faceIndex.host_arr[hi].f1;
			//	double diffX = CurrN.host_arr[f1].x - CurrN.host_arr[f0].x;
			//	double diffY = CurrN.host_arr[f1].y - CurrN.host_arr[f0].y;
			//	double diffZ = CurrN.host_arr[f1].z - CurrN.host_arr[f0].z;
			//	double expected = diffX * diffX + diffY * diffY + diffZ * diffZ;
			//	double epsilon = 1e-3;
			//	double diff = d_normals.host_arr[hi] - expected;
			//	if (diff > epsilon || diff < -epsilon) {
			//		std::cout << "Error at index" << hi << std::endl;
			//		std::cout << "Expected = " << expected << std::endl;
			//		std::cout << "d_normals.host_arr[hi] = " << d_normals.host_arr[hi] << std::endl;
			//		std::cout << "diff = " << diff << std::endl;
			//		exit(1);
			//	}
			//	else {
			//		std::cout << "okay!\n";
			//	}
			//}
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
				return 0;
		}

		__device__ void gradient1Kernel(
			double* grad,
			const hinge* hinges_faceIndex,
			const rowVector<double>* CurrN,
			const double* d_normals,
			const double* restAreaPerHinge,
			const double planarParameter,
			const FunctionType functionType,
			const double w1,
			const int hi,
			const int thread,
			const int num_faces,
			const int num_vertices)
		{
			int f0 = hinges_faceIndex[hi].f0;
			int f1 = hinges_faceIndex[hi].f1;
			double coeff = w1 * restAreaPerHinge[hi] * dPhi_dm(d_normals[hi], planarParameter, functionType);

			if (thread == 0) { //n0.x;
				atomicAdd(
					&grad[f0 + (3 * num_vertices)], 
					coeff * 2 * (CurrN[f0].x - CurrN[f1].x),
					0);
			}
			else if (thread == 1) { //n1.x
				atomicAdd(
					&grad[f1 + (3 * num_vertices)],
					coeff * 2 * (CurrN[f1].x - CurrN[f0].x),
					0);	
			}
			else if (thread == 2) {//n0.y
				atomicAdd(
					&grad[f0 + (3 * num_vertices) + num_faces],
					coeff * 2 * (CurrN[f0].y - CurrN[f1].y),
					0);	
			}
			else if (thread == 3) { //n1.y
				atomicAdd(
					&grad[f1 + (3 * num_vertices) + num_faces],
					coeff * 2 * (CurrN[f1].y - CurrN[f0].y),
					0);	
			}
			else if (thread == 4) { //n0.z
				atomicAdd(
					&grad[f0 + (3 * num_vertices) + (2 * num_faces)],
					coeff * 2 * (CurrN[f0].z - CurrN[f1].z),
					0);	
			}
			else if (thread == 5) { //n1.z
				atomicAdd(
					&grad[f1 + (3 * num_vertices) + (2 * num_faces)],
					coeff * 2 * (CurrN[f1].z - CurrN[f0].z),
					0);	
			}
		}
		__device__ void gradient2Kernel(
			double* grad,
			const rowVector<double>* CurrN,
			const int fi,
			const int thread,
			const double w2,
			const int num_vertices,
			const int num_faces)
		{
			double coeff = w2 * 4 * (mulVectors(CurrN[fi], CurrN[fi]) - 1);
			if (thread == 0) { //N.x
				atomicAdd(
					&grad[fi + (3 * num_vertices)],
					coeff * CurrN[fi].x,
					0);
			}
			else if (thread == 1) { //N.y
				atomicAdd(
					&grad[fi + (3 * num_vertices) + num_faces],
					coeff * CurrN[fi].y,
					0);
			}
			else if (thread == 2) { //N.z
				atomicAdd(
					&grad[fi + (3 * num_vertices) + (2 * num_faces)],
					coeff * CurrN[fi].z,
					0);
			}
		}
		__device__ void gradient3Kernel(
			double* grad,
			const rowVector<int>* restShapeF,
			const rowVector<double>* CurrV,
			const rowVector<double>* CurrN,
			const int fi,
			const int thread,
			const double w3,
			const int num_vertices,
			const int num_faces)
		{
			int x0 = restShapeF[fi].x;
			int x1 = restShapeF[fi].y;
			int x2 = restShapeF[fi].z;
			rowVector<double> e21 = subVectors(CurrV[x2], CurrV[x1]);
			rowVector<double> e10 = subVectors(CurrV[x1], CurrV[x0]);
			rowVector<double> e02 = subVectors(CurrV[x0], CurrV[x2]);
			double N02 = mulVectors(CurrN[fi], e02);
			double N10 = mulVectors(CurrN[fi], e10);
			double N21 = mulVectors(CurrN[fi], e21);
			double coeff = 2 * w3;
			int num_2verices = 2 * num_vertices;
			int num_3verices_fi = fi + num_2verices + num_vertices;

			switch (thread) {
			case 0: //x0
				atomicAdd(
					&grad[x0],
					coeff * CurrN[fi].x * (N02 - N10),
					0);
				break;
			case 1: //y0
				atomicAdd(
					&grad[x0 + num_vertices],
					coeff * CurrN[fi].y * (N02 - N10),
					0);
				break;
			case 2: //z0
				atomicAdd(
					&grad[x0 + num_2verices],
					coeff * CurrN[fi].z * (N02 - N10),
					0);
				break;
			case 3: //x1
				atomicAdd(
					&grad[x1],
					coeff * CurrN[fi].x * (N10 - N21),
					0);
				break;
			case 4: //y1
				atomicAdd(
					&grad[x1 + num_vertices],
					coeff * CurrN[fi].y * (N10 - N21),
					0);
				break;
			case 5: //z1
				atomicAdd(
					&grad[x1 + num_2verices],
					coeff * CurrN[fi].z * (N10 - N21),
					0);
				break;
			case 6: //x2
				atomicAdd(
					&grad[x2],
					coeff * CurrN[fi].x * (N21 - N02),
					0);
				break;
			case 7: //y2
				atomicAdd(
					&grad[x2 + num_vertices],
					coeff * CurrN[fi].y * (N21 - N02),
					0);
				break;
			case 8: //z2
				atomicAdd(
					&grad[x2 + num_2verices],
					coeff * CurrN[fi].z * (N21 - N02),
					0);
				break;
			case 9: //Nx
				atomicAdd(
					&grad[num_3verices_fi],
					coeff * (N10 * e10.x + N21 * e21.x + N02 * e02.x),
					0);
				break;
			case 10: //Ny
				atomicAdd(
					&grad[num_3verices_fi + num_faces],
					coeff * (N10 * e10.y + N21 * e21.y + N02 * e02.y),
					0);
				break;
			case 11: //Nz
				atomicAdd(
					&grad[num_3verices_fi + (2 * num_faces)],
					coeff * (N10 * e10.z + N21 * e21.z + N02 * e02.z),
					0);
				break;
			}
		}

		__global__ void gradientKernel(
			double* grad,
			const hinge* hinges_faceIndex,
			const double* d_normals,
			const rowVector<double>* CurrV,
			const rowVector<double>* CurrN,
			const rowVector<int>* restShapeF,
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
					CurrV,
					CurrN,
					Bl_index,
					Th_Index,
					w3,
					num_vertices,
					num_faces);
			}
			else if (Bl_index < (2 * num_faces)) {
				gradient2Kernel(
					grad, 
					CurrN, 
					Bl_index - num_faces, 
					Th_Index,
					w2,
					num_vertices,
					num_faces);
			}
			else {
				gradient1Kernel(
					grad,
					hinges_faceIndex,
					CurrN,
					d_normals,
					restAreaPerHinge,
					planarParameter,
					functionType,
					w1,
					Bl_index - (2 * num_faces),
					Th_Index,
					num_faces,
					num_vertices);
			}
		}

		template<typename T>
		__global__ void setZeroKernel(T* vec)
		{
			int index = blockIdx.x;
			vec[index] = 0;
		}

		void gradient()
		{
			setZeroKernel << <grad.size, 1 >> > (grad.cuda_arr);
			CheckErr(cudaDeviceSynchronize());

			gradientKernel << <num_hinges + num_faces + num_faces, 12 >> > (
				grad.cuda_arr,
				hinges_faceIndex.cuda_arr,
				d_normals.cuda_arr,
				CurrV.cuda_arr,
				CurrN.cuda_arr,
				restShapeF.cuda_arr,
				restAreaPerHinge.cuda_arr,
				planarParameter,
				functionType,
				num_hinges,
				num_faces,
				num_vertices,
				w1,
				w2,
				w3);
			CheckErr(cudaDeviceSynchronize());
			//MemCpyDeviceToHost(grad);


			////Energy 1: per hinge
			//for (int hi = 0; hi < num_hinges; hi++) {
			//	int f0 = hinges_faceIndex.cuda_arr[hi].f0;
			//	int f1 = hinges_faceIndex.cuda_arr[hi].f1;

			//	////////////////////////////////dm_dN
			//	double dm_dN[6];
			//	dm_dN[0] = 2 * (CurrN.cuda_arr[f0].x - CurrN.cuda_arr[f1].x);	//n0.x
			//	dm_dN[1] = 2 * (CurrN.cuda_arr[f0].y - CurrN.cuda_arr[f1].y); // n0.y
			//	dm_dN[2] = 2 * (CurrN.cuda_arr[f0].z - CurrN.cuda_arr[f1].z); //n0.z
			//	dm_dN[3] = -dm_dN[0];	//n1.x
			//	dm_dN[4] = -dm_dN[1];	//n1.y
			//	dm_dN[5] = -dm_dN[2];	//n1.z
			//	////////////////////////////////

			//	double coeff = w1 *	restAreaPerHinge.cuda_arr[hi] *	dphi_dm(hi);
			//	//N0.x , N1.x		
			//	grad.cuda_arr[f0 + (3 * num_vertices)] += coeff * dm_dN[0];
			//	grad.cuda_arr[f1 + (3 * num_vertices)] += coeff * dm_dN[3];
			//	//N0.y , N1.y	
			//	grad.cuda_arr[f0 + (3 * num_vertices) + num_faces] += coeff * dm_dN[1];
			//	grad.cuda_arr[f1 + (3 * num_vertices) + num_faces] += coeff * dm_dN[4];
			//	//N0.z , N1.z	
			//	grad.cuda_arr[f0 + (3 * num_vertices) + (2 * num_faces)] += coeff * dm_dN[2];
			//	grad.cuda_arr[f1 + (3 * num_vertices) + (2 * num_faces)] += coeff * dm_dN[5];
			//	
			//}

			////Energy 2: per face
			//for (int fi = 0; fi < num_faces; fi++) {
			//	double coeff = w2 * 4 * (CurrN.cuda_arr[fi].squaredNorm() - 1);
			//	//N.x
			//	grad.cuda_arr[fi + (3 * num_vertices)] +=
			//		coeff * CurrN.cuda_arr[fi].x;
			//	//N.y
			//	grad.cuda_arr[fi + (3 * num_vertices) + num_faces] +=
			//		coeff * CurrN.cuda_arr[fi].y;
			//	//N.z
			//	grad.cuda_arr[fi + (3 * num_vertices) + (2 * num_faces)] +=
			//		coeff * CurrN.cuda_arr[fi].z;

			//}
			//	
			//		
			////Energy 3: per face
			//for (int fi = 0; fi < num_faces; fi++) {
			//	int x0 = restShapeF.cuda_arr[fi].x;
			//	int x1 = restShapeF.cuda_arr[fi].y;
			//	int x2 = restShapeF.cuda_arr[fi].z;
			//	rowVector<double> e21 = CurrV.row(x2) - CurrV.row(x1);
			//	rowVector<double> e10 = CurrV.row(x1) - CurrV.row(x0);
			//	rowVector<double> e02 = CurrV.row(x0) - CurrV.row(x2);
			//	double N02 = CurrN.row(fi) * e02;
			//	double N10 = CurrN.row(fi) * e10;
			//	double N21 = CurrN.row(fi) * e21;
			//	double coeff = 2 * w3;
			//	int num_2verices = 2 * num_vertices;
			//	int num_3verices_fi = fi + num_2verices + num_vertices;

			//	grad.cuda_arr[x0]				+= coeff * CurrN.cuda_arr[fi].x * (N02 - N10);//x0
			//	grad.cuda_arr[x0 + num_vertices]+= coeff * CurrN.cuda_arr[fi].y * (N02 - N10);//y0
			//	grad.cuda_arr[x0 + num_2verices]+= coeff * CurrN.cuda_arr[fi].z * (N02 - N10);//z0

			//	grad.cuda_arr[x1]				+= coeff * CurrN.cuda_arr[fi].x * (N10 - N21);//x1
			//	grad.cuda_arr[x1 + num_vertices]+= coeff * CurrN.cuda_arr[fi].y * (N10 - N21);//y1
			//	grad.cuda_arr[x1 + num_2verices]+= coeff * CurrN.cuda_arr[fi].z * (N10 - N21);//z1

			//	grad.cuda_arr[x2]				+= coeff * CurrN.cuda_arr[fi].x * (N21 - N02);//x2
			//	grad.cuda_arr[x2 + num_vertices]+= coeff * CurrN.cuda_arr[fi].y * (N21 - N02);//y2
			//	grad.cuda_arr[x2 + num_2verices]+= coeff * CurrN.cuda_arr[fi].z * (N21 - N02);//z2
			//	
			//	grad.cuda_arr[num_3verices_fi]					+= coeff * (N10 * e10.x + N21 * e21.x + N02 * e02.x);//Nx
			//	grad.cuda_arr[num_3verices_fi + num_faces]		+= coeff * (N10 * e10.y + N21 * e21.y + N02 * e02.y);//Ny
			//	grad.cuda_arr[num_3verices_fi + (2 * num_faces)]+= coeff * (N10 * e10.z + N21 * e21.z + N02 * e02.z);//Nz

			//	
			//}
		}
		
		void FreeAllVariables() {
			cudaGetErrorString(cudaGetLastError());
			FreeMemory(restShapeF);
			FreeMemory(grad);
			FreeMemory(CurrV);
			FreeMemory(CurrN);
			FreeMemory(restAreaPerFace);
			FreeMemory(restAreaPerHinge);
			FreeMemory(d_normals);
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
