//#include "Cuda_FixChosenConstraints.cuh"
//#include "Cuda_Minimizer.cuh"
//
//Cuda_FixChosenConstraints::Cuda_FixChosenConstraints(const unsigned int numF, const unsigned int numV)
//{
//	cudaStreamCreate(&stream_value);
//	cudaStreamCreate(&stream_gradient);
//	Cuda::initIndices(mesh_indices, numF, numV, 0);
//	Cuda::AllocateMemory(grad, (3 * numV) + (7 * numF));
//	Cuda::AllocateMemory(EnergyAtomic, 1);
//	Cuda::AllocateMemory(Const_Ind, 0);
//	Cuda::AllocateMemory(Const_Pos, 0);
//	//Choose the kind of constraints
//	startX = mesh_indices.startVx;
//	startY = mesh_indices.startVy;
//	startZ = mesh_indices.startVz;
//	
//	//init host buffers...
//	for (int i = 0; i < grad.size; i++) {
//		grad.host_arr[i] = 0;
//	}
//	// Copy input vectors from host memory to GPU buffers.
//	Cuda::MemCpyHostToDevice(grad);
//}
//
//Cuda_FixChosenConstraints::~Cuda_FixChosenConstraints() {
//	cudaStreamDestroy(stream_value);
//	cudaStreamDestroy(stream_gradient);
//	cudaGetErrorString(cudaGetLastError());
//	FreeMemory(grad);
//	FreeMemory(EnergyAtomic);
//	FreeMemory(Const_Ind);
//	FreeMemory(Const_Pos);
//}
