#include "FixAllVertices.h"

FixAllVertices::FixAllVertices(
	const Eigen::MatrixXd& V,
	const Eigen::MatrixX3i& F)
{
	init_mesh(V, F);
	cuda_FixAllV = std::make_shared<Cuda_FixAllVertices>();
    name = "Fix All Vertices";
	w = 0.3;
	internalInitCuda();
	std::cout << "\t" << name << " constructor" << std::endl;
}

FixAllVertices::~FixAllVertices()
{
	std::cout << "\t" << name << " destructor" << std::endl;
}

void FixAllVertices::internalInitCuda() {
	unsigned int numF = restShapeF.rows();
	unsigned int numV = restShapeV.rows();
	cuda_FixAllV->num_faces = numF;
	cuda_FixAllV->num_vertices = numV;
	
	//alocate memory on host & device
	Cuda::AllocateMemory(cuda_FixAllV->restShapeV, numV);
	Cuda::AllocateMemory(cuda_FixAllV->grad, (3 * numV) + (10 * numF));
	Cuda::AllocateMemory(cuda_FixAllV->EnergyAtomic, 1);
	//init host buffers...
	for (int i = 0; i < cuda_FixAllV->grad.size; i++) {
		cuda_FixAllV->grad.host_arr[i] = 0;
	}
	for (int v = 0; v < numV; v++) {
		cuda_FixAllV->restShapeV.host_arr[v] = make_double3(restShapeV(v, 0), restShapeV(v, 1), restShapeV(v, 2));
	}
	// Copy input vectors from host memory to GPU buffers.
	Cuda::MemCpyHostToDevice(cuda_FixAllV->grad);
	Cuda::MemCpyHostToDevice(cuda_FixAllV->restShapeV);
}

void FixAllVertices::value(Cuda::Array<double>& curr_x)
{
	cuda_FixAllV->value(curr_x);
}

void FixAllVertices::gradient(Cuda::Array<double>& X)
{
	cuda_FixAllV->gradient(X);
}
