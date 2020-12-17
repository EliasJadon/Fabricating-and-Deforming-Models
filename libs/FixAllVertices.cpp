#include "FixAllVertices.h"

FixAllVertices::FixAllVertices()
{
    name = "Fix All Vertices";
	w = 0;
	std::cout << "\t" << name << " constructor" << std::endl;
}

FixAllVertices::~FixAllVertices()
{
	std::cout << "\t" << name << " destructor" << std::endl;
}

void FixAllVertices::init()
{
	std::cout << "\t" << name << " initialization" << std::endl;
	if (restShapeV.size() == 0 || restShapeF.size() == 0)
		throw name + " must define members V,F before init()!";
	internalInitCuda();
}

void FixAllVertices::internalInitCuda() {
	unsigned int numF = restShapeF.rows();
	unsigned int numV = restShapeV.rows();
	Cuda::FixAllVertices::num_faces = numF;
	Cuda::FixAllVertices::num_vertices = numV;
	
	//alocate memory on host & device
	Cuda::AllocateMemory(Cuda::FixAllVertices::restShapeV, numV);
	Cuda::AllocateMemory(Cuda::FixAllVertices::grad, (3 * numV) + (7 * numF));
	Cuda::AllocateMemory(Cuda::FixAllVertices::EnergyAtomic, 1);
	//init host buffers...
	for (int i = 0; i < Cuda::FixAllVertices::grad.size; i++) {
		Cuda::FixAllVertices::grad.host_arr[i] = 0;
	}
	for (int v = 0; v < numV; v++) {
		Cuda::FixAllVertices::restShapeV.host_arr[v] = make_double3(restShapeV(v, 0), restShapeV(v, 1), restShapeV(v, 2));
	}
	// Copy input vectors from host memory to GPU buffers.
	Cuda::MemCpyHostToDevice(Cuda::FixAllVertices::grad);
	Cuda::MemCpyHostToDevice(Cuda::FixAllVertices::restShapeV);
}

void FixAllVertices::updateX(Cuda::Array<double>& curr_x)
{
	Cuda::MemCpyDeviceToHost(curr_x);
	assert(curr_x.size == (restShapeV.size() + 7*restShapeF.rows()));
	CurrV.resize(restShapeV.rows(), 3);
	for (int v = 0; v < restShapeV.rows(); v++) {
		CurrV(v, 0) = curr_x.host_arr[v];
		CurrV(v, 1) = curr_x.host_arr[v + restShapeV.rows()];
		CurrV(v, 2) = curr_x.host_arr[v + 2 * restShapeV.rows()];
	}
	//CurrV = Eigen::Map<const Eigen::MatrixX3d>(Cuda::Minimizer::curr_x.host_arr.middleRows(0,restShapeV.size()).data(), restShapeV.rows(), 3);
}

double FixAllVertices::value(Cuda::Array<double>& curr_x, const bool update)
{
	double E = Cuda::FixAllVertices::value(curr_x);
	///////////////////////////////////////////////////
	// for debugging...
	/*updateX(Eigen::VectorXd::Zero(1));
	double oldE = (CurrV - restShapeV).squaredNorm();
	std::cout << "oldE = \t" << oldE << std::endl;
	std::cout << "E = \t" << E << std::endl;*/
	///////////////////////////////////////////////////

	if (update)
		energy_value = E;
	return E;
}

void FixAllVertices::gradient(Cuda::Array<double>& X, Eigen::VectorXd& g, const bool update)
{
	///////////////////////////////////////////////////
	// for debugging...
	//updateX(Eigen::VectorXd::Zero(1));

	int n = restShapeV.rows();
	g.conservativeResize(restShapeV.size()+ 7*restShapeF.rows());
	g.setZero();

	Eigen::MatrixX3d diff = CurrV - restShapeV;
	for (int i = 0; i < n; i++) {
		g(i + (0 * n)) = 2 * diff(i, 0); //X-coordinate
		g(i + (1 * n)) = 2 * diff(i, 1); //Y-coordinate
		g(i + (2 * n)) = 2 * diff(i, 2); //Z-coordinate
	}
	///////////////////////////////////////////////////
	if(update)
		gradient_norm = g.norm();
}
