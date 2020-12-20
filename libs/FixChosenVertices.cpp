#include "FixChosenVertices.h"

FixChosenVertices::FixChosenVertices(
	const unsigned int numF,
	const unsigned int numV)
{
	this->numF = numF;
	this->numV = numV;
	Cuda_FixChosConst = std::make_shared<Cuda_FixChosenConstraints>(
		numF, numV, ConstraintsType::VERTICES);
    name = "Fix Chosen Vertices";
	w = 100000;
	std::cout << "\t" << name << " constructor" << std::endl;
}

FixChosenVertices::~FixChosenVertices() 
{
	std::cout << "\t" << name << " destructor" << std::endl;
}

void FixChosenVertices::init()
{
	std::cout << "\t" << name << " initialization" << std::endl;
	if(numV==0 || numF == 0)
		throw name + " must define members numV & numF before init()!";
	startV_x = (0 * numV);
	startV_y = (1 * numV);
	startV_z = (2 * numV);
}

void FixChosenVertices::updateExtConstraints(
	std::vector<int>& CVInd,
	Eigen::MatrixX3d& CVPos)
{
	m_value.lock();
	m_gradient.lock();

	Cuda::FreeMemory(Cuda_FixChosConst->Const_Ind);
	Cuda::FreeMemory(Cuda_FixChosConst->Const_Pos);
	Cuda::AllocateMemory(Cuda_FixChosConst->Const_Ind, CVInd.size());
	Cuda::AllocateMemory(Cuda_FixChosConst->Const_Pos, CVInd.size());
	//init host buffers...
	for (int i = 0; i < CVInd.size(); i++) {
		Cuda_FixChosConst->Const_Ind.host_arr[i] = CVInd[i];
		Cuda_FixChosConst->Const_Pos.host_arr[i] = make_double3(
			CVPos(i, 0),
			CVPos(i, 1),
			CVPos(i, 2)
		);
	}
	// Copy input vectors from host memory to GPU buffers.
	Cuda::MemCpyHostToDevice(Cuda_FixChosConst->Const_Ind);
	Cuda::MemCpyHostToDevice(Cuda_FixChosConst->Const_Pos);
	
	m_gradient.unlock();
	m_value.unlock();
}

void FixChosenVertices::updateX(Cuda::Array<double>& curr_x)
{
	//m.lock();
	//Eigen::MatrixX3d CurrConstrainedVerticesPos;
	//CurrConstrainedVerticesPos.resizeLike(ConstrainedVerticesPos);
	//for (int i = 0; i < ConstrainedVerticesInd.size(); i++)
	//{
	//	CurrConstrainedVerticesPos.row(i) <<
	//		X(ConstrainedVerticesInd[i] + startV_x),	//X-coordinate
	//		X(ConstrainedVerticesInd[i] + startV_y),	//Y-coordinate
	//		X(ConstrainedVerticesInd[i] + startV_z);	//Z-coordinate
	//}
	//diff = (CurrConstrainedVerticesPos - ConstrainedVerticesPos);
	//currConstrainedVerticesInd = ConstrainedVerticesInd;
	//m.unlock();
}

double FixChosenVertices::value(Cuda::Array<double>& curr_x, const bool update)
{
	m_value.lock();
	double value = Cuda_FixChosConst->value(curr_x);
	m_value.unlock();
	
	//double E = diff.squaredNorm();
	if (update) {
		energy_value = value;
	}
	return value;
}

void FixChosenVertices::gradient(Cuda::Array<double>& X, Eigen::VectorXd& g, const bool update)
{
	m_gradient.lock();
	Cuda_FixChosConst->gradient(X);
	m_gradient.unlock();
	//g.conservativeResize(numV * 3 + numF * 7);
	//g.setZero();
	//for (int i = 0; i < currConstrainedVerticesInd.size(); i++)
	//{
	//	g(currConstrainedVerticesInd[i] + startV_x) = 2 * diff(i, 0); //X-coordinate
	//	g(currConstrainedVerticesInd[i] + startV_y) = 2 * diff(i, 1); //Y-coordinate
	//	g(currConstrainedVerticesInd[i] + startV_z) = 2 * diff(i, 2); //Z-coordinate
	//}
	//if(update)
	//	gradient_norm = g.norm();
}