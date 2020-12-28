#include "FixChosenConstraints.h"

FixChosenConstraints::FixChosenConstraints(
	const unsigned int numF,
	const unsigned int numV,
	const ConstraintsType type)
{
	Cuda_FixChosConst = std::make_shared<Cuda_FixChosenConstraints>(numF, numV, type);
    if(type == ConstraintsType::VERTICES)
		name = "Fix Chosen Vertices";
	if(type == ConstraintsType::NORMALS)
		name = "Fix Chosen Normals";
	if(type == ConstraintsType::SPHERES)
		name = "Fix Chosen Centers";
	w = 100000;
	std::cout << "\t" << name << " constructor" << std::endl;
}

FixChosenConstraints::~FixChosenConstraints() 
{
	std::cout << "\t" << name << " destructor" << std::endl;
}

void FixChosenConstraints::updateExtConstraints(
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

void FixChosenConstraints::value(Cuda::Array<double>& curr_x)
{
	m_value.lock();
	Cuda_FixChosConst->value(curr_x);
	m_value.unlock();
}

void FixChosenConstraints::gradient(Cuda::Array<double>& X)
{
	m_gradient.lock();
	Cuda_FixChosConst->gradient(X);
	m_gradient.unlock();
}