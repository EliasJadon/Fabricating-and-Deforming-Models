#include "AuxBendingNormal.h"
#include <unsupported/Eigen/MatrixFunctions>
#include <igl/triangle_triangle_adjacency.h>

AuxBendingNormal::AuxBendingNormal(
	const Eigen::MatrixXd& V, 
	const Eigen::MatrixX3i& F,
	const Cuda::PenaltyFunction penaltyFunction)
{
	init_mesh(V, F);
	name = "Aux Bending Normal";
	w = 1;
	colorP = Eigen::Vector3f(51 / 255.0f, 1, 1);
	colorM = Eigen::Vector3f(1, 51 / 255.0f, 1);

	//Initialize rest variables (X0) 
	calculateHinges();
	restAreaPerHinge.resize(num_hinges);
	igl::doublearea(restShapeV, restShapeF, restAreaPerFace);
	restAreaPerFace /= 2;
	for (int hi = 0; hi < num_hinges; hi++) {
		int f0 = hinges_faceIndex[hi](0);
		int f1 = hinges_faceIndex[hi](1);
		restAreaPerHinge(hi) = (restAreaPerFace(f0) + restAreaPerFace(f1)) / 2;
	}
	Efi.setZero();

	//Init Cuda variables
	const unsigned int numF = restShapeF.rows();
	const unsigned int numV = restShapeV.rows();
	const unsigned int numH = num_hinges;
	cuda_ABN = std::make_shared<Cuda_AuxBendingNormal>(penaltyFunction, numF, numV, numH);
	internalInitCuda();


	std::cout << "\t" << name << " constructor" << std::endl;
}

AuxBendingNormal::~AuxBendingNormal() {
	std::cout << "\t" << name << " destructor" << std::endl;
}

void AuxBendingNormal::internalInitCuda() {
	//init host buffers...
	for (int i = 0; i < cuda_ABN->grad.size; i++) {
		cuda_ABN->grad.host_arr[i] = 0;
	}
	for (int f = 0; f < restShapeF.rows(); f++) {
		cuda_ABN->restShapeF.host_arr[f] = make_int3(restShapeF(f, 0), restShapeF(f, 1), restShapeF(f, 2));
		cuda_ABN->restAreaPerFace.host_arr[f] = restAreaPerFace[f];
	}
	for (int h = 0; h < num_hinges; h++) {
		cuda_ABN->restAreaPerHinge.host_arr[h] = restAreaPerHinge[h];
		cuda_ABN->weight_PerHinge.host_arr[h] = 1;
		cuda_ABN->Sigmoid_PerHinge.host_arr[h] = cuda_ABN->get_SigmoidParameter();
		cuda_ABN->hinges_faceIndex.host_arr[h] = Cuda::newHinge(hinges_faceIndex[h][0], hinges_faceIndex[h][1]);
		cuda_ABN->x0_GlobInd.host_arr[h] = x0_GlobInd[h];
		cuda_ABN->x1_GlobInd.host_arr[h] = x1_GlobInd[h];
		cuda_ABN->x2_GlobInd.host_arr[h] = x2_GlobInd[h];
		cuda_ABN->x3_GlobInd.host_arr[h] = x3_GlobInd[h];
		cuda_ABN->x0_LocInd.host_arr[h] = Cuda::newHinge(x0_LocInd(h, 0), x0_LocInd(h, 1));
		cuda_ABN->x1_LocInd.host_arr[h] = Cuda::newHinge(x1_LocInd(h, 0), x1_LocInd(h, 1));
		cuda_ABN->x2_LocInd.host_arr[h] = Cuda::newHinge(x2_LocInd(h, 0), x2_LocInd(h, 1));
		cuda_ABN->x3_LocInd.host_arr[h] = Cuda::newHinge(x3_LocInd(h, 0), x3_LocInd(h, 1));
	}

	// Copy input vectors from host memory to GPU buffers.
	Cuda::MemCpyHostToDevice(cuda_ABN->grad);
	Cuda::MemCpyHostToDevice(cuda_ABN->restShapeF);
	Cuda::MemCpyHostToDevice(cuda_ABN->restAreaPerFace);
	Cuda::MemCpyHostToDevice(cuda_ABN->restAreaPerHinge);
	Cuda::MemCpyHostToDevice(cuda_ABN->weight_PerHinge);
	Cuda::MemCpyHostToDevice(cuda_ABN->Sigmoid_PerHinge);
	Cuda::MemCpyHostToDevice(cuda_ABN->hinges_faceIndex);
	Cuda::MemCpyHostToDevice(cuda_ABN->x0_GlobInd);
	Cuda::MemCpyHostToDevice(cuda_ABN->x1_GlobInd);
	Cuda::MemCpyHostToDevice(cuda_ABN->x2_GlobInd);
	Cuda::MemCpyHostToDevice(cuda_ABN->x3_GlobInd);
	Cuda::MemCpyHostToDevice(cuda_ABN->x0_LocInd);
	Cuda::MemCpyHostToDevice(cuda_ABN->x1_LocInd);
	Cuda::MemCpyHostToDevice(cuda_ABN->x2_LocInd);
	Cuda::MemCpyHostToDevice(cuda_ABN->x3_LocInd);
}

void AuxBendingNormal::calculateHinges() {
	std::vector<std::vector<std::vector<int>>> TT;
	igl::triangle_triangle_adjacency(restShapeF, TT);
	assert(TT.size() == restShapeF.rows());
	hinges_faceIndex.clear();

	///////////////////////////////////////////////////////////
	//Part 1 - Find unique hinges
	for (int fi = 0; fi < TT.size(); fi++) {
		std::vector< std::vector<int>> CurrFace = TT[fi];
		assert(CurrFace.size() == 3 && "Each face should be a triangle (not square for example)!");
		for (std::vector<int> hinge : CurrFace) {
			if (hinge.size() == 1) {
				//add this "hinge"
				int FaceIndex1 = fi;
				int FaceIndex2 = hinge[0];

				if (FaceIndex2 < FaceIndex1) {
					//Skip
					//This hinge already exists!
					//Empty on purpose
				}
				else {
					hinges_faceIndex.push_back(Eigen::Vector2d(FaceIndex1, FaceIndex2));
				}
			}
			else if (hinge.size() == 0) {
				//Skip
				//This triangle has no another adjacent triangle on that edge
				//Empty on purpose
			}
			else {
				//We shouldn't get here!
				//The mesh is invalid
				assert("Each triangle should have only one adjacent triangle on each edge!");
			}

		}
	}
	num_hinges = hinges_faceIndex.size();

	///////////////////////////////////////////////////////////
	//Part 2 - Find x0,x1,x2,x3 indecis for each hinge
	x0_GlobInd.resize(num_hinges);
	x1_GlobInd.resize(num_hinges);
	x2_GlobInd.resize(num_hinges);
	x3_GlobInd.resize(num_hinges);
	x0_LocInd.resize(num_hinges, 2); x0_LocInd.setConstant(-1);
	x1_LocInd.resize(num_hinges, 2); x1_LocInd.setConstant(-1);
	x2_LocInd.resize(num_hinges, 2); x2_LocInd.setConstant(-1);
	x3_LocInd.resize(num_hinges, 2); x3_LocInd.setConstant(-1);

	for (int hi = 0; hi < num_hinges; hi++) {
		//first triangle vertices
		int v1 = restShapeF(hinges_faceIndex[hi](0), 0);
		int v2 = restShapeF(hinges_faceIndex[hi](0), 1);
		int v3 = restShapeF(hinges_faceIndex[hi](0), 2);
		//second triangle vertices
		int V1 = restShapeF(hinges_faceIndex[hi](1), 0);
		int V2 = restShapeF(hinges_faceIndex[hi](1), 1);
		int V3 = restShapeF(hinges_faceIndex[hi](1), 2);

		/*
		* Here we should find x0,x1,x2,x3
		* from the given two triangles (v1,v2,v3),(V1,V2,V3)
		*
		*	x0--x2
		*  / \ /
		* x3--x1
		*
		*/
		if (v1 != V1 && v1 != V2 && v1 != V3) {
			x2_GlobInd(hi) = v1; x2_LocInd(hi, 0) = 0;
			x0_GlobInd(hi) = v2; x0_LocInd(hi, 0) = 1;
			x1_GlobInd(hi) = v3; x1_LocInd(hi, 0) = 2;

			if (V1 != v1 && V1 != v2 && V1 != v3) {
				x3_GlobInd(hi) = V1; x3_LocInd(hi, 1) = 0;
			}
			else if (V2 != v1 && V2 != v2 && V2 != v3) {
				x3_GlobInd(hi) = V2; x3_LocInd(hi, 1) = 1;
			}
			else {
				x3_GlobInd(hi) = V3; x3_LocInd(hi, 1) = 2;
			}
		}
		else if (v2 != V1 && v2 != V2 && v2 != V3) {
			x2_GlobInd(hi) = v2; x2_LocInd(hi, 0) = 1;
			x0_GlobInd(hi) = v1; x0_LocInd(hi, 0) = 0;
			x1_GlobInd(hi) = v3; x1_LocInd(hi, 0) = 2;

			if (V1 != v1 && V1 != v2 && V1 != v3) {
				x3_GlobInd(hi) = V1; x3_LocInd(hi, 1) = 0;
			}	
			else if (V2 != v1 && V2 != v2 && V2 != v3) {
				x3_GlobInd(hi) = V2; x3_LocInd(hi, 1) = 1;
			}
			else {
				x3_GlobInd(hi) = V3; x3_LocInd(hi, 1) = 2;
			}
		}
		else {
			x2_GlobInd(hi) = v3; x2_LocInd(hi, 0) = 2;
			x0_GlobInd(hi) = v1; x0_LocInd(hi, 0) = 0;
			x1_GlobInd(hi) = v2; x1_LocInd(hi, 0) = 1;

			if (V1 != v1 && V1 != v2 && V1 != v3) {
				x3_GlobInd(hi) = V1; x3_LocInd(hi, 1) = 0;
			}
			else if (V2 != v1 && V2 != v2 && V2 != v3) {
				x3_GlobInd(hi) = V2; x3_LocInd(hi, 1) = 1;
			}
			else {
				x3_GlobInd(hi) = V3; x3_LocInd(hi, 1) = 2;
			}
		}

		if (V1 == x0_GlobInd(hi))
			x0_LocInd(hi, 1) = 0;
		else if (V2 == x0_GlobInd(hi))
			x0_LocInd(hi, 1) = 1;
		else if (V3 == x0_GlobInd(hi))
			x0_LocInd(hi, 1) = 2;

		if (V1 == x1_GlobInd(hi))
			x1_LocInd(hi, 1) = 0;
		else if (V2 == x1_GlobInd(hi))
			x1_LocInd(hi, 1) = 1;
		else if (V3 == x1_GlobInd(hi))
			x1_LocInd(hi, 1) = 2;
	}
}

void AuxBendingNormal::Incr_HingesWeights(
	const std::vector<int> faces_indices,
	const double add)
{
	for (int fi : faces_indices) {
		std::vector<int> H = OptimizationUtils::FaceToHinge_indices(hinges_faceIndex, faces_indices, fi);
		for (int hi : H) {
			cuda_ABN->weight_PerHinge.host_arr[hi] += add;
			if (cuda_ABN->weight_PerHinge.host_arr[hi] <= 1) {
				cuda_ABN->weight_PerHinge.host_arr[hi] = 1;
			}
		}
	}
	Cuda::MemCpyHostToDevice(cuda_ABN->weight_PerHinge);
}

void AuxBendingNormal::Set_HingesWeights(
	const std::vector<int> faces_indices,
	const double value)
{
	assert(value == 0 || value == 1);
	for (int fi : faces_indices) {
		std::vector<int> H = OptimizationUtils::FaceToHinge_indices(hinges_faceIndex, faces_indices, fi);
		for (int hi : H)
			cuda_ABN->weight_PerHinge.host_arr[hi] = value;
	}
	Cuda::MemCpyHostToDevice(cuda_ABN->weight_PerHinge);
}

void AuxBendingNormal::Update_HingesSigmoid(
	const std::vector<int> faces_indices,
	const double factor)
{
	for (int fi : faces_indices) {
		std::vector<int> H = OptimizationUtils::FaceToHinge_indices(hinges_faceIndex, faces_indices, fi);
		for (int hi : H) {
			cuda_ABN->Sigmoid_PerHinge.host_arr[hi] *= factor;
			if (cuda_ABN->Sigmoid_PerHinge.host_arr[hi] > 1) {
				cuda_ABN->Sigmoid_PerHinge.host_arr[hi] = 1;
			}
		}
	}
	Cuda::MemCpyHostToDevice(cuda_ABN->Sigmoid_PerHinge);
}

void AuxBendingNormal::Reset_HingesSigmoid(const std::vector<int> faces_indices)
{
	for (int fi : faces_indices) {
		std::vector<int> H = OptimizationUtils::FaceToHinge_indices(hinges_faceIndex, faces_indices, fi);
		for (int hi : H) {
			cuda_ABN->Sigmoid_PerHinge.host_arr[hi] = 1;
		}
	}
	Cuda::MemCpyHostToDevice(cuda_ABN->Sigmoid_PerHinge);
}

void AuxBendingNormal::Clear_HingesWeights() {
	for (int hi = 0; hi < num_hinges; hi++) {
		cuda_ABN->weight_PerHinge.host_arr[hi] = 1;
	}
	Cuda::MemCpyHostToDevice(cuda_ABN->weight_PerHinge);
}

void AuxBendingNormal::Clear_HingesSigmoid() {
	for (int hi = 0; hi < num_hinges; hi++) {
		cuda_ABN->Sigmoid_PerHinge.host_arr[hi] = cuda_ABN->get_SigmoidParameter();
	}
	Cuda::MemCpyHostToDevice(cuda_ABN->Sigmoid_PerHinge);
}


void AuxBendingNormal::value(Cuda::Array<double>& curr_x)
{
	cuda_ABN->value(curr_x);
}

void AuxBendingNormal::gradient(Cuda::Array<double>& X)
{
	cuda_ABN->gradient(X);
}

