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

	//Initialize rest variables (X0) m
	calculateHinges();
	restAreaPerHinge.resize(num_hinges);
	igl::doublearea(restShapeV, restShapeF, restAreaPerFace);
	restAreaPerFace /= 2;
	for (int hi = 0; hi < num_hinges; hi++) {
		int f0 = hinges_faceIndex[hi](0);
		int f1 = hinges_faceIndex[hi](1);
		restAreaPerHinge(hi) = (restAreaPerFace(f0) + restAreaPerFace(f1)) / 2;
	}
	

	//Init Cuda variables
	const unsigned int numF = restShapeF.rows();
	const unsigned int numV = restShapeV.rows();
	const unsigned int numH = num_hinges;
	this->penaltyFunction = penaltyFunction;
	SigmoidParameter = 1;

	Cuda::AllocateMemory(weight_PerHinge, numH);
	Cuda::AllocateMemory(Sigmoid_PerHinge, numH);
	//init host buffers...
	for (int h = 0; h < num_hinges; h++) {
		weight_PerHinge.host_arr[h] = 1;
		Sigmoid_PerHinge.host_arr[h] = get_SigmoidParameter();
	}
	std::cout << "\t" << name << " constructor" << std::endl;
}

AuxBendingNormal::~AuxBendingNormal() {
	Cuda::FreeMemory(weight_PerHinge);
	Cuda::FreeMemory(Sigmoid_PerHinge);
	std::cout << "\t" << name << " destructor" << std::endl;
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
			weight_PerHinge.host_arr[hi] += add;
			if (weight_PerHinge.host_arr[hi] <= 1) {
				weight_PerHinge.host_arr[hi] = 1;
			}
		}
	}
}

void AuxBendingNormal::Set_HingesWeights(
	const std::vector<int> faces_indices,
	const double value)
{
	assert(value == 0 || value == 1);
	for (int fi : faces_indices) {
		std::vector<int> H = OptimizationUtils::FaceToHinge_indices(hinges_faceIndex, faces_indices, fi);
		for (int hi : H)
			weight_PerHinge.host_arr[hi] = value;
	}
}

void AuxBendingNormal::Update_HingesSigmoid(
	const std::vector<int> faces_indices,
	const double factor)
{
	for (int fi : faces_indices) {
		std::vector<int> H = OptimizationUtils::FaceToHinge_indices(hinges_faceIndex, faces_indices, fi);
		for (int hi : H) {
			Sigmoid_PerHinge.host_arr[hi] *= factor;
			if (Sigmoid_PerHinge.host_arr[hi] > 1) {
				Sigmoid_PerHinge.host_arr[hi] = 1;
			}
		}
	}
}

void AuxBendingNormal::Reset_HingesSigmoid(const std::vector<int> faces_indices)
{
	for (int fi : faces_indices) {
		std::vector<int> H = OptimizationUtils::FaceToHinge_indices(hinges_faceIndex, faces_indices, fi);
		for (int hi : H) {
			Sigmoid_PerHinge.host_arr[hi] = 1;
		}
	}
}

void AuxBendingNormal::Clear_HingesWeights() {
	for (int hi = 0; hi < num_hinges; hi++) {
		weight_PerHinge.host_arr[hi] = 1;
	}
}

void AuxBendingNormal::Clear_HingesSigmoid() {
	for (int hi = 0; hi < num_hinges; hi++) {
		Sigmoid_PerHinge.host_arr[hi] = get_SigmoidParameter();
	}
}


double AuxBendingNormal::value(Cuda::Array<double>& curr_x, const bool update)
{
	double value = 0;
	for (int hi = 0; hi < num_hinges; hi++) {
		int f0 = hinges_faceIndex[hi][0];
		int f1 = hinges_faceIndex[hi][1];
		double3 N0 = getN(curr_x, f0);
		double3 N1 = getN(curr_x, f1);
		double3 diff = sub(N1, N0);
		double d_normals = squared_norm(diff);
		value += w1 * restAreaPerHinge[hi] * weight_PerHinge.host_arr[hi] *
			Phi(d_normals, Sigmoid_PerHinge.host_arr[hi], penaltyFunction);
	}

	for (int fi = 0; fi < mesh_indices.num_faces; fi++) {
		// (N^T*(x1-x0))^2 + (N^T*(x2-x1))^2 + (N^T*(x0-x2))^2
		const int x0 = restShapeF(fi, 0);
		const int x1 = restShapeF(fi, 1);
		const int x2 = restShapeF(fi, 2);
		double3 V0 = getV(curr_x, x0);
		double3 V1 = getV(curr_x, x1);
		double3 V2 = getV(curr_x, x2);
		double3 N = getN(curr_x, fi);
		
		double3 e21 = sub(V2, V1);
		double3 e10 = sub(V1, V0);
		double3 e02 = sub(V0, V2);
		value += w3 * (pow(dot(N, e21), 2) + pow(dot(N, e10), 2) + pow(dot(N, e02), 2));
		value += pow(squared_norm(N) - 1, 2) * w2;
	}
	
	if (update)
		energy_value = value;
	return value;
}

void AuxBendingNormal::gradient(Cuda::Array<double>& X, const bool update)
{
	for (int i = 0; i < grad.size; i++) {
		grad.host_arr[i] = 0;
	}

	for (int hi = 0; hi < num_hinges; hi++) {
		int f0 = hinges_faceIndex[hi][0];
		int f1 = hinges_faceIndex[hi][1];
		double3 N0 = getN(X, f0); 
		double3 N1 = getN(X, f1); 
		double3 diff = sub(N1, N0);
		double d_normals = squared_norm(diff);

		double coeff = 2 * w1 * restAreaPerHinge[hi] * weight_PerHinge.host_arr[hi] *
			dPhi_dm(d_normals, Sigmoid_PerHinge.host_arr[hi], penaltyFunction);

		grad.host_arr[f0 + mesh_indices.startNx] += coeff * (N0.x - N1.x);
		grad.host_arr[f1 + mesh_indices.startNx] += coeff * (N1.x - N0.x);
		grad.host_arr[f0 + mesh_indices.startNy] += coeff * (N0.y - N1.y);
		grad.host_arr[f1 + mesh_indices.startNy] += coeff * (N1.y - N0.y);
		grad.host_arr[f0 + mesh_indices.startNz] += coeff * (N0.z - N1.z);
		grad.host_arr[f1 + mesh_indices.startNz] += coeff * (N1.z - N0.z);
	}
	

	for (int fi = 0; fi < mesh_indices.num_faces; fi++) {
		const unsigned int x0 = restShapeF(fi, 0);
		const unsigned int x1 = restShapeF(fi, 1);
		const unsigned int x2 = restShapeF(fi, 2);
		double3 V0 = getV(X, x0); 
		double3 V1 = getV(X, x1); 
		double3 V2 = getV(X, x2); 
		double3 N = getN(X, fi); 

		double3 e21 = sub(V2, V1);
		double3 e10 = sub(V1, V0);
		double3 e02 = sub(V0, V2);
		double N02 = dot(N, e02);
		double N10 = dot(N, e10);
		double N21 = dot(N, e21);
		double coeff = 2 * w3;
		double coeff2 = w2 * 4 * (squared_norm(N) - 1);

		grad.host_arr[x0 + mesh_indices.startVx] += coeff * N.x * (N02 - N10);
		grad.host_arr[x0 + mesh_indices.startVy] += coeff * N.y * (N02 - N10);
		grad.host_arr[x0 + mesh_indices.startVz] += coeff * N.z * (N02 - N10);
		grad.host_arr[x1 + mesh_indices.startVx] += coeff * N.x * (N10 - N21);
		grad.host_arr[x1 + mesh_indices.startVy] += coeff * N.y * (N10 - N21);
		grad.host_arr[x1 + mesh_indices.startVz] += coeff * N.z * (N10 - N21);
		grad.host_arr[x2 + mesh_indices.startVx] += coeff * N.x * (N21 - N02);
		grad.host_arr[x2 + mesh_indices.startVy] += coeff * N.y * (N21 - N02);
		grad.host_arr[x2 + mesh_indices.startVz] += coeff * N.z * (N21 - N02);
		grad.host_arr[fi + mesh_indices.startNx] += (coeff2 * N.x) + (coeff * (N10 * e10.x + N21 * e21.x + N02 * e02.x));
		grad.host_arr[fi + mesh_indices.startNy] += (coeff2 * N.y) + (coeff * (N10 * e10.y + N21 * e21.y + N02 * e02.y));
		grad.host_arr[fi + mesh_indices.startNz] += (coeff2 * N.z) + (coeff * (N10 * e10.z + N21 * e21.z + N02 * e02.z));
	}
	
	if (update) {
		gradient_norm = 0;
		for (int i = 0; i < grad.size; i++)
			gradient_norm += pow(grad.host_arr[i], 2);
	}
}

