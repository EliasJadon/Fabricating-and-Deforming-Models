#include "AuxSpherePerHinge.h"
#include <unsupported/Eigen/MatrixFunctions>
#include <igl/triangle_triangle_adjacency.h>

AuxSpherePerHinge::AuxSpherePerHinge(
	const Eigen::MatrixXd& V, 
	const Eigen::MatrixX3i& F,
	const Cuda::PenaltyFunction type)
{
	init_mesh(V, F);
	name = "Aux Sphere Per Hinge";
	w = 0;
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

	//Init Cuda variables
	const unsigned int numF = restShapeF.rows();
	const unsigned int numV = restShapeV.rows();
	const unsigned int numH = num_hinges;
	Cuda::AllocateMemory(weight_PerHinge, num_hinges);
	Cuda::AllocateMemory(Sigmoid_PerHinge, num_hinges);

	for (int h = 0; h < num_hinges; h++) {
		weight_PerHinge.host_arr[h] = 1;
		Sigmoid_PerHinge.host_arr[h] = get_SigmoidParameter();
	}
	std::cout << "\t" << name << " constructor" << std::endl;
}

AuxSpherePerHinge::~AuxSpherePerHinge() {
	Cuda::FreeMemory(weight_PerHinge);
	Cuda::FreeMemory(Sigmoid_PerHinge);
	std::cout << "\t" << name << " destructor" << std::endl;
}

void AuxSpherePerHinge::calculateHinges() {
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

void AuxSpherePerHinge::Incr_HingesWeights(
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

void AuxSpherePerHinge::Set_HingesWeights(
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

void AuxSpherePerHinge::Update_HingesSigmoid(
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

void AuxSpherePerHinge::Reset_HingesSigmoid(const std::vector<int> faces_indices)
{
	for (int fi : faces_indices) {
		std::vector<int> H = OptimizationUtils::FaceToHinge_indices(hinges_faceIndex, faces_indices, fi);
		for (int hi : H) {
			Sigmoid_PerHinge.host_arr[hi] = 1;
		}
	}
}

void AuxSpherePerHinge::Clear_HingesWeights() {
	for (int hi = 0; hi < num_hinges; hi++) {
		weight_PerHinge.host_arr[hi] = 1;
	}
}

void AuxSpherePerHinge::Clear_HingesSigmoid() {
	for (int hi = 0; hi < num_hinges; hi++) {
		Sigmoid_PerHinge.host_arr[hi] = get_SigmoidParameter();
	}
}

double AuxSpherePerHinge::value(Cuda::Array<double>& curr_x, const bool update)
{	
	double value = 0;

	for (int hi = 0; hi < num_hinges; hi++) {
		int f0 = hinges_faceIndex[hi][0];
		int f1 = hinges_faceIndex[hi][1];
		double R0 = getR(curr_x, f0); 
		double R1 = getR(curr_x, f1); 
		double3 C0 = getC(curr_x, f0);
		double3 C1 = getC(curr_x, f1);
			
		double d_center = squared_norm(sub(C1, C0));
		double d_radius = pow(R1 - R0, 2);
		value += w1 * restAreaPerHinge[hi] * weight_PerHinge.host_arr[hi] *
			Phi(d_center + d_radius, Sigmoid_PerHinge.host_arr[hi], penaltyFunction);
	}
	
	for (int fi = 0; fi < restShapeF.rows(); fi++) {
		const unsigned int x0 = restShapeF(fi,0);
		const unsigned int x1 = restShapeF(fi,1);
		const unsigned int x2 = restShapeF(fi,2);
		double3 V0 = getV(curr_x, x0);
		double3 V1 = getV(curr_x, x1);
		double3 V2 = getV(curr_x, x2);
		double3 C = getC(curr_x, fi);
		double R = getR(curr_x, fi);
		
		double res =
			pow(squared_norm(sub(V0, C)) - pow(R, 2), 2) +
			pow(squared_norm(sub(V1, C)) - pow(R, 2), 2) +
			pow(squared_norm(sub(V2, C)) - pow(R, 2), 2);

		value += w2 * res;
	}
	if (update)
		energy_value = value;
	return value;
}

void AuxSpherePerHinge::gradient(Cuda::Array<double>& X, const bool update)
{
	for (int i = 0; i < grad.size; i++)
		grad.host_arr[i] = 0;

	for (int hi = 0; hi < num_hinges; hi++) {
		int f0 = hinges_faceIndex[hi][0];
		int f1 = hinges_faceIndex[hi][1];
		
		double R0 = getR(X, f0);
		double R1 = getR(X, f1);
		double3 C0 = getC(X, f0); 
		double3 C1 = getC(X, f1); 
		
		double d_center = squared_norm(sub(C1, C0));
		double d_radius = pow(R1 - R0, 2);
		double coeff = 2 * w1 * restAreaPerHinge[hi] * weight_PerHinge.host_arr[hi] *
			dPhi_dm(d_center + d_radius, Sigmoid_PerHinge.host_arr[hi], penaltyFunction);

		grad.host_arr[f0 + mesh_indices.startCx] += (C0.x - C1.x) * coeff; //C0.x
		grad.host_arr[f0 + mesh_indices.startCy] += (C0.y - C1.y) * coeff;	//C0.y
		grad.host_arr[f0 + mesh_indices.startCz] += (C0.z - C1.z) * coeff;	//C0.z
		grad.host_arr[f1 + mesh_indices.startCx] += (C1.x - C0.x) * coeff;	//C1.x
		grad.host_arr[f1 + mesh_indices.startCy] += (C1.y - C0.y) * coeff;	//C1.y
		grad.host_arr[f1 + mesh_indices.startCz] += (C1.z - C0.z) * coeff;	//C1.z
		grad.host_arr[f0 + mesh_indices.startR] += (R0 - R1) * coeff;		//r0
		grad.host_arr[f1 + mesh_indices.startR] += (R1 - R0) * coeff;		//r1
	}
	

	for (int fi = 0; fi < restShapeF.rows(); fi++) {
		const unsigned int x0 = restShapeF(fi, 0);
		const unsigned int x1 = restShapeF(fi, 1);
		const unsigned int x2 = restShapeF(fi, 2);
		
		double3 V0 = getV(X, x0); 
		double3 V1 = getV(X, x1); 
		double3 V2 = getV(X, x2); 
		double3 C = getC(X, fi); 
		double R = getR(X, fi); 
		
		double coeff = w2 * 4;
		double E0 = coeff * (squared_norm(sub(V0, C)) - pow(R, 2));
		double E1 = coeff * (squared_norm(sub(V1, C)) - pow(R, 2));
		double E2 = coeff * (squared_norm(sub(V2, C)) - pow(R, 2));

		grad.host_arr[x0 + mesh_indices.startVx] += E0 * (V0.x - C.x); // V0x
		grad.host_arr[x0 + mesh_indices.startVy] += E0 * (V0.y - C.y); // V0y
		grad.host_arr[x0 + mesh_indices.startVz] += E0 * (V0.z - C.z); // V0z
		grad.host_arr[x1 + mesh_indices.startVx] += E1 * (V1.x - C.x); // V1x
		grad.host_arr[x1 + mesh_indices.startVy] += E1 * (V1.y - C.y); // V1y
		grad.host_arr[x1 + mesh_indices.startVz] += E1 * (V1.z - C.z); // V1z
		grad.host_arr[x2 + mesh_indices.startVx] += E2 * (V2.x - C.x); // V2x
		grad.host_arr[x2 + mesh_indices.startVy] += E2 * (V2.y - C.y); // V2y
		grad.host_arr[x2 + mesh_indices.startVz] += E2 * (V2.z - C.z); // V2z
		grad.host_arr[fi + mesh_indices.startCx] +=
			(E0 * (C.x - V0.x)) +
				(E1 * (C.x - V1.x)) +
				(E2 * (C.x - V2.x)); // Cx
		grad.host_arr[fi + mesh_indices.startCy] +=
			(E0 * (C.y - V0.y)) +
				(E1 * (C.y - V1.y)) +
				(E2 * (C.y - V2.y)); // Cy
		grad.host_arr[fi + mesh_indices.startCz] +=
			(E0 * (C.z - V0.z)) +
				(E1 * (C.z - V1.z)) +
				(E2 * (C.z - V2.z)); // Cz
		grad.host_arr[fi + mesh_indices.startR] +=
			(E0 * (-1) * R) +
				(E1 * (-1) * R) +
				(E2 * (-1) * R); //r
	}

	if (update) {
		gradient_norm = 0;
		for (int i = 0; i < grad.size; i++)
			gradient_norm += pow(grad.host_arr[i], 2);
	}
}
