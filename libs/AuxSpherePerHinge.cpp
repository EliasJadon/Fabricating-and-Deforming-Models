#include "AuxSpherePerHinge.h"
#include <unsupported/Eigen/MatrixFunctions>
#include <igl/triangle_triangle_adjacency.h>


namespace EliasMath {
	double Phi(
		const double x,
		const double SigmoidParameter,
		const PenaltyFunction penaltyFunction)
	{
		if (penaltyFunction == PenaltyFunction::SIGMOID) {
			double x2 = pow(x, 2);
			return x2 / (x2 + SigmoidParameter);
		}
		if (penaltyFunction == PenaltyFunction::QUADRATIC)
			return pow(x, 2);
		if (penaltyFunction == PenaltyFunction::EXPONENTIAL)
			return exp(x * x);
	}
	double3 sub(const double3 a, const double3 b)
	{
		return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
	}
	double3 add(double3 a, double3 b)
	{
		return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
	}
	double dot(const double3 a, const double3 b)
	{
		return a.x * b.x + a.y * b.y + a.z * b.z;
	}
	double3 mul(const double a, const double3 b)
	{
		return make_double3(a * b.x, a * b.y, a * b.z);
	}
	double squared_norm(const double3 a)
	{
		return dot(a, a);
	}
	double norm(const double3 a)
	{
		return sqrt(squared_norm(a));
	}
	double3 normalize(const double3 a)
	{
		return mul(1.0f / norm(a), a);
	}
	double3 cross(const double3 a, const double3 b)
	{
		return make_double3(
			a.y * b.z - a.z * b.y,
			a.z * b.x - a.x * b.z,
			a.x * b.y - a.y * b.x
		);
	}
	double dPhi_dm(
		const double x,
		const double SigmoidParameter,
		const PenaltyFunction PenaltyFunction)
	{
		if (PenaltyFunction == PenaltyFunction::SIGMOID)
			return (2 * x * SigmoidParameter) / pow(x * x + SigmoidParameter, 2);
		if (PenaltyFunction == PenaltyFunction::QUADRATIC)
			return 2 * x;
		if (PenaltyFunction == PenaltyFunction::EXPONENTIAL)
			return 2 * x * exp(x * x);
	}
}






AuxSpherePerHinge::AuxSpherePerHinge(
	const Eigen::MatrixXd& V, 
	const Eigen::MatrixX3i& F,
	const PenaltyFunction type)
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
	Efi.setZero();

	//Init Cuda variables
	const unsigned int numF = restShapeF.rows();
	const unsigned int numV = restShapeV.rows();
	const unsigned int numH = num_hinges;
	cuda_ASH = std::make_shared<Cuda_AuxSpherePerHinge>(type, numF, numV, numH);
	internalInitCuda();
	std::cout << "\t" << name << " constructor" << std::endl;
}

AuxSpherePerHinge::~AuxSpherePerHinge() {
	std::cout << "\t" << name << " destructor" << std::endl;
}

void AuxSpherePerHinge::internalInitCuda() {
	//init host buffers...
	for (int i = 0; i < cuda_ASH->grad.size; i++) {
		cuda_ASH->grad.host_arr[i] = 0;
	}
	for (int f = 0; f < restShapeF.rows(); f++) {
		cuda_ASH->restShapeF.host_arr[f] = make_int3(restShapeF(f, 0), restShapeF(f, 1), restShapeF(f, 2));
		cuda_ASH->restAreaPerFace.host_arr[f] = restAreaPerFace[f];
	}
	for (int h = 0; h < num_hinges; h++) {
		cuda_ASH->weight_PerHinge.host_arr[h] = 1;
		cuda_ASH->Sigmoid_PerHinge.host_arr[h] = cuda_ASH->get_SigmoidParameter();
		cuda_ASH->restAreaPerHinge.host_arr[h] = restAreaPerHinge[h];
		cuda_ASH->hinges_faceIndex.host_arr[h] = Cuda::newHinge(hinges_faceIndex[h][0], hinges_faceIndex[h][1]);
		cuda_ASH->x0_GlobInd.host_arr[h] = x0_GlobInd[h];
		cuda_ASH->x1_GlobInd.host_arr[h] = x1_GlobInd[h];
		cuda_ASH->x2_GlobInd.host_arr[h] = x2_GlobInd[h];
		cuda_ASH->x3_GlobInd.host_arr[h] = x3_GlobInd[h];
		cuda_ASH->x0_LocInd.host_arr[h] = Cuda::newHinge(x0_LocInd(h, 0), x0_LocInd(h, 1));
		cuda_ASH->x1_LocInd.host_arr[h] = Cuda::newHinge(x1_LocInd(h, 0), x1_LocInd(h, 1));
		cuda_ASH->x2_LocInd.host_arr[h] = Cuda::newHinge(x2_LocInd(h, 0), x2_LocInd(h, 1));
		cuda_ASH->x3_LocInd.host_arr[h] = Cuda::newHinge(x3_LocInd(h, 0), x3_LocInd(h, 1));
	}

	// Copy input vectors from host memory to GPU buffers.
	Cuda::MemCpyHostToDevice(cuda_ASH->grad);
	Cuda::MemCpyHostToDevice(cuda_ASH->restShapeF);
	Cuda::MemCpyHostToDevice(cuda_ASH->restAreaPerFace);
	Cuda::MemCpyHostToDevice(cuda_ASH->weight_PerHinge);
	Cuda::MemCpyHostToDevice(cuda_ASH->Sigmoid_PerHinge);
	Cuda::MemCpyHostToDevice(cuda_ASH->restAreaPerHinge);
	Cuda::MemCpyHostToDevice(cuda_ASH->hinges_faceIndex);
	Cuda::MemCpyHostToDevice(cuda_ASH->x0_GlobInd);
	Cuda::MemCpyHostToDevice(cuda_ASH->x1_GlobInd);
	Cuda::MemCpyHostToDevice(cuda_ASH->x2_GlobInd);
	Cuda::MemCpyHostToDevice(cuda_ASH->x3_GlobInd);
	Cuda::MemCpyHostToDevice(cuda_ASH->x0_LocInd);
	Cuda::MemCpyHostToDevice(cuda_ASH->x1_LocInd);
	Cuda::MemCpyHostToDevice(cuda_ASH->x2_LocInd);
	Cuda::MemCpyHostToDevice(cuda_ASH->x3_LocInd);
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
			cuda_ASH->weight_PerHinge.host_arr[hi] += add;
			if (cuda_ASH->weight_PerHinge.host_arr[hi] <= 1) {
				cuda_ASH->weight_PerHinge.host_arr[hi] = 1;
			}
		}
	}
	Cuda::MemCpyHostToDevice(cuda_ASH->weight_PerHinge);
}

void AuxSpherePerHinge::Set_HingesWeights(
	const std::vector<int> faces_indices, 
	const double value) 
{
	assert(value == 0 || value == 1);
	for (int fi : faces_indices) {
		std::vector<int> H = OptimizationUtils::FaceToHinge_indices(hinges_faceIndex, faces_indices, fi);
		for (int hi : H)
			cuda_ASH->weight_PerHinge.host_arr[hi] = value;
	}
	Cuda::MemCpyHostToDevice(cuda_ASH->weight_PerHinge);
}

void AuxSpherePerHinge::Update_HingesSigmoid(
	const std::vector<int> faces_indices, 
	const double factor) 
{
	for (int fi : faces_indices) {
		std::vector<int> H = OptimizationUtils::FaceToHinge_indices(hinges_faceIndex, faces_indices, fi);
		for (int hi : H) {
			cuda_ASH->Sigmoid_PerHinge.host_arr[hi] *= factor;
			if (cuda_ASH->Sigmoid_PerHinge.host_arr[hi] > 1) {
				cuda_ASH->Sigmoid_PerHinge.host_arr[hi] = 1;
			}
		}
	}
	Cuda::MemCpyHostToDevice(cuda_ASH->Sigmoid_PerHinge);
}

void AuxSpherePerHinge::Reset_HingesSigmoid(const std::vector<int> faces_indices)
{
	for (int fi : faces_indices) {
		std::vector<int> H = OptimizationUtils::FaceToHinge_indices(hinges_faceIndex, faces_indices, fi);
		for (int hi : H) {
			cuda_ASH->Sigmoid_PerHinge.host_arr[hi] = 1;
		}
	}
	Cuda::MemCpyHostToDevice(cuda_ASH->Sigmoid_PerHinge);
}

void AuxSpherePerHinge::Clear_HingesWeights() {
	for (int hi = 0; hi < num_hinges; hi++) {
		cuda_ASH->weight_PerHinge.host_arr[hi] = 1;
	}
	Cuda::MemCpyHostToDevice(cuda_ASH->weight_PerHinge);
}

void AuxSpherePerHinge::Clear_HingesSigmoid() {
	for (int hi = 0; hi < num_hinges; hi++) {
		cuda_ASH->Sigmoid_PerHinge.host_arr[hi] = cuda_ASH->get_SigmoidParameter();
	}
	Cuda::MemCpyHostToDevice(cuda_ASH->Sigmoid_PerHinge);
}

void AuxSpherePerHinge::value(Cuda::Array<double>& curr_x)
{
	cuda_ASH->value(curr_x);
	return;
	
	Cuda::MemCpyDeviceToHost(curr_x);
	cuda_ASH->EnergyAtomic.host_arr[0] = 0;

	auto& I = cuda_ASH->mesh_indices;
	for (int hi = 0; hi < num_hinges; hi++) {
		int f0 = cuda_ASH->hinges_faceIndex.host_arr[hi].f0;
		int f1 = cuda_ASH->hinges_faceIndex.host_arr[hi].f1;
		double R0 = curr_x.host_arr[f0 + I.startR];
		double R1 = curr_x.host_arr[f1 + I.startR];
		double3 C0 = make_double3(
			curr_x.host_arr[f0 + I.startCx],
			curr_x.host_arr[f0 + I.startCy],
			curr_x.host_arr[f0 + I.startCz]
		);
		double3 C1 = make_double3(
			curr_x.host_arr[f1 + I.startCx],
			curr_x.host_arr[f1 + I.startCy],
			curr_x.host_arr[f1 + I.startCz]
		);
		double d_center = EliasMath::squared_norm(EliasMath::sub(C1, C0));
		double d_radius = pow(R1 - R0, 2);
		cuda_ASH->EnergyAtomic.host_arr[0] += cuda_ASH->w1 * restAreaPerHinge[hi] * cuda_ASH->weight_PerHinge.host_arr[hi] *
			EliasMath::Phi(d_center + d_radius, cuda_ASH->Sigmoid_PerHinge.host_arr[hi], cuda_ASH->penaltyFunction);
	}
	

	for (int fi = 0; fi < restShapeF.rows(); fi++) {
		const unsigned int x0 = cuda_ASH->restShapeF.host_arr[fi].x;
		const unsigned int x1 = cuda_ASH->restShapeF.host_arr[fi].y;
		const unsigned int x2 = cuda_ASH->restShapeF.host_arr[fi].z;
		double3 V0 = make_double3(
			curr_x.host_arr[x0 + I.startVx],
			curr_x.host_arr[x0 + I.startVy],
			curr_x.host_arr[x0 + I.startVz]
		);
		double3 V1 = make_double3(
			curr_x.host_arr[x1 + I.startVx],
			curr_x.host_arr[x1 + I.startVy],
			curr_x.host_arr[x1 + I.startVz]
		);
		double3 V2 = make_double3(
			curr_x.host_arr[x2 + I.startVx],
			curr_x.host_arr[x2 + I.startVy],
			curr_x.host_arr[x2 + I.startVz]
		);
		double3 C = make_double3(
			curr_x.host_arr[fi + I.startCx],
			curr_x.host_arr[fi + I.startCy],
			curr_x.host_arr[fi + I.startCz]
		);
		double R = curr_x.host_arr[fi + I.startR];
		double res =
			pow(EliasMath::squared_norm(EliasMath::sub(V0, C)) - pow(R, 2), 2) +
			pow(EliasMath::squared_norm(EliasMath::sub(V1, C)) - pow(R, 2), 2) +
			pow(EliasMath::squared_norm(EliasMath::sub(V2, C)) - pow(R, 2), 2);
		cuda_ASH->EnergyAtomic.host_arr[0] += cuda_ASH->w2 * res;
	}
	Cuda::MemCpyHostToDevice(cuda_ASH->EnergyAtomic);
}

void AuxSpherePerHinge::gradient(Cuda::Array<double>& X)
{
	Cuda::MemCpyDeviceToHost(X);
	for (int i = 0; i < cuda_ASH->grad.size; i++)
		cuda_ASH->grad.host_arr[i] = 0;

	for (int hi = 0; hi < num_hinges; hi++) {
		int f0 = cuda_ASH->hinges_faceIndex.host_arr[hi].f0;
		int f1 = cuda_ASH->hinges_faceIndex.host_arr[hi].f1;
		if (f0 >= cuda_ASH->mesh_indices.num_faces || f1 >= cuda_ASH->mesh_indices.num_faces) {
			std::cerr << "Error: in AuxSpherePerHinge::gradient!";
			exit(-1);
		}
		double R0 = X.host_arr[f0 + cuda_ASH->mesh_indices.startR];
		double R1 = X.host_arr[f1 + cuda_ASH->mesh_indices.startR];
		double3 C0 = make_double3(
			X.host_arr[f0 + cuda_ASH->mesh_indices.startCx],
			X.host_arr[f0 + cuda_ASH->mesh_indices.startCy],
			X.host_arr[f0 + cuda_ASH->mesh_indices.startCz]
		);
		double3 C1 = make_double3(
			X.host_arr[f1 + cuda_ASH->mesh_indices.startCx],
			X.host_arr[f1 + cuda_ASH->mesh_indices.startCy],
			X.host_arr[f1 + cuda_ASH->mesh_indices.startCz]
		);
		double d_center = EliasMath::squared_norm(EliasMath::sub(C1, C0));
		double d_radius = pow(R1 - R0, 2);
		double coeff = 2 * cuda_ASH->w1 * restAreaPerHinge[hi] * cuda_ASH->weight_PerHinge.host_arr[hi] *
			EliasMath::dPhi_dm(d_center + d_radius, cuda_ASH->Sigmoid_PerHinge.host_arr[hi], cuda_ASH->penaltyFunction);

		cuda_ASH->grad.host_arr[f0 + cuda_ASH->mesh_indices.startCx] += (C0.x - C1.x) * coeff; //C0.x
		cuda_ASH->grad.host_arr[f0 + cuda_ASH->mesh_indices.startCy] += (C0.y - C1.y) * coeff;	//C0.y
		cuda_ASH->grad.host_arr[f0 + cuda_ASH->mesh_indices.startCz] += (C0.z - C1.z) * coeff;	//C0.z
		cuda_ASH->grad.host_arr[f1 + cuda_ASH->mesh_indices.startCx] += (C1.x - C0.x) * coeff;	//C1.x
		cuda_ASH->grad.host_arr[f1 + cuda_ASH->mesh_indices.startCy] += (C1.y - C0.y) * coeff;	//C1.y
		cuda_ASH->grad.host_arr[f1 + cuda_ASH->mesh_indices.startCz] += (C1.z - C0.z) * coeff;	//C1.z
		cuda_ASH->grad.host_arr[f0 + cuda_ASH->mesh_indices.startR] += (R0 - R1) * coeff;		//r0
		cuda_ASH->grad.host_arr[f1 + cuda_ASH->mesh_indices.startR] += (R1 - R0) * coeff;		//r1
	}
	

	for (int fi = 0; fi < restShapeF.rows(); fi++) {
		const unsigned int x0 = cuda_ASH->restShapeF.host_arr[fi].x;
		const unsigned int x1 = cuda_ASH->restShapeF.host_arr[fi].y;
		const unsigned int x2 = cuda_ASH->restShapeF.host_arr[fi].z;
		double3 V0 = make_double3(
			X.host_arr[x0 + cuda_ASH->mesh_indices.startVx],
			X.host_arr[x0 + cuda_ASH->mesh_indices.startVy],
			X.host_arr[x0 + cuda_ASH->mesh_indices.startVz]
		);
		double3 V1 = make_double3(
			X.host_arr[x1 + cuda_ASH->mesh_indices.startVx],
			X.host_arr[x1 + cuda_ASH->mesh_indices.startVy],
			X.host_arr[x1 + cuda_ASH->mesh_indices.startVz]
		);
		double3 V2 = make_double3(
			X.host_arr[x2 + cuda_ASH->mesh_indices.startVx],
			X.host_arr[x2 + cuda_ASH->mesh_indices.startVy],
			X.host_arr[x2 + cuda_ASH->mesh_indices.startVz]
		);
		double3 C = make_double3(
			X.host_arr[fi + cuda_ASH->mesh_indices.startCx],
			X.host_arr[fi + cuda_ASH->mesh_indices.startCy],
			X.host_arr[fi + cuda_ASH->mesh_indices.startCz]
		);
		double R = X.host_arr[fi + cuda_ASH->mesh_indices.startR];

		double coeff = cuda_ASH->w2 * 4;
		double E0 = coeff * (EliasMath::squared_norm(EliasMath::sub(V0, C)) - pow(R, 2));
		double E1 = coeff * (EliasMath::squared_norm(EliasMath::sub(V1, C)) - pow(R, 2));
		double E2 = coeff * (EliasMath::squared_norm(EliasMath::sub(V2, C)) - pow(R, 2));

		cuda_ASH->grad.host_arr[x0 + cuda_ASH->mesh_indices.startVx] += E0 * (V0.x - C.x); // V0x
		cuda_ASH->grad.host_arr[x0 + cuda_ASH->mesh_indices.startVy] += E0 * (V0.y - C.y); // V0y
		cuda_ASH->grad.host_arr[x0 + cuda_ASH->mesh_indices.startVz] += E0 * (V0.z - C.z); // V0z
		cuda_ASH->grad.host_arr[x1 + cuda_ASH->mesh_indices.startVx] += E1 * (V1.x - C.x); // V1x
		cuda_ASH->grad.host_arr[x1 + cuda_ASH->mesh_indices.startVy] += E1 * (V1.y - C.y); // V1y
		cuda_ASH->grad.host_arr[x1 + cuda_ASH->mesh_indices.startVz] += E1 * (V1.z - C.z); // V1z
		cuda_ASH->grad.host_arr[x2 + cuda_ASH->mesh_indices.startVx] += E2 * (V2.x - C.x); // V2x
		cuda_ASH->grad.host_arr[x2 + cuda_ASH->mesh_indices.startVy] += E2 * (V2.y - C.y); // V2y
		cuda_ASH->grad.host_arr[x2 + cuda_ASH->mesh_indices.startVz] += E2 * (V2.z - C.z); // V2z
		cuda_ASH->grad.host_arr[fi + cuda_ASH->mesh_indices.startCx] +=
			(E0 * (C.x - V0.x)) +
				(E1 * (C.x - V1.x)) +
				(E2 * (C.x - V2.x)); // Cx
		cuda_ASH->grad.host_arr[fi + cuda_ASH->mesh_indices.startCy] +=
			(E0 * (C.y - V0.y)) +
				(E1 * (C.y - V1.y)) +
				(E2 * (C.y - V2.y)); // Cy
		cuda_ASH->grad.host_arr[fi + cuda_ASH->mesh_indices.startCz] +=
			(E0 * (C.z - V0.z)) +
				(E1 * (C.z - V1.z)) +
				(E2 * (C.z - V2.z)); // Cz
		cuda_ASH->grad.host_arr[fi + cuda_ASH->mesh_indices.startR] +=
			(E0 * (-1) * R) +
				(E1 * (-1) * R) +
				(E2 * (-1) * R); //r
	}
	Cuda::MemCpyHostToDevice(cuda_ASH->grad);
}
