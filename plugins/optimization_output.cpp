#include "optimization_output.h"

OptimizationOutput::OptimizationOutput(
	igl::opengl::glfw::Viewer* viewer,
	const MinimizerType minimizer_type,
	const OptimizationUtils::LineSearch linesearchType)
{
	//update viewer
	CoreID = viewer->append_core(Eigen::Vector4f::Zero());
	viewer->core(CoreID).background_color = Eigen::Vector4f(1, 1, 1, 0);
	viewer->core(CoreID).is_animating = true;
	viewer->core(CoreID).lighting_factor = 0.5;
	// Initialize minimizer thread
	minimizer = std::make_shared<Minimizer>(CoreID);
	minimizer->lineSearch_type = linesearchType;
	updateActiveMinimizer(minimizer_type);
	totalObjective = std::make_shared<TotalObjective>();
	showFacesNorm = showSphereEdges = showNormEdges =
		showTriangleCenters = showSphereCenters =
		showCylinderDir = showCylinderEdges = false;
	for (int i = 0; i < 9; i++)
		UserInterface_facesGroups.push_back(FacesGroup(UserInterface_facesGroups.size()));
	UserInterface_IsTranslate = false;
}

void OptimizationOutput::setAuxVariables(
	const Eigen::MatrixXd& V,
	const Eigen::MatrixXi& F,
	const Eigen::MatrixXd& center_of_sphere,
	const Eigen::VectorXd& radius_of_sphere,
	const Eigen::MatrixXd& cylinder_dir, 
	const Eigen::MatrixXd& norm) 
{
	this->center_of_faces = OptimizationUtils::center_per_triangle(V, F);
	this->center_of_sphere = center_of_sphere;
	this->radius_of_sphere = radius_of_sphere;
	this->faces_normals = norm;
	this->cylinder_dir = cylinder_dir;
}

double OptimizationOutput::getRadiusOfSphere(int index) 
{
	return this->radius_of_sphere(index);
}

void OptimizationOutput::clustering(
	const double ratio, 
	const double MSE, 
	const bool isNormal) 
{
	std::vector<std::vector<int>> clusters_ind;
	std::vector<Eigen::RowVectorXd> clusters_val;
	std::vector<Eigen::RowVectorXd> clusters_center;
	std::vector<double> clusters_radius;
	clusters_init(ratio, MSE, clusters_val, clusters_center, clusters_radius, isNormal);

	int numFaces;
	if (isNormal)
		numFaces = faces_normals.rows();
	else
		numFaces = center_of_sphere.rows();
	//Do 5 rounds of K-means clustering alg.
	for (int _ = 0; _ < 5; _++) {
		clusters_ind.clear();
		int numClusters;
		if (isNormal)
			numClusters = clusters_val.size();
		else
			numClusters = clusters_center.size();
		clusters_ind.resize(numClusters);
		for (int fi = 0; fi < numFaces; fi++)
		{
			bool found = false;
			double minMSE = MSE;
			int argmin;
			for (int ci = 0; ci < numClusters; ci++)
			{
				double currMSE;
				if (isNormal) {
					currMSE = (faces_normals.row(fi) - clusters_val[ci]).squaredNorm();
				}
				else {
					currMSE = (ratio*((center_of_sphere.row(fi) - clusters_center[ci]).norm()) + (1 - ratio)*abs(radius_of_sphere(fi) - clusters_radius[ci]));
				}
				if (currMSE < minMSE)
				{
					minMSE = currMSE;
					argmin = ci;
					found = true;
				}
			}
			if (found)
			{
				clusters_ind[argmin].push_back(fi);
			}
			else
			{
				clusters_ind.push_back({ fi });
				if (isNormal)
					clusters_val.push_back(faces_normals.row(fi));
				else {
					clusters_center.push_back(center_of_sphere.row(fi));
					clusters_radius.push_back(radius_of_sphere(fi));
				}
			}
		}
		//Remove empty clusters
		auto& it_N = clusters_val.begin();
		auto& it_C = clusters_center.begin();
		auto& it_R = clusters_radius.begin();
		auto& it_i = clusters_ind.begin();
		while (it_i != clusters_ind.end()) {
			if (it_i->size() == 0) {
				it_i = clusters_ind.erase(it_i);
				if (isNormal)
					it_N = clusters_val.erase(it_N);
				else {
					it_C = clusters_center.erase(it_C);
					it_R = clusters_radius.erase(it_R);
				}
			}
			else {
				it_i++;
				if (isNormal)
					it_N++;
				else {
					it_R++;
					it_C++;
				}
			}
		}
		//Update average
		for (int ci = 0; ci < clusters_ind.size(); ci++) {
			if (isNormal) {
				Eigen::RowVectorXd avg;
				avg.resize(3);
				avg << 0, 0, 0;
				for (int currf : clusters_ind[ci]) {
					avg += faces_normals.row(currf);
				}
				avg /= clusters_ind[ci].size();
				clusters_val[ci] = avg;
			}
			else {
				Eigen::RowVectorXd avgC;
				double avgR = 0;
				avgC.resize(3);
				avgC << 0, 0, 0;
				for (int currf : clusters_ind[ci]) {
					avgC += center_of_sphere.row(currf);
					avgR += radius_of_sphere(currf);
				}
				avgC /= clusters_ind[ci].size();
				avgR /= clusters_ind[ci].size();
				clusters_center[ci] = avgC;
				clusters_radius[ci] = avgR;
			}
		}
		//Union similar clusters
		std::vector<Eigen::RowVectorXd>::iterator val1, val2, cent1, cent2;
		std::vector<double>::iterator radius1, radius2;
		if (isNormal) {
			val1 = clusters_val.begin();
			val2 = val1 + 1;
		}
		else {
			cent1 = clusters_center.begin();
			cent2 = cent1 + 1;
			radius1 = clusters_radius.begin();
			radius2 = radius1 + 1;
		}

		auto& ind1 = clusters_ind.begin();
		auto& ind2 = ind1 + 1;
		while (ind1 != clusters_ind.end())
		{
			if (isNormal) {
				val2 = val1 + 1;
			}
			else {
				cent2 = cent1 + 1;
				radius2 = radius1 + 1;
			}
			for (ind2 = ind1 + 1; ind2 != clusters_ind.end();)
			{
				double diff;
				if (isNormal)
					diff = (*val1 - *val2).squaredNorm();
				else
					diff = (ratio*((*cent1 - *cent2).norm()) + (1 - ratio)*abs(*radius1 - *radius2));

				if (diff < MSE) {
					for (int currf : (*ind2)) {
						ind1->push_back(currf);
					}
					if (isNormal) {
						Eigen::RowVectorXd avg;
						avg.resize(3);
						avg << 0, 0, 0;
						for (int currf : (*ind1)) {
							avg += faces_normals.row(currf);
						}
						avg /= ind1->size();
						*val1 = avg;
						val2 = clusters_val.erase(val2);
					}
					else {
						Eigen::RowVectorXd avgC;
						double avgR = 0;
						avgC.resize(3);
						avgC << 0, 0, 0;
						for (int currf : (*ind1)) {
							avgC += center_of_sphere.row(currf);
							avgR += radius_of_sphere(currf);
						}
						avgC /= ind1->size();
						avgR /= ind1->size();
						*cent1 = avgC;
						*radius1 = avgR;
						cent2 = clusters_center.erase(cent2);
						radius2 = clusters_radius.erase(radius2);
					}
					ind2 = clusters_ind.erase(ind2);
				}
				else {
					if (isNormal) {
						val2++;
					}
					else {
						cent2++;
						radius2++;
					}
					ind2++;
				}
			}

			if (isNormal)
				val1++;
			else {
				cent1++; radius1++;
			}
			ind1++;
		}
	}
	clusters_indices = clusters_ind;
}

void OptimizationOutput::clusters_init(
	const double ratio,
	const double MSE,
	std::vector<Eigen::RowVectorXd>& clusters_val,
	std::vector<Eigen::RowVectorXd>& clusters_center,
	std::vector<double>& clusters_radius,
	const bool isNormal)
{
	std::vector<std::vector<int>> clusters_ind;
	int numFaces;
	if (isNormal) {
		clusters_val.clear();
		clusters_val.push_back(faces_normals.row(0));
		numFaces = faces_normals.rows();
	}
	else {
		clusters_center.clear();
		clusters_center.push_back(center_of_sphere.row(0));
		clusters_radius.clear();
		clusters_radius.push_back(radius_of_sphere(0));
		numFaces = center_of_sphere.rows();
	}
	clusters_ind.push_back({ 0 });

	for (int fi = 1; fi < numFaces; fi++)
	{
		bool found = false;
		double minMSE = MSE;
		int argmin;
		int numClusters;
		if (isNormal)
			numClusters = clusters_val.size();
		else
			numClusters = clusters_center.size();
		for (int ci = 0; ci < numClusters; ci++)
		{
			double currMSE;
			if (isNormal) {
				currMSE = (faces_normals.row(fi) - clusters_val[ci]).squaredNorm();
			}
			else {
				currMSE = (ratio*((center_of_sphere.row(fi) - clusters_center[ci]).norm()) + (1 - ratio)*abs(radius_of_sphere(fi) - clusters_radius[ci]));
			}
			if (currMSE < minMSE)
			{
				minMSE = currMSE;
				argmin = ci;
				found = true;
			}
		}
		if (found)
		{
			clusters_ind[argmin].push_back(fi);
			if (isNormal) {
				Eigen::RowVectorXd avg;
				avg.resize(3);
				avg << 0, 0, 0;
				for (int currf : clusters_ind[argmin]) {
					avg += faces_normals.row(currf);
				}
				avg /= clusters_ind[argmin].size();
				clusters_val[argmin] << avg;
			}
			else {
				Eigen::RowVectorXd avgC;
				double avgR = 0;
				avgC.resize(3);
				avgC << 0, 0, 0;
				for (int currf : clusters_ind[argmin]) {
					avgC += center_of_sphere.row(currf);
					avgR += radius_of_sphere(currf);
				}
				avgC /= clusters_ind[argmin].size();
				avgR /= clusters_ind[argmin].size();
				clusters_center[argmin] << avgC;
				clusters_radius[argmin] = avgR;
			}
		}
		else
		{
			clusters_ind.push_back({ fi });
			if (isNormal)
				clusters_val.push_back(faces_normals.row(fi));
			else {
				clusters_center.push_back(center_of_sphere.row(fi));
				clusters_radius.push_back(radius_of_sphere(fi));
			}
		}
	}
}

void OptimizationOutput::translateFaces(
	const int fi, 
	const Eigen::Vector3d translateValue) {
	this->center_of_sphere.row(fi) += translateValue;
	this->faces_normals.row(fi) += translateValue;
}

Eigen::MatrixXd OptimizationOutput::getCenterOfFaces() {
	return center_of_faces;
}

Eigen::MatrixXd OptimizationOutput::getFacesNormals() {
	return faces_normals;
}

Eigen::MatrixXd OptimizationOutput::getFacesNorm() {
	return center_of_faces + faces_normals;
}

std::vector<int> OptimizationOutput::GlobNeighSphereCenters(
	const int fi, 
	const float distance) 
{
	std::vector<int> Neighbors; Neighbors.clear();
	for (int i = 0; i < center_of_sphere.rows(); i++)
		if (((center_of_sphere.row(fi) - center_of_sphere.row(i)).norm() + abs(radius_of_sphere(fi) - radius_of_sphere(i))) < distance)
			Neighbors.push_back(i);
	return Neighbors;
}

std::vector<int> OptimizationOutput::FaceNeigh(
	const Eigen::Vector3d center, 
	const float distance) {
	std::vector<int> Neighbors; Neighbors.clear();
	for (int i = 0; i < center_of_faces.rows(); i++)
		if ((center.transpose() - center_of_faces.row(i)).norm() < distance)
			Neighbors.push_back(i);
	return Neighbors;
}

std::vector<int> OptimizationOutput::GlobNeighNorms(
	const int fi, 
	const float distance) 
{
	std::vector<int> Neighbors; Neighbors.clear();
	for (int i = 0; i < faces_normals.rows(); i++)
		if ((faces_normals.row(fi) - faces_normals.row(i)).squaredNorm() < distance)
			Neighbors.push_back(i);
	return Neighbors;
}

std::vector<int> OptimizationOutput::getNeigh(
	const app_utils::NeighborType type, 
	const Eigen::MatrixXi& F, 
	const int fi, 
	const float distance) 
{
	std::vector<int> neigh;
	if (type == app_utils::NeighborType::CURR_FACE)
		return neigh;
	if (type == app_utils::NeighborType::GLOBAL_NORMALS)
		return GlobNeighNorms(fi, distance);
	if (type == app_utils::NeighborType::GLOBAL_SPHERE)
		return GlobNeighSphereCenters(fi, distance);
	if (type == app_utils::NeighborType::LOCAL_NORMALS)
		neigh = GlobNeighNorms(fi, distance);
	else if (type == app_utils::NeighborType::LOCAL_SPHERE)
		neigh = GlobNeighSphereCenters(fi, distance);

	std::vector<int> result; result.push_back(fi);
	std::vector<std::vector<std::vector<int>>> TT;
	igl::triangle_triangle_adjacency(F, TT);
	int prevSize;
	do {
		prevSize = result.size();
		result = vectorsIntersection(adjSetOfTriangles(F, result, TT), neigh);
	} while (prevSize != result.size());
	return result;
}

std::vector<int> OptimizationOutput::adjSetOfTriangles(
	const Eigen::MatrixXi& F, 
	const std::vector<int> selected, 
	std::vector<std::vector<std::vector<int>>> TT) 
{
	std::vector<int> adj = selected;
	for (int selectedFace : selected) {
		for (std::vector<int> _ : TT[selectedFace]) {
			for (int fi : _) {
				if (std::find(adj.begin(), adj.end(), fi) == adj.end())
					adj.push_back(fi);
			}
		}
	}
	return adj;
}

std::vector<int> OptimizationOutput::vectorsIntersection(
	const std::vector<int>& A, 
	const std::vector<int>& B) 
{
	std::vector<int> intersection;
	for (int fi : A) {
		if (std::find(B.begin(), B.end(), fi) != B.end())
			intersection.push_back(fi);
	}
	return intersection;
}

Eigen::MatrixXd OptimizationOutput::getCenterOfSphere() 
{
	return center_of_sphere;
}

Eigen::MatrixXd OptimizationOutput::getCylinderDir(){
	return center_of_sphere + cylinder_dir;
}

Eigen::MatrixXd OptimizationOutput::getSphereEdges() 
{
	int numF = center_of_sphere.rows();
	Eigen::MatrixXd c(numF, 3);
	Eigen::MatrixXd empty;
	if (getCenterOfFaces().size() == 0 || getCenterOfSphere().size() == 0)
		return empty;
	for (int fi = 0; fi < numF; fi++) {
		Eigen::RowVectorXd v = (getCenterOfSphere().row(fi) - getCenterOfFaces().row(fi)).normalized();
		c.row(fi) = getCenterOfFaces().row(fi) + radius_of_sphere(fi) *v;
	}
	return c;
}

void OptimizationOutput::initFaceColors(
	const int numF,
	const Eigen::Vector3f center_sphere_color,
	const Eigen::Vector3f center_vertex_color,
	const Eigen::Vector3f centers_sphere_edge_color,
	const Eigen::Vector3f centers_norm_edge_color,
	const Eigen::Vector3f per_cylinder_dir_color,
	const Eigen::Vector3f per_cylinder_edge_color,
	const Eigen::Vector3f face_norm_color)
{
	color_per_face.resize(numF, 3);
	color_per_sphere_center.resize(numF, 3);
	color_per_cylinder_dir.resize(numF, 3);
	color_per_cylinder_edge.resize(numF, 3);
	color_per_vertex_center.resize(numF, 3);
	color_per_face_norm.resize(numF, 3);
	color_per_sphere_edge.resize(numF, 3);
	color_per_norm_edge.resize(numF, 3);
	for (int fi = 0; fi < numF; fi++) {
		color_per_sphere_center.row(fi) = center_sphere_color.cast<double>();
		color_per_cylinder_dir.row(fi) = per_cylinder_dir_color.cast<double>();
		color_per_cylinder_edge.row(fi) = per_cylinder_edge_color.cast<double>();
		color_per_vertex_center.row(fi) = center_vertex_color.cast<double>();
		color_per_face_norm.row(fi) = face_norm_color.cast<double>();
		color_per_sphere_edge.row(fi) = centers_sphere_edge_color.cast<double>();
		color_per_norm_edge.row(fi) = centers_norm_edge_color.cast<double>();
	}
}

void OptimizationOutput::updateFaceColors(
	const int fi, 
	const Eigen::Vector3f color) 
{
	color_per_face.row(fi) = color.cast<double>();
	color_per_sphere_center.row(fi) = color.cast<double>();
	color_per_cylinder_dir.row(fi) = color.cast<double>();
	color_per_cylinder_edge.row(fi) = color.cast<double>();
	color_per_vertex_center.row(fi) = color.cast<double>();
	color_per_face_norm.row(fi) = color.cast<double>();
	color_per_sphere_edge.row(fi) = color.cast<double>();
	color_per_norm_edge.row(fi) = color.cast<double>();
}

void OptimizationOutput::initMinimizers(
	const Eigen::MatrixXd& V, 
	const Eigen::MatrixXi& F, 
	const OptimizationUtils::InitSphereAuxiliaryVariables& typeAuxVar,
	const int distance)
{
	Eigen::VectorXd initVertices = Eigen::Map<const Eigen::VectorXd>(V.data(), V.size());
	Eigen::MatrixX3d normals;
	igl::per_face_normals((Eigen::MatrixX3d)V, (Eigen::MatrixX3i)F, normals);
	Eigen::VectorXd initNormals = Eigen::Map<const Eigen::VectorXd>(normals.data(), F.size());
	
	
	Eigen::MatrixXd center0, Cylinder_dir0;
	Eigen::VectorXd Radius0;
	if (typeAuxVar == OptimizationUtils::InitSphereAuxiliaryVariables::LEAST_SQUARE_SPHERE)
		OptimizationUtils::Least_Squares_Sphere_Fit(distance, V, F, center0, Radius0);
	else if (typeAuxVar == OptimizationUtils::InitSphereAuxiliaryVariables::MODEL_CENTER_POINT)
		OptimizationUtils::center_of_mesh(V, F, center0, Radius0);
	else if (typeAuxVar == OptimizationUtils::InitSphereAuxiliaryVariables::MINUS_NORMALS) {
		this->center_of_faces = OptimizationUtils::center_per_triangle(V, F);
		Radius0.resize(F.rows());
		center0.resize(F.rows(), 3);
		Radius0.setConstant(0.1);

		for (int i = 0; i < center0.rows(); i++) {
			center0.row(i) = this->center_of_faces.row(i) - Radius0(i) * normals.row(i);
		}
	}


	OptimizationUtils::Least_Squares_Cylinder_Fit(
		distance,
		V,
		F,
		center0,
		Cylinder_dir0,
		Radius0);
	

	setAuxVariables(V, F, center0, Radius0, Cylinder_dir0, normals);

	minimizer->init(
		totalObjective,
		initVertices,
		initNormals,
		Eigen::Map<Eigen::VectorXd>(center0.data(), F.size()),
		Radius0,
		Eigen::Map<Eigen::VectorXd>(Cylinder_dir0.data(), F.size()),
		F,
		V
	);
}

void OptimizationOutput::updateActiveMinimizer(
	const MinimizerType minimizer_type) 
{
	minimizer->step_type = minimizer_type;
}

