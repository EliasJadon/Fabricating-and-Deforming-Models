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

Eigen::VectorXd OptimizationOutput::getRadiusOfSphere() 
{
	return this->radius_of_sphere;
}

void OptimizationOutput::clustering_Average(
	const app_utils::ClusteringType type,
	const std::vector<int> clusters_ind,
	Eigen::RowVectorXd& clusters_val,
	Eigen::RowVectorXd& clusters_center,
	double& clusters_radius)
{
	if (type == app_utils::ClusteringType::CLUSTERING_NORMAL) {
		Eigen::RowVectorXd avg;
		avg.resize(3);
		avg << 0, 0, 0;
		for (int currf : clusters_ind) {
			avg += faces_normals.row(currf);
		}
		avg /= clusters_ind.size();
		clusters_val = avg;
	}
	if (type == app_utils::ClusteringType::CLUSTERING_SPHERE) {
		Eigen::RowVectorXd avgC;
		double avgR = 0;
		avgC.resize(3);
		avgC << 0, 0, 0;
		for (int currf : clusters_ind) {
			avgC += center_of_sphere.row(currf);
			avgR += radius_of_sphere(currf);
		}
		avgC /= clusters_ind.size();
		avgR /= clusters_ind.size();
		clusters_center = avgC;
		clusters_radius = avgR;
	}
	if (type == app_utils::ClusteringType::CLUSTERING_CYLINDER) {
		Eigen::RowVectorXd avgC, avgA;
		double avgR = 0;
		avgC.resize(3);
		avgA.resize(3);
		avgC << 0, 0, 0;
		avgA << 0, 0, 0;
		for (int currf : clusters_ind) {
			avgC += center_of_sphere.row(currf);
			avgA += cylinder_dir.row(currf);
			avgR += radius_of_sphere(currf);
		}
		avgC /= clusters_ind.size();
		avgA /= clusters_ind.size();
		avgR /= clusters_ind.size();
		clusters_center = avgC;
		clusters_radius = avgR;
		clusters_val = avgA;
	}
}

double OptimizationOutput::clustering_MSE(
	const app_utils::ClusteringType type,
	const int fi,
	const double center_ratio,
	const double radius_ratio,
	const double dir_ratio,
	const Eigen::RowVectorXd& clusters_val,
	const Eigen::RowVectorXd& clusters_center,
	const double& clusters_radius)
{
	if (type == app_utils::ClusteringType::CLUSTERING_NORMAL) 
	{
		Eigen::RowVectorXd N1 = faces_normals.row(fi);
		Eigen::RowVectorXd N0 = clusters_val;
		return (N1 - N0).squaredNorm();
	}
	else if (type == app_utils::ClusteringType::CLUSTERING_SPHERE) 
	{
		Eigen::RowVectorXd C1 = center_of_sphere.row(fi);
		Eigen::RowVectorXd C0 = clusters_center;
		double R1 = radius_of_sphere(fi);
		double R0 = clusters_radius;
		return (center_ratio * ((C1 - C0).squaredNorm()) + radius_ratio * pow(R1 - R0, 2));
	}
	else if (type == app_utils::ClusteringType::CLUSTERING_CYLINDER) 
	{
		Eigen::RowVectorXd A1 = cylinder_dir.row(fi);
		Eigen::RowVectorXd A0 = clusters_val;
		Eigen::RowVectorXd C1 = center_of_sphere.row(fi);
		Eigen::RowVectorXd C0 = clusters_center;
		double R1 = radius_of_sphere(fi);
		double R0 = clusters_radius;

		return (
			center_ratio * pow(pow((C1 - C0).normalized() * A0.normalized().transpose(), 2) - 1, 2)
			+
			center_ratio * pow(pow((C0 - C1).normalized() * A1.normalized().transpose(), 2) - 1, 2)
			+
			dir_ratio * ((A1 - A0).squaredNorm())
			+
			radius_ratio * pow(R1 - R0, 2)
			);
	}
	return -1; //Error
}

void OptimizationOutput::clustering(
	const double center_ratio, 
	const double radius_ratio, 
	const double dir_ratio,
	const double MSE, 
	const app_utils::ClusteringType type)
{
	if (type != app_utils::ClusteringType::CLUSTERING_NORMAL &&
		type != app_utils::ClusteringType::CLUSTERING_SPHERE &&
		type != app_utils::ClusteringType::CLUSTERING_CYLINDER)
	{
		clusters_indices.clear();
		return;
	}

	std::vector<std::vector<int>> clusters_ind;
	std::vector<Eigen::RowVectorXd> clusters_val;
	std::vector<Eigen::RowVectorXd> clusters_center;
	std::vector<double> clusters_radius;
	clusters_init(center_ratio,radius_ratio,dir_ratio, 
		MSE, clusters_val, clusters_center, clusters_radius, type);

	const int numFaces = faces_normals.rows();
	//Do 5 rounds of K-means clustering alg.
	for (int _ = 0; _ < 5; _++) {
		clusters_ind.clear();
		int numClusters;
		if (type != app_utils::ClusteringType::CLUSTERING_SPHERE)
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
				double currMSE = clustering_MSE(type, fi,
					center_ratio, radius_ratio, dir_ratio,
					clusters_val[ci], clusters_center[ci], clusters_radius[ci]);

				if (currMSE < minMSE)
				{
					minMSE = currMSE;
					argmin = ci;
					found = true;
				}
			}
			if (found)
				clusters_ind[argmin].push_back(fi);
			else
			{
				clusters_ind.push_back({ fi });
				if (type == app_utils::ClusteringType::CLUSTERING_NORMAL)
					clusters_val.push_back(faces_normals.row(fi));
				if (type == app_utils::ClusteringType::CLUSTERING_SPHERE) {
					clusters_center.push_back(center_of_sphere.row(fi));
					clusters_radius.push_back(radius_of_sphere(fi));
				}
				if (type == app_utils::ClusteringType::CLUSTERING_CYLINDER) {
					clusters_val.push_back(cylinder_dir.row(fi));
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
				if (type == app_utils::ClusteringType::CLUSTERING_NORMAL)
					it_N = clusters_val.erase(it_N);
				if (type == app_utils::ClusteringType::CLUSTERING_SPHERE){
					it_C = clusters_center.erase(it_C);
					it_R = clusters_radius.erase(it_R);
				}
				if (type == app_utils::ClusteringType::CLUSTERING_CYLINDER) {
					it_N = clusters_val.erase(it_N);
					it_C = clusters_center.erase(it_C);
					it_R = clusters_radius.erase(it_R);
				}

			}
			else {
				it_i++;
				if (type == app_utils::ClusteringType::CLUSTERING_NORMAL)
					it_N++;
				if (type == app_utils::ClusteringType::CLUSTERING_SPHERE) {
					it_R++;
					it_C++;
				}
				if (type == app_utils::ClusteringType::CLUSTERING_CYLINDER) {
					it_N++; 
					it_R++;
					it_C++;
				}
			}
		}
		//Update average
		for (int ci = 0; ci < clusters_ind.size(); ci++)
			clustering_Average(
				type,
				clusters_ind[ci],
				clusters_val[ci],
				clusters_center[ci],
				clusters_radius[ci]);
		//Union similar clusters
		std::vector<Eigen::RowVectorXd>::iterator val1, val2, cent1, cent2;
		std::vector<double>::iterator radius1, radius2;
		if (type == app_utils::ClusteringType::CLUSTERING_NORMAL) {
			val1 = clusters_val.begin();
			val2 = val1 + 1;
		}
		if (type == app_utils::ClusteringType::CLUSTERING_SPHERE) {
			cent1 = clusters_center.begin();
			cent2 = cent1 + 1;
			radius1 = clusters_radius.begin();
			radius2 = radius1 + 1;
		}
		if (type == app_utils::ClusteringType::CLUSTERING_CYLINDER) {
			cent1 = clusters_center.begin();
			cent2 = cent1 + 1;
			radius1 = clusters_radius.begin();
			radius2 = radius1 + 1;
			val1 = clusters_val.begin();
			val2 = val1 + 1;
		}

		auto& ind1 = clusters_ind.begin();
		auto& ind2 = ind1 + 1;
		while (ind1 != clusters_ind.end())
		{
			if (type == app_utils::ClusteringType::CLUSTERING_NORMAL) {
				val2 = val1 + 1;
			}
			if (type == app_utils::ClusteringType::CLUSTERING_SPHERE) {
				cent2 = cent1 + 1;
				radius2 = radius1 + 1;
			}
			if (type == app_utils::ClusteringType::CLUSTERING_CYLINDER) {
				val2 = val1 + 1; 
				cent2 = cent1 + 1;
				radius2 = radius1 + 1;
			}
			for (ind2 = ind1 + 1; ind2 != clusters_ind.end();)
			{
				double diff;
				if (type == app_utils::ClusteringType::CLUSTERING_NORMAL)
				{
					Eigen::RowVectorXd N1 = *val1;
					Eigen::RowVectorXd N0 = *val2;
					diff = (N1 - N0).squaredNorm();
				}
				else if (type == app_utils::ClusteringType::CLUSTERING_SPHERE)
				{
					Eigen::RowVectorXd C1 = *cent1;
					Eigen::RowVectorXd C0 = *cent2;
					double R1 = *radius1;
					double R0 = *radius2;
					diff = (center_ratio * ((C1 - C0).squaredNorm()) + radius_ratio * pow(R1 - R0, 2));
				}
				else if (type == app_utils::ClusteringType::CLUSTERING_CYLINDER)
				{
					Eigen::RowVectorXd A1 = *val1;
					Eigen::RowVectorXd A0 = *val2;
					Eigen::RowVectorXd C1 = *cent1;
					Eigen::RowVectorXd C0 = *cent2;
					double R1 = *radius1;
					double R0 = *radius2;
					diff = (
						center_ratio * pow(pow((C1 - C0).normalized() * A0.normalized().transpose(), 2) - 1, 2)
						+
						center_ratio * pow(pow((C0 - C1).normalized() * A1.normalized().transpose(), 2) - 1, 2)
						+
						dir_ratio * ((A1 - A0).squaredNorm())
						+
						radius_ratio * pow(R1 - R0, 2)
						);
				}





				if (diff < MSE) {
					for (int currf : (*ind2)) {
						ind1->push_back(currf);
					}
					if (type == app_utils::ClusteringType::CLUSTERING_NORMAL) {
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
					if (type == app_utils::ClusteringType::CLUSTERING_SPHERE) {
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
					if (type == app_utils::ClusteringType::CLUSTERING_CYLINDER) {
						Eigen::RowVectorXd avgC, avgA;
						double avgR = 0;
						avgC.resize(3);
						avgA.resize(3);
						avgC << 0, 0, 0;
						avgA << 0, 0, 0;
						for (int currf : (*ind1)) {
							avgC += center_of_sphere.row(currf);
							avgA += cylinder_dir.row(currf);
							avgR += radius_of_sphere(currf);
						}
						avgC /= ind1->size();
						avgA /= ind1->size();
						avgR /= ind1->size();
						*cent1 = avgC;
						*val1 = avgA;
						*radius1 = avgR;
						cent2 = clusters_center.erase(cent2);
						radius2 = clusters_radius.erase(radius2);
						val2 = clusters_val.erase(val2);
					}

					ind2 = clusters_ind.erase(ind2);
				}
				else {
					if (type == app_utils::ClusteringType::CLUSTERING_NORMAL) {
						val2++;
					}
					if (type == app_utils::ClusteringType::CLUSTERING_SPHERE) {
						cent2++;
						radius2++;
					}
					if (type == app_utils::ClusteringType::CLUSTERING_CYLINDER) {
						val2++; 
						cent2++;
						radius2++;
					}
					ind2++;
				}
			}

			if (type == app_utils::ClusteringType::CLUSTERING_NORMAL)
				val1++;
			if (type == app_utils::ClusteringType::CLUSTERING_SPHERE) {
				cent1++; radius1++;
			}
			if (type == app_utils::ClusteringType::CLUSTERING_CYLINDER) {
				val1++; cent1++; radius1++;
			}
			ind1++;
		}
	}
	clusters_indices = clusters_ind;
}

void OptimizationOutput::clusters_init(
	const double center_ratio,
	const double radius_ratio,
	const double dir_ratio,
	const double MSE,
	std::vector<Eigen::RowVectorXd>& clusters_val,
	std::vector<Eigen::RowVectorXd>& clusters_center,
	std::vector<double>& clusters_radius,
	const app_utils::ClusteringType type)
{
	std::vector<std::vector<int>> clusters_ind;
	int numFaces;
	if (type == app_utils::ClusteringType::CLUSTERING_NORMAL) {
		clusters_val.clear();
		clusters_val.push_back(faces_normals.row(0));
		numFaces = faces_normals.rows();
	}
	if (type == app_utils::ClusteringType::CLUSTERING_SPHERE){
		clusters_center.clear();
		clusters_center.push_back(center_of_sphere.row(0));
		clusters_radius.clear();
		clusters_radius.push_back(radius_of_sphere(0));
		numFaces = center_of_sphere.rows();
	}
	if (type == app_utils::ClusteringType::CLUSTERING_CYLINDER){
		clusters_center.clear();
		clusters_center.push_back(center_of_sphere.row(0));
		clusters_radius.clear();
		clusters_radius.push_back(radius_of_sphere(0));
		clusters_val.clear();
		clusters_val.push_back(cylinder_dir.row(0));
		numFaces = center_of_sphere.rows();
	}
	clusters_ind.push_back({ 0 });

	for (int fi = 1; fi < numFaces; fi++)
	{
		bool found = false;
		double minMSE = MSE;
		int argmin;
		int numClusters;
		if (type == app_utils::ClusteringType::CLUSTERING_NORMAL)
			numClusters = clusters_val.size();
		if (type == app_utils::ClusteringType::CLUSTERING_CYLINDER)
			numClusters = clusters_val.size();
		if (type == app_utils::ClusteringType::CLUSTERING_SPHERE)
			numClusters = clusters_center.size();
		for (int ci = 0; ci < numClusters; ci++)
		{
			double currMSE = clustering_MSE(type, fi,
				center_ratio, radius_ratio, dir_ratio,
				clusters_val[ci], clusters_center[ci], clusters_radius[ci]);
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
			clustering_Average(
				type,
				clusters_ind[argmin],
				clusters_val[argmin],
				clusters_center[argmin],
				clusters_radius[argmin]);
		}
		else
		{
			clusters_ind.push_back({ fi });
			if (type == app_utils::ClusteringType::CLUSTERING_NORMAL)
				clusters_val.push_back(faces_normals.row(fi));
			if (type == app_utils::ClusteringType::CLUSTERING_SPHERE) {
				clusters_center.push_back(center_of_sphere.row(fi));
				clusters_radius.push_back(radius_of_sphere(fi));
			}
			if (type == app_utils::ClusteringType::CLUSTERING_CYLINDER) {
				clusters_center.push_back(center_of_sphere.row(fi));
				clusters_radius.push_back(radius_of_sphere(fi));
				clusters_val.push_back(cylinder_dir.row(fi));
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

std::vector<int> OptimizationOutput::GlobNeighCylinders(
	const int fi, 
	const float distance) 
{
	std::vector<int> Neighbors; Neighbors.clear();
	const Eigen::RowVectorXd A1 = cylinder_dir.row(fi);
	const Eigen::RowVectorXd C1 = center_of_sphere.row(fi);
	const double R1 = radius_of_sphere(fi);

	for (int i = 0; i < cylinder_dir.rows(); i++) {
		const Eigen::RowVectorXd A0 = cylinder_dir.row(i);
		const Eigen::RowVectorXd C0 = center_of_sphere.row(i);
		const double R0 = radius_of_sphere(i);

		if ((pow(pow((C1 - C0).normalized() * A0.normalized().transpose(), 2) - 1, 2)
			+
			pow(pow((C0 - C1).normalized() * A1.normalized().transpose(), 2) - 1, 2)
			+
			((A1 - A0).squaredNorm())
			+
			pow(R1 - R0, 2)) < distance)
			Neighbors.push_back(i);
	}
	return Neighbors;
}

std::vector<int> OptimizationOutput::FaceNeigh(const Eigen::Vector3d center, const float distance) 
{
	std::vector<int> Neighbors; Neighbors.clear();
	for (int i = 0; i < center_of_faces.rows(); i++) {
		double x = center(0) - center_of_faces(i, 0);
		double y = center(1) - center_of_faces(i, 1);
		double z = center(2) - center_of_faces(i, 2);
		if ((pow(x, 2) + pow(y, 2) + pow(z, 2)) < pow(distance,2))
			Neighbors.push_back(i);
	}
	return Neighbors;
}

std::vector<int> OptimizationOutput::GlobNeighNorms(const int fi, const float distance) 
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
		return { fi };
	if (type == app_utils::NeighborType::GLOBAL_NORMALS)
		return GlobNeighNorms(fi, distance);
	if (type == app_utils::NeighborType::GLOBAL_SPHERE)
		return GlobNeighSphereCenters(fi, distance);
	if (type == app_utils::NeighborType::GLOBAL_CYLINDERS)
		return GlobNeighCylinders(fi, distance);
	if (type == app_utils::NeighborType::LOCAL_NORMALS)
		neigh = GlobNeighNorms(fi, distance);
	else if (type == app_utils::NeighborType::LOCAL_SPHERE)
		neigh = GlobNeighSphereCenters(fi, distance);
	else if (type == app_utils::NeighborType::LOCAL_CYLINDERS)
		neigh = GlobNeighCylinders(fi, distance);
	
	//pick only adjanced faces in order to get local faces
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

Eigen::MatrixXd OptimizationOutput::getCylinderDirOnly(){
	return cylinder_dir;
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

void OptimizationOutput::setFaceColors(
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

void OptimizationOutput::shiftFaceColors(
	const int fi, 
	const double alpha,
	const Eigen::Vector3f model_color,
	const Eigen::Vector3f color) 
{
	double w = std::min<double>(std::max<double>(alpha, 0), 1);
	auto averaged = color.cast<double>() * w + model_color.cast<double>() * (1 - w);
	color_per_face.row(fi) = averaged;
	color_per_sphere_center.row(fi) = averaged;
	color_per_cylinder_dir.row(fi) = averaged;
	color_per_cylinder_edge.row(fi) = averaged;
	color_per_vertex_center.row(fi) = averaged;
	color_per_face_norm.row(fi) = averaged;
	color_per_sphere_edge.row(fi) = averaged;
	color_per_norm_edge.row(fi) = averaged;
}

void OptimizationOutput::initMinimizers(
	const Eigen::MatrixXd& V, 
	const Eigen::MatrixXi& F, 
	const OptimizationUtils::InitAuxVariables& typeAuxVar,
	const int distance_from,
	const int distance_to,
	const int imax,
	const int jmax,
	const std::vector<int> copy_index,
	const std::vector < std::set<int>> paste_index,
	const std::vector < std::set<int>> group_index)
{
	Eigen::MatrixX3d normals;
	igl::per_face_normals((Eigen::MatrixX3d)V, (Eigen::MatrixX3i)F, normals);
	
	Eigen::MatrixXd center0, Cylinder_dir0(F.rows(),3);
	Cylinder_dir0.setConstant(1);
	Eigen::VectorXd Radius0;

	if (typeAuxVar == OptimizationUtils::InitAuxVariables::SPHERE_FIT)
		OptimizationUtils::Least_Squares_Sphere_Fit(distance_from, distance_to, V, F, center0, Radius0, group_index);
	else if (typeAuxVar == OptimizationUtils::InitAuxVariables::MODEL_CENTER_POINT)
		OptimizationUtils::center_of_mesh(V, F, center0, Radius0);
	else if (typeAuxVar == OptimizationUtils::InitAuxVariables::MINUS_NORMALS) {
		this->center_of_faces = OptimizationUtils::center_per_triangle(V, F);
		Radius0.resize(F.rows());
		center0.resize(F.rows(), 3);
		Radius0.setConstant(0.1);

		for (int i = 0; i < center0.rows(); i++) {
			center0.row(i) = this->center_of_faces.row(i) - Radius0(i) * normals.row(i);
		}
	}
	else if (typeAuxVar == OptimizationUtils::InitAuxVariables::CYLINDER_FIT) {
		OptimizationUtils::Least_Squares_Cylinder_Fit(imax, jmax, distance_from, distance_to, V, F, center0, Cylinder_dir0, Radius0, group_index);

		std::vector<std::set<int>> TT = OptimizationUtils::Triangle_triangle_adjacency(F);
		for (int i = 0; i < 5; i++) {
			std::cout << "---------------Round" << i << "!!!" << std::endl;
			for (int f1 = 0; f1 < TT.size(); f1++) {
				for (int f2 : TT[f1]) {
					if (f1 != f2)
					{
						Eigen::RowVector3d C1 = center0.row(f1);
						Eigen::RowVector3d C2 = center0.row(f2);
						Eigen::RowVector3d A1 = Cylinder_dir0.row(f1);
						Eigen::RowVector3d A2 = Cylinder_dir0.row(f2);

						if ((C1 - C2).norm() < 0.001) {
							std::cout << "---------------Error3!!!" << std::endl;
							std::cout << "f1 = " << f1 << std::endl;
							std::cout << "f2 = " << f2 << std::endl;
							std::cout << "A1 = " << A1 << std::endl;
							std::cout << "A2 = " << A2 << std::endl;
							std::cout << "C1 = " << C1 << std::endl;
							std::cout << "C2 = " << C2 << std::endl;
							double drand = ((double)rand() / RAND_MAX) * 10;
							center0(f1, 0) = Cylinder_dir0(f1, 0) * drand + center0(f1, 0);
							center0(f1, 1) = Cylinder_dir0(f1, 1) * drand + center0(f1, 1);
							center0(f1, 2) = Cylinder_dir0(f1, 2) * drand + center0(f1, 2);
						}
					}
				}
			}
		}
		
	}

	for (int i = 0; i < copy_index.size(); i++) {
		for (int f : paste_index[i]) {
			Radius0(f) = Radius0(copy_index[i]);
			center0.row(f) = center0.row(copy_index[i]);
			Cylinder_dir0.row(f) = Cylinder_dir0.row(copy_index[i]);
			normals.row(f) = normals.row(copy_index[i]);
		}
	}
	
	setAuxVariables(V, F, center0, Radius0, Cylinder_dir0, normals);

	minimizer->init(
		totalObjective,
		Eigen::Map<const Eigen::VectorXd>(V.data(), V.size()),
		Eigen::Map<const Eigen::VectorXd>(normals.data(), F.size()),
		Eigen::Map<const Eigen::VectorXd>(center0.data(), F.size()),
		Radius0,
		Eigen::Map<const Eigen::VectorXd>(Cylinder_dir0.data(), F.size()),
		F,
		V
	);
}

void OptimizationOutput::updateActiveMinimizer(
	const MinimizerType minimizer_type) 
{
	minimizer->step_type = minimizer_type;
}

