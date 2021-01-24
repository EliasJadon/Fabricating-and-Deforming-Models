#pragma once
#include "app_utils.h"

class OptimizationOutput {
private:
	Eigen::MatrixXd center_of_faces;
	Eigen::MatrixXd center_of_sphere;
	Eigen::MatrixXd cylinder_dir;
	Eigen::MatrixXd faces_normals;
	Eigen::VectorXd radius_of_sphere;
public:

	std::shared_ptr <FixChosenConstraints> Energy_FixChosenVertices;
	std::shared_ptr <FixChosenConstraints> Energy_FixChosenNormals;
	std::shared_ptr< FixChosenConstraints> Energy_FixChosenSpheres;
	std::shared_ptr< Grouping> Energy_GroupSpheres;
	std::shared_ptr< Grouping> Energy_GroupNormals;
	std::shared_ptr< Grouping> Energy_GroupCylinders;

	std::set<int> UserInterface_FixedFaces, UserInterface_FixedVertices;
	std::vector<FacesGroup> UserInterface_facesGroups;
	std::shared_ptr<Minimizer> minimizer;
	std::vector<std::vector<int>> clusters_indices;
	std::shared_ptr<TotalObjective> totalObjective;
	float prev_camera_zoom;
	Eigen::Vector3f prev_camera_translation;
	Eigen::Quaternionf prev_trackball_angle;
	Eigen::MatrixXd fixed_vertices_positions;
	Eigen::MatrixXd
		color_per_face,
		color_per_vertex,
		color_per_sphere_center,
		color_per_cylinder_dir,
		color_per_cylinder_edge,
		color_per_vertex_center,
		color_per_face_norm,
		color_per_sphere_edge,
		color_per_norm_edge;
	int ModelID, CoreID;
	bool UserInterface_IsTranslate;
	int UserInterface_TranslateIndex;
	ImVec2 screen_position, screen_size, results_window_position, outputs_window_position;
	bool showSphereEdges, showNormEdges,
		showTriangleCenters, showSphereCenters,
		showCylinderDir, showCylinderEdges,
		showFacesNorm;

	OptimizationOutput(
		igl::opengl::glfw::Viewer* viewer,
		const MinimizerType minimizer_type,
		const OptimizationUtils::LineSearch linesearchType);
	~OptimizationOutput() = default;
	void setAuxVariables(
		const Eigen::MatrixXd& V,
		const Eigen::MatrixXi& F,
		const Eigen::MatrixXd& center_of_sphere,
		const Eigen::VectorXd& radius_of_sphere,
		const Eigen::MatrixXd& cylinder_dir,
		const Eigen::MatrixXd& norm
	);
	double getRadiusOfSphere(int index);
	Eigen::VectorXd getRadiusOfSphere();
	Eigen::MatrixXd getCylinderDirOnly();
	double clustering_MSE(
		const app_utils::ClusteringType type,
		const int fi,
		const double w_center,
		const double w_radius,
		const double w_dir,
		const Eigen::RowVectorXd& clusters_val,
		const Eigen::RowVectorXd& clusters_center,
		const double& clusters_radius);
	void OptimizationOutput::clustering_Average(
		const app_utils::ClusteringType type,
		const std::vector<int> clusters_ind,
		Eigen::RowVectorXd& clusters_val,
		Eigen::RowVectorXd& clusters_center,
		double& clusters_radius);
	void clustering(
		const double center_ratio,
		const double radius_ratio,
		const double dir_ratio,
		const double MSE,
		const app_utils::ClusteringType type);
	void clusters_init(
		const double center_ratio,
		const double radius_ratio,
		const double dir_ratio,
		const double MSE,
		std::vector<Eigen::RowVectorXd>& clusters_val,
		std::vector<Eigen::RowVectorXd>& clusters_center,
		std::vector<double>& clusters_radius,
		const app_utils::ClusteringType type);
	void translateFaces(const int fi, const Eigen::Vector3d translateValue);
	Eigen::MatrixXd getCenterOfFaces();
	Eigen::MatrixXd getFacesNormals();
	Eigen::MatrixXd getFacesNorm();
	std::vector<int> GlobNeighSphereCenters(const int fi, const float distance);
	std::vector<int> FaceNeigh(const Eigen::Vector3d center, const float distance);
	std::vector<int> GlobNeighNorms(const int fi, const float distance);
	std::vector<int> getNeigh(const app_utils::NeighborType type, const Eigen::MatrixXi& F, const int fi, const float distance);
	std::vector<int> adjSetOfTriangles(const Eigen::MatrixXi& F, const std::vector<int> selected, std::vector<std::vector<std::vector<int>>> TT);
	std::vector<int> vectorsIntersection(const std::vector<int>& A, const std::vector<int>& B);
	Eigen::MatrixXd getCenterOfSphere();
	Eigen::MatrixXd getCylinderDir();
	Eigen::MatrixXd getSphereEdges();
	void initFaceColors(
		const int numF,
		const Eigen::Vector3f center_sphere_color,
		const Eigen::Vector3f center_vertex_color,
		const Eigen::Vector3f centers_sphere_edge_color,
		const Eigen::Vector3f centers_norm_edge_color,
		const Eigen::Vector3f per_cylinder_dir_color,
		const Eigen::Vector3f per_cylinder_edge_color,
		const Eigen::Vector3f face_norm_color);
	void updateFaceColors(const int fi, const Eigen::Vector3f color);
	void initMinimizers(
		const Eigen::MatrixXd& V,
		const Eigen::MatrixXi& F,
		const OptimizationUtils::InitAuxVariables& typeAuxVar,
		const int distance_from,
		const int distance_to,
		const int imax,
		const int jmax);
	void updateActiveMinimizer(const MinimizerType minimizer_type);
};
