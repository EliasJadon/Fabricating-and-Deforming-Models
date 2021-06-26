#pragma once
#include "app_utils.h"

class OptimizationOutput {
public:
	Eigen::MatrixXd center_of_faces, center_of_sphere, normals;
	Eigen::VectorXd radiuses;
	std::vector<std::vector<int>> clustering_faces_indices;
	Eigen::MatrixXd clustering_faces_colors;
	std::shared_ptr <AuxSpherePerHinge> Energy_auxSpherePerHinge;
	std::shared_ptr <AuxBendingNormal> Energy_auxBendingNormal;
	std::shared_ptr <FixChosenConstraints> Energy_FixChosenVertices;
	std::shared_ptr<Minimizer> minimizer;
	std::shared_ptr<TotalObjective> totalObjective;
	float prev_camera_zoom;
	Eigen::Vector3f prev_camera_translation;
	Eigen::Quaternionf prev_trackball_angle;
	Eigen::MatrixXd color_per_face, color_per_sphere_center, color_per_vertex_center;
	Eigen::MatrixXd color_per_face_norm, color_per_sphere_edge, color_per_norm_edge;
	int ModelID, CoreID;
	ImVec2 screen_position, screen_size, results_window_position, outputs_window_position;
	bool showSphereEdges, showNormEdges, showTriangleCenters, showSphereCenters, showFacesNorm;



	OptimizationOutput(
		igl::opengl::glfw::Viewer* viewer,
		const Cuda::OptimizerType Optimizer_type,
		const OptimizationUtils::LineSearch linesearchType);
	~OptimizationOutput() = default;
	void setAuxVariables(
		const Eigen::MatrixXd& V,
		const Eigen::MatrixXi& F,
		const Eigen::MatrixXd& center_of_sphere,
		const Eigen::VectorXd& radius_of_sphere,
		const Eigen::MatrixXd& norm
	);
	double getRadiusOfSphere(int index);
	Eigen::VectorXd getRadiusOfSphere();
	Eigen::MatrixXd getCenterOfFaces();
	Eigen::MatrixXd getFacesNormals();
	Eigen::MatrixXd getFacesNorm();
	std::vector<int> GlobNeighSphereCenters(const int fi, const float distance);
	std::vector<int> FaceNeigh(const Eigen::Vector3d center, const float distance);
	std::vector<int> GlobNeighNorms(const int fi, const float distance);
	std::vector<int> getNeigh(const app_utils::Neighbor_Type type, const Eigen::MatrixXi& F, const int fi, const float distance);
	std::vector<int> adjSetOfTriangles(const Eigen::MatrixXi& F, const std::vector<int> selected, std::vector<std::vector<std::vector<int>>> TT);
	std::vector<int> vectorsIntersection(const std::vector<int>& A, const std::vector<int>& B);
	Eigen::MatrixXd getCenterOfSphere();
	Eigen::MatrixXd getSphereEdges();
	void initFaceColors(
		const int numF,
		const Eigen::Vector3f center_sphere_color,
		const Eigen::Vector3f center_vertex_color,
		const Eigen::Vector3f centers_sphere_edge_color,
		const Eigen::Vector3f centers_norm_edge_color,
		const Eigen::Vector3f face_norm_color);
	void setFaceColors(const int fi, const Eigen::Vector3f color);
	void shiftFaceColors(const int fi, const double alpha, const Eigen::Vector3f model_color, const Eigen::Vector3f color);
	void initMinimizers(
		const Eigen::MatrixXd& V,
		const Eigen::MatrixXi& F,
		const OptimizationUtils::InitSphereAuxVariables& typeAuxVar,
		const int distance_from,
		const int distance_to,
		const std::vector<int> copy_index,
		const std::vector < std::set<int>> paste_index,
		const double minus_normals_radius_length);
	void updateActiveMinimizer(const Cuda::OptimizerType optimizer_type);
};
