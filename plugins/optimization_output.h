#pragma once
#include "app_utils.h"

class OptimizationOutput {
private:
	Eigen::MatrixXd center_of_faces;
	Eigen::MatrixXd center_of_sphere;
	Eigen::MatrixXd faces_normals;
	Eigen::VectorXd radius_of_sphere;
public:
	std::shared_ptr <FixChosenVertices> Energy_FixChosenVertices;
	std::shared_ptr <FixChosenVertices> Energy_FixChosenNormals;
	std::shared_ptr< FixChosenVertices> Energy_FixChosenSpheres;
	std::shared_ptr< GroupSpheres> Energy_GroupSpheres;
	std::shared_ptr< GroupNormals> Energy_GroupNormals;

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
		color_per_vertex_center,
		color_per_face_norm,
		color_per_sphere_edge,
		color_per_norm_edge;
	int ModelID, CoreID;
	bool UserInterface_IsTranslate;
	int UserInterface_TranslateIndex;
	ImVec2 screen_position, screen_size, results_window_position, outputs_window_position;
	bool showSphereEdges, showNormEdges, showTriangleCenters, showSphereCenters, showFacesNorm;

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
		const Eigen::MatrixXd& norm
	);
	double getRadiusOfSphere(int index);
	void clustering(const double ratio, const double MSE, const bool isNormal);
	void clusters_init(
		const double ratio,
		const double MSE,
		std::vector<Eigen::RowVectorXd>& clusters_val,
		std::vector<Eigen::RowVectorXd>& clusters_center,
		std::vector<double>& clusters_radius,
		const bool isNormal);
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
	Eigen::MatrixXd getSphereEdges();
	void initFaceColors(
		const int numF,
		const Eigen::Vector3f center_sphere_color,
		const Eigen::Vector3f center_vertex_color,
		const Eigen::Vector3f centers_sphere_edge_color,
		const Eigen::Vector3f centers_norm_edge_color,
		const Eigen::Vector3f face_norm_color);
	void updateFaceColors(const int fi, const Eigen::Vector3f color);
	void initMinimizers(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const OptimizationUtils::InitSphereAuxiliaryVariables& typeAuxVar);
	void updateActiveMinimizer(const MinimizerType minimizer_type);
};
