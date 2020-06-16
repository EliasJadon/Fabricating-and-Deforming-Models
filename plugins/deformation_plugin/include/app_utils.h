#pragma once

#include <iostream>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/unproject_in_mesh.h>
#include <igl/lscm.h>
#include <igl/harmonic.h>
#include <igl/arap.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/project.h>
#include <igl/unproject.h>
#include <igl/slice.h>
#include <igl/edge_lengths.h>
#include <imgui/imgui.h>

#include "../../libs/optimization_lib/include/minimizers/Minimizer.h"
#include "../../libs/optimization_lib/include/minimizers/NewtonMinimizer.h"
#include "../../libs/optimization_lib/include/minimizers/GradientDescentMinimizer.h"
#include "../../libs/optimization_lib/include/minimizers/AdamMinimizer.h"

#include "../../libs/optimization_lib/include/objective_functions/STVK.h"
#include "../../libs/optimization_lib/include/objective_functions/SymmetricDirichlet.h"
#include "../../libs/optimization_lib/include/objective_functions/AllVertexPositions.h"
#include "../../libs/optimization_lib/include/objective_functions/BendingEdge.h"
#include "../../libs/optimization_lib/include/objective_functions/AuxBendingNormal.h"
#include "../../libs/optimization_lib/include/objective_functions/AuxSpherePerHinge.h"
#include "../../libs/optimization_lib/include/objective_functions/BendingNormal.h"
#include "../../libs/optimization_lib/include/objective_functions/MembraneConstraints.h"
#include "../../libs/optimization_lib/include/objective_functions/PenaltyPositionalConstraints.h"

#define RED_COLOR Eigen::Vector3f(1, 0, 0)
#define BLUE_COLOR Eigen::Vector3f(0, 0, 1)
#define GREEN_COLOR Eigen::Vector3f(0, 1, 0)
#define GOLD_COLOR Eigen::Vector3f(1, 215.0f / 255.0f, 0)
#define GREY_COLOR Eigen::Vector3f(0.75, 0.75, 0.75)
#define WHITE_COLOR Eigen::Vector3f(1, 1, 1)
#define BLACK_COLOR Eigen::Vector3f(0, 0, 0)
#define M_PI 3.14159

namespace app_utils
{
	enum View {
		HORIZONTAL = 0,
		VERTICAL = 1,
		INPUT_ONLY = 2,
		OUTPUT_ONLY_0 = 3
	};
	enum MouseMode { 
		NONE = 0, 
		FACE_SELECT, 
		VERTEX_SELECT, 
		CLEAR 
	};
	enum MinimizerType {
		NEWTON = 0,
		GRADIENT_DESCENT = 1,
		ADAM_MINIMIZER = 2
	};

	static Eigen::Vector3f computeTranslation(
		const int mouse_x, 
		const int from_x, 
		const int mouse_y, 
		const int from_y, 
		const Eigen::RowVector3d pt3D,
		igl::opengl::ViewerCore& core) {
		Eigen::Matrix4f modelview = core.view;
		//project the given point (typically the handle centroid) to get a screen space depth
		Eigen::Vector3f proj = igl::project(pt3D.transpose().cast<float>().eval(),
			modelview,
			core.proj,
			core.viewport);
		float depth = proj[2];
		double x, y;
		Eigen::Vector3f pos1, pos0;
		//unproject from- and to- points
		x = mouse_x;
		y = core.viewport(3) - mouse_y;
		pos1 = igl::unproject(Eigen::Vector3f(x, y, depth),
			modelview,
			core.proj,
			core.viewport);
		x = from_x;
		y = core.viewport(3) - from_y;
		pos0 = igl::unproject(Eigen::Vector3f(x, y, depth),
			modelview,
			core.proj,
			core.viewport);
		//translation is the vector connecting the two
		Eigen::Vector3f translation;
		translation = pos1 - pos0;
		return translation;
	}
	
	static std::string ExtractModelName(const std::string& str)
	{
		size_t head, tail;
		head = str.find_last_of("/\\");
		tail = str.find_last_of("/.");
		return (str.substr((head + 1), (tail - head - 1)));
	}

	static bool IsMesh2D(const Eigen::MatrixXd& V) {
		return (V.col(2).array() == 0).all();
	}

	static char* build_view_names_list(const int size) {
		std::string cStr("");
		cStr += "Horizontal";
		cStr += '\0';
		cStr += "Vertical";
		cStr += '\0';
		cStr += "InputOnly";
		cStr += '\0';
		for (int i = 0; i < size; i++) {
			std::string sts;
			sts = "OutputOnly " + std::to_string(i);
			cStr += sts.c_str();
			cStr += '\0';
		}
		cStr += '\0';
		int listLength = cStr.length();
		char* comboList = new char[listLength];
		for (unsigned int i = 0; i < listLength; i++)
			comboList[i] = cStr.at(i);
		return comboList;
	}

	static char* build_color_energies_list(const std::shared_ptr<TotalObjective>& totalObjective) {
		std::string cStr("");
		cStr += "No colors";
		cStr += '\0';
		cStr += "Total energy";
		cStr += '\0';
		for (auto& obj : totalObjective->objectiveList) {
			cStr += (obj->name).c_str();
			cStr += '\0';
		}
		cStr += '\0';
		int listLength = cStr.length();
		char* comboList = new char[listLength];
		for (unsigned int i = 0; i < listLength; i++)
			comboList[i] = cStr.at(i);
		return comboList;
	}

	static Eigen::RowVector3d get_face_avg(const igl::opengl::glfw::Viewer *viewer, const int Model_Translate_ID,const int Translate_Index){
		Eigen::RowVector3d avg; avg << 0, 0, 0;
		Eigen::RowVector3i face = viewer->data(Model_Translate_ID).F.row(Translate_Index);

		avg += viewer->data(Model_Translate_ID).V.row(face[0]);
		avg += viewer->data(Model_Translate_ID).V.row(face[1]);
		avg += viewer->data(Model_Translate_ID).V.row(face[2]);
		avg /= 3;

		return avg;
	}
}

class OptimizationOutput
{
private:
	std::shared_ptr<NewtonMinimizer> newtonMinimizer;
	std::shared_ptr<GradientDescentMinimizer> gradientDescentMinimizer;
	std::shared_ptr<AdamMinimizer> adamMinimizer;
	Eigen::MatrixXd center_of_triangle;
	Eigen::MatrixXd center_of_sphere;

public:
	std::shared_ptr<TotalObjective> totalObjective;
	std::shared_ptr<Minimizer> activeMinimizer;
	float prev_camera_zoom;
	Eigen::Vector3f prev_camera_translation;
	Eigen::Quaternionf prev_trackball_angle;
	std::vector<int> *HandlesInd; //pointer to indices in constraitPositional
	Eigen::MatrixX3d *HandlesPosDeformed; //pointer to positions in constraitPositional
	Eigen::MatrixXd color_per_face, Vertices_output;
	int ModelID, CoreID;
	ImVec2 window_position, window_size, text_position;
	
	//Constructor & initialization
	OptimizationOutput(
		igl::opengl::glfw::Viewer* viewer, 
		const app_utils::MinimizerType minimizer_type,
		const OptimizationUtils::LineSearch linesearchType) 
	{
		//update viewer
		CoreID = viewer->append_core(Eigen::Vector4f::Zero());
		viewer->core(CoreID).background_color = Eigen::Vector4f(0.9, 0.9, 0.9, 0);
		viewer->core(CoreID).is_animating = true;
		viewer->core(CoreID).lighting_factor = 0.5;
		// Initialize minimizer thread
		newtonMinimizer = std::make_shared<NewtonMinimizer>(CoreID);
		gradientDescentMinimizer = std::make_shared<GradientDescentMinimizer>(CoreID);
		adamMinimizer = std::make_shared<AdamMinimizer>(CoreID);
		updateActiveMinimizer(minimizer_type);
		activeMinimizer->lineSearch_type = linesearchType;
		totalObjective = std::make_shared<TotalObjective>();
	}

	~OptimizationOutput() {}

	void setCenters(
		const Eigen::MatrixXd& V, 
		const Eigen::MatrixXi& F, 
		const Eigen::MatrixXd& center_of_sphere
	) {
		this->center_of_triangle = OptimizationUtils::center_per_triangle(V, F);;
		this->center_of_sphere = center_of_sphere;
	}

	Eigen::MatrixXd getCenterOfTriangle() {
		return center_of_triangle;
	}

	Eigen::MatrixXd getCenterOfSphere() {
		return center_of_sphere;
	}

	Eigen::MatrixXd getAllCenters() {
		int numF = center_of_sphere.rows();
		Eigen::MatrixXd c(2 * numF, 3);
		c.middleRows(0, numF) = getCenterOfTriangle();
		c.middleRows(numF, numF) = getCenterOfSphere();
		return c;
	}

	void init(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,const OptimizationUtils::InitAuxVariables& typeAuxVar){
		Eigen::VectorXd init = Eigen::Map<const Eigen::VectorXd>(V.data(), V.size());
		Eigen::MatrixX3d normals;
		igl::per_face_normals((Eigen::MatrixX3d)V, (Eigen::MatrixX3i)F, normals);
		Eigen::VectorXd initNormals = Eigen::Map<const Eigen::VectorXd>(normals.data(), F.size());
		Eigen::MatrixXd center0;
		Eigen::VectorXd Radius0;
		if (typeAuxVar == OptimizationUtils::InitAuxVariables::SPHERE)
			OptimizationUtils::Least_Squares_Sphere_Fit(V, F, center0, Radius0);
		else if (typeAuxVar == OptimizationUtils::InitAuxVariables::MESH_CENTER)
			OptimizationUtils::center_of_mesh(V, F, center0, Radius0);
		setCenters(V,F, center0);
		newtonMinimizer->init(
			totalObjective,
			init,
			initNormals,
			Eigen::Map<Eigen::VectorXd>(center0.data(), F.size()),
			Radius0,
			F,
			V
		);
		gradientDescentMinimizer->init(
			totalObjective,
			init,
			initNormals,
			Eigen::Map<Eigen::VectorXd>(center0.data(), F.size()),
			Radius0,
			F,
			V
		);
		adamMinimizer->init(
			totalObjective,
			init,
			initNormals,
			Eigen::Map<Eigen::VectorXd>(center0.data(), F.size()),
			Radius0,
			F,
			V
		);
	}

	void updateActiveMinimizer(const app_utils::MinimizerType minimizer_type) {
		switch (minimizer_type) {
		case app_utils::MinimizerType::NEWTON:
			activeMinimizer = newtonMinimizer;
			break;
		case app_utils::MinimizerType::GRADIENT_DESCENT:
			activeMinimizer = gradientDescentMinimizer;
			break;
		case app_utils::MinimizerType::ADAM_MINIMIZER:
			activeMinimizer = adamMinimizer;
			break;
		}
	}
};
