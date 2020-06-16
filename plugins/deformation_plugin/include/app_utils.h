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

#include "../../libs/optimization_lib/include/solvers/solver.h"
#include "../../libs/optimization_lib/include/solvers/NewtonSolver.h"
#include "../../libs/optimization_lib/include/solvers/GradientDescentSolver.h"
#include "../../libs/optimization_lib/include/solvers/AdamMinimizer.h"

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
		Horizontal = 0,
		Vertical = 1,
		InputOnly = 2,
		OutputOnly0 = 3
	};
	enum MouseMode { 
		NONE = 0, 
		FACE_SELECT, 
		VERTEX_SELECT, 
		CLEAR 
	};
	enum Parametrization { 
		RANDOM = 0,  
		None 
	};
	enum Distortion { 
		NO_DISTORTION, 
		AREA_DISTORTION, 
		LENGTH_DISTORTION, 
		ANGLE_DISTORTION, 
		TOTAL_DISTORTION 
	};
	enum SolverType {
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

	static void angle_degree(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& angle) {
		int numF = F.rows();
		Eigen::VectorXd Area;
		Eigen::MatrixXd Length, alfa, sum;
		Eigen::ArrayXXd sin_alfa(numF, 3);

		igl::doublearea(V, F, Area);
		igl::edge_lengths(V, F, Length);

		// double_area = a*b*sin(alfa)
		// sin(alfa) = (double_area / a) / b
		sin_alfa.col(0) = Length.col(1).cwiseInverse().cwiseProduct(Length.col(2).cwiseInverse().cwiseProduct(Area));
		sin_alfa.col(1) = Length.col(0).cwiseInverse().cwiseProduct(Length.col(2).cwiseInverse().cwiseProduct(Area));
		sin_alfa.col(2) = Length.col(0).cwiseInverse().cwiseProduct(Length.col(1).cwiseInverse().cwiseProduct(Area));

		// alfa = arcsin ((double_area / a) / b)
		alfa = ((sin_alfa - Eigen::ArrayXXd::Constant(numF, 3, 1e-10)).asin())*(180 / M_PI);


		//here we deal with errors with sin function
		//especially when the sum of the angles isn't equal to 180!
		sum = alfa.rowwise().sum();
		for (int i = 0; i < alfa.rows(); i++) {
			double diff = 180 - sum(i, 0);
			double c0 = 2 * (90 - alfa(i, 0));
			double c1 = 2 * (90 - alfa(i, 1));
			double c2 = 2 * (90 - alfa(i, 2));

			if ((c0 > (diff - 1)) && (c0 < (diff + 1)))
				alfa(i, 0) += c0;
			else if ((c1 > (diff - 1)) && (c1 < (diff + 1)))
				alfa(i, 1) += c1;
			else if ((c2 > (diff - 1)) && (c2 < (diff + 1)))
				alfa(i, 2) += c2;

			/////////////////////////////////
			//sorting - you can remove this part if the order of angles is important!
			if (alfa(i, 0) > alfa(i, 1)) {
				double temp = alfa(i, 0);
				alfa(i, 0) = alfa(i, 1);
				alfa(i, 1) = temp;
			}
			if (alfa(i, 0) > alfa(i, 2)) {
				double temp = alfa(i, 0);
				alfa(i, 0) = alfa(i, 2);
				alfa(i, 2) = alfa(i, 1);
				alfa(i, 1) = temp;
			}
			else if (alfa(i, 1) > alfa(i, 2)) {
				double temp = alfa(i, 1);
				alfa(i, 1) = alfa(i, 2);
				alfa(i, 2) = temp;
			}
			/////////////////////////////////
		}
		angle = alfa;

		////Checkpoint
		//sum = alfa.rowwise().sum();
		//for (int i = 0; i < alfa.rows(); i++) {
		//	cout << i << ": " << alfa(i, 0) << " " << alfa(i, 1) << " " << alfa(i, 2) << " " << sum.row(i) << endl;
		//}
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
public:
	float prev_camera_zoom;
	Eigen::Vector3f prev_camera_translation;
	Eigen::Quaternionf prev_trackball_angle;
	std::vector<int> *HandlesInd; //pointer to indices in constraitPositional
	Eigen::MatrixX3d *HandlesPosDeformed; //pointer to positions in constraitPositional
	Eigen::MatrixXd color_per_face, Vertices_output;
	int ModelID, CoreID;
	ImVec2 window_position, window_size, text_position;
	
	// Solver thread
	std::shared_ptr<NewtonSolver> newton;
	std::shared_ptr<GradientDescentSolver> gradient_descent;
	std::shared_ptr<AdamMinimizer> adam_minimizer;
	std::shared_ptr<solver> solver;
	std::shared_ptr<TotalObjective> totalObjective;

	//Constructor & initialization
	OptimizationOutput(
		igl::opengl::glfw::Viewer* viewer, 
		const app_utils::SolverType solver_type,
		const OptimizationUtils::LineSearch linesearchType) 
	{
		//update viewer
		CoreID = viewer->append_core(Eigen::Vector4f::Zero());
		viewer->core(CoreID).background_color = Eigen::Vector4f(0.9, 0.9, 0.9, 0);
		viewer->core(CoreID).is_animating = true;
		viewer->core(CoreID).lighting_factor = 0.5;
		
		// Initialize solver thread
		std::cout << "CoreID = " << CoreID << std::endl;
		newton = std::make_shared<NewtonSolver>(CoreID);
		gradient_descent = std::make_shared<GradientDescentSolver>(CoreID);
		adam_minimizer = std::make_shared<AdamMinimizer>(CoreID);
	
		switch (solver_type) {
		case app_utils::SolverType::NEWTON:
			solver = newton;
			break;
		case app_utils::SolverType::GRADIENT_DESCENT:
			solver = gradient_descent;
			break;
		case app_utils::SolverType::ADAM_MINIMIZER:
			solver = adam_minimizer;
			break;
		}


		solver->lineSearch_type = linesearchType;
		totalObjective = std::make_shared<TotalObjective>();
	}
	~OptimizationOutput() {}
};
