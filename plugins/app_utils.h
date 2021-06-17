#pragma once

#include <iostream>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/unproject_in_mesh.h>
#include <igl/unproject_ray.h>
#include <igl/lscm.h>
#include <igl/harmonic.h>
#include <igl/arap.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/project.h>
#include <igl/unproject.h>
#include <igl/triangle_triangle_adjacency.h>
#include <igl/connected_components.h>
#include <igl/slice.h>
#include <igl/edge_lengths.h>
#include <imgui/imgui.h>
#include <chrono>

#include "faces_group.h"
#include "unique_colors.h"

#include "../../libs/Minimizer.h"
#include "../../libs/ROY_soup2D.h"
#include "../../libs/Grouping.h"
#include "../../libs/STVK.h"
#include "../../libs/SDenergy.h"
#include "../../libs/FixAllVertices.h"
#include "../../libs/AuxBendingNormal.h"
#include "../../libs/AuxCylinder.h"
#include "../../libs/AuxSpherePerHinge.h"
#include "../../libs/FixChosenConstraints.h"
#include "../../libs/fixRadius.h"
#include "../../libs/UniformSmoothness.h"

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
	enum ClusteringType {
		NO_CLUSTERING = 0,
		CLUSTERING_NORMAL,
		CLUSTERING_SPHERE,
		CLUSTERING_CYLINDER,
		RGB_NORMAL,
		RGB_SPHERE,
		RGB_CYLINDER
	};
	enum View {
		HORIZONTAL = 0,
		VERTICAL,
		SHOW_INPUT_SCREEN_ONLY,
		SHOW_OUTPUT_SCREEN_ONLY_0
	};
	enum NeighborType {
		CURR_FACE,
		LOCAL_SPHERE,
		GLOBAL_SPHERE,
		LOCAL_NORMALS,
		GLOBAL_NORMALS,
		LOCAL_CYLINDERS,
		GLOBAL_CYLINDERS
	};
	enum UserInterfaceOptions { 
		NONE,
		FIX_VERTICES,
		FIX_FACES,
		BRUSH_WEIGHTS_INCR,
		BRUSH_WEIGHTS_DECR,
		BRUSH_SIGMOID,
		ADJ_WEIGHTS,
		ADJ_SIGMOID
	};
	
	static bool writeOFFwithColors(
		const std::string& path,
		const Eigen::MatrixXd& V,
		const Eigen::MatrixX3i& F,
		const Eigen::MatrixXd& C)
	{
		if (V.cols() != 3 || F.cols() != 3 || C.cols() != 3)
			return false;
		if (V.rows() <= 0 || F.rows() <= 0 || F.rows() != C.rows())
			return false;

		std::ofstream myfile;
		myfile.open(path);
		myfile << "OFF\n";
		myfile << V.rows() << " " << F.rows() << " 0\n";
		for (int vi = 0; vi < V.rows(); vi++) {
			myfile << V(vi, 0) << " " << V(vi, 1) << " " << V(vi, 2) << "\n";
		}
		for (int fi = 0; fi < F.rows(); fi++) {
			myfile << "3 " << F(fi, 0) << " " << F(fi, 1) << " " << F(fi, 2) << " ";
			myfile << int(255 * C(fi, 0)) << " " << int(255 * C(fi, 1)) << " " << int(255 * C(fi, 2)) << "\n";
		}
		myfile.close();
		return true;
	}

	static std::string CurrentTime() {
		char date_buffer[80] = { 0 };
		{
			time_t rawtime_;
			struct tm* timeinfo_;
			time(&rawtime_);
			timeinfo_ = localtime(&rawtime_);
			strftime(date_buffer, 80, "_%H_%M_%S__%d_%m_%Y", timeinfo_);
		}
		return std::string(date_buffer);
	}

	static bool writeTXTFile(
		const std::string& path,
		const std::string& modelName,
		const bool isSphere,
		const std::vector<std::vector<int>>& clustering_faces_indices,
		const Eigen::MatrixXd& V,
		const Eigen::MatrixX3i& F,
		const Eigen::MatrixXd& C,
		const Eigen::VectorXd& Radiuses,
		const Eigen::MatrixX3d& Centers)
	{
		if (V.cols() != 3 || F.cols() != 3 || C.cols() != 3)
			return false;
		if (V.rows() <= 0 || F.rows() <= 0 || F.rows() != C.rows())
			return false;


		std::ofstream myfile;
		myfile.open(path);
		myfile << "\n\n===============================================\n";
		myfile << "Model name: \t"						<< modelName << "\n";
		myfile << "Num Faces: \t"						<< F.rows() << "\n";
		myfile << "Num Vertices: \t"					<< V.rows() << "\n";
		if (isSphere) {
			myfile << "Num spheres: \t" << clustering_faces_indices.size() << "\n";
		}
		else {
			myfile << "Num polygons: \t" << clustering_faces_indices.size() << "\n";
			myfile << "-----------------------List of polygons:" << "\n";
		}
		myfile << "===============================================\n\n\n";
		

			
		
		
		for (int ci = 0; ci < clustering_faces_indices.size(); ci++) {
			myfile << "\n";
			//calculating the avg center&radius for each group/cluster
			double avgRadius = 0;
			Eigen::RowVector3d avgCenter(0, 0, 0), avgColor(0, 0, 0);
			for (int fi = 0; fi < clustering_faces_indices[ci].size(); fi++) {
				const int face_index = clustering_faces_indices[ci][fi];
				if (isSphere) {
					avgRadius += Radiuses[face_index];
					avgCenter = avgCenter + Centers.row(face_index);
				}
				avgColor = avgColor + C.row(face_index);
			}
			if (isSphere) {
				avgRadius /= clustering_faces_indices[ci].size();
				avgCenter /= clustering_faces_indices[ci].size();
			}
			avgColor /= clustering_faces_indices[ci].size();
			

			//output data
			if (isSphere) {
				myfile << "Sphere ID:\t" << ci << "\n";
				myfile << "Radius length: " << avgRadius << "\n";
				myfile << "Center point: " << "(" << avgCenter(0) << ", " << avgCenter(1) << ", " << avgCenter(2) << ")" << "\n";
			}
			else {
				myfile << "Polygon ID:\t" << ci << "\n";
			}
			
			myfile << "color: " << "(" << avgColor(0) << ", " << avgColor(1) << ", " << avgColor(2) << ")" << "\n";
			myfile << "Num faces: " << clustering_faces_indices[ci].size() << "\n";
			myfile << "faces list: ";
			for (int fi = 0; fi < clustering_faces_indices[ci].size(); fi++) {
				const int face_index = clustering_faces_indices[ci][fi];
				myfile << face_index << ", ";
			}
			myfile << "\n";
			myfile << "----------------------------\n";
		}
		
		myfile.close();
		return true;
	}

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

	static int calculateHinges(std::vector<Eigen::Vector2d>& hinges_faceIndex, const Eigen::MatrixX3i& F) {
		std::vector<std::vector<std::vector<int>>> TT;
		igl::triangle_triangle_adjacency(F, TT);
		assert(TT.size() == F.rows());
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
		return hinges_faceIndex.size(); // num_hinges
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

	static char* build_groups_names_list(const std::vector<FacesGroup> fgs) {
		std::string cStr("");
		for (FacesGroup fg : fgs) {
			cStr += fg.name.c_str();
			cStr += '\0';
		}
		cStr += '\0';
		int listLength = cStr.length();
		char* comboList = new char[listLength];
		for (unsigned int i = 0; i < listLength; i++)
			comboList[i] = cStr.at(i);
		return comboList;
	}

	static char* build_inputColoring_list(const int size) {
		std::string cStr("");
		cStr += "None";
		cStr += '\0';
		for (int i = 0; i < size; i++) {
			cStr += "Output ";
			cStr += std::to_string(i).c_str();
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

	static Eigen::RowVector3d get_face_avg(
		const igl::opengl::ViewerData& model,
		const int Translate_Index)
	{
		Eigen::RowVector3d avg; avg << 0, 0, 0;
		Eigen::RowVector3i face = model.F.row(Translate_Index);
		avg += model.V.row(face[0]);
		avg += model.V.row(face[1]);
		avg += model.V.row(face[2]);
		avg /= 3;
		return avg;
	}
}
