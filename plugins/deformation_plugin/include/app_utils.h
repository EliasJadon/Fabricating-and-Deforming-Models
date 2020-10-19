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

#include "../../libs/optimization_lib/include/minimizers/Minimizer.h"
#include "../../libs/optimization_lib/include/minimizers/NewtonMinimizer.h"
#include "../../libs/optimization_lib/include/minimizers/GradientDescentMinimizer.h"
#include "../../libs/optimization_lib/include/minimizers/AdamMinimizer.h"

#include "../../libs/optimization_lib/include/objective_functions/ClusterSpheres.h"
#include "../../libs/optimization_lib/include/objective_functions/ClusterNormals.h"
#include "../../libs/optimization_lib/include/objective_functions/FixChosenSpheres.h"
#include "../../libs/optimization_lib/include/objective_functions/STVK.h"
#include "../../libs/optimization_lib/include/objective_functions/SymmetricDirichlet.h"
#include "../../libs/optimization_lib/include/objective_functions/FixAllVertices.h"
#include "../../libs/optimization_lib/include/objective_functions/BendingEdge.h"
#include "../../libs/optimization_lib/include/objective_functions/AuxBendingNormal.h"
#include "../../libs/optimization_lib/include/objective_functions/AuxSpherePerHinge.h"
#include "../../libs/optimization_lib/include/objective_functions/BendingNormal.h"
#include "../../libs/optimization_lib/include/objective_functions/FixChosenVertices.h"

#define RED_COLOR Eigen::Vector3f(1, 0, 0)
#define BLUE_COLOR Eigen::Vector3f(0, 0, 1)
#define GREEN_COLOR Eigen::Vector3f(0, 1, 0)
#define GOLD_COLOR Eigen::Vector3f(1, 215.0f / 255.0f, 0)
#define GREY_COLOR Eigen::Vector3f(0.75, 0.75, 0.75)
#define WHITE_COLOR Eigen::Vector3f(1, 1, 1)
#define BLACK_COLOR Eigen::Vector3f(0, 0, 0)
#define M_PI 3.14159

class FacesGroup {
public:
	Eigen::Vector3f color;
	std::string name;
	std::set<int> faces;

	FacesGroup(const int index) {
		auto put_color = [&](const float r, const float g, const float b) {
			this->color << r / 255.0f, g / 255.0f, b / 255.0f;
		};
		faces.clear();
		if (index == 0)
		{
			put_color(255, 255, 0);//Yellow
			name = "Yellow";
		}
		else if (index == 1)
		{
			put_color(0, 0, 255); //Blue
			name = "Blue";
		}
		else if (index == 2)
		{
			put_color(0, 255, 0); //Green
			name = "Green";
		}
		else if (index == 3)
		{
			put_color(255, 0, 0); //red
			name = "Red";
		}
		else if (index == 4)
		{
			put_color(128, 128, 128); //Gray
			name = "Gray";
		}
		else if (index == 5)
		{
			put_color(128, 0, 128); //Purple
			name = "Purple";
		}
		else if (index == 6)
		{
			put_color(255, 165, 0); //orange
			name = "Orange";
		}
		else if (index == 7)
		{
			put_color(210, 105, 30); //chocolate
			name = "Chocolate";
		}
		else if (index == 8)
		{
			put_color(0, 0, 0); //black
			name = "Black";
		}
		else if (index == 9) 
		{
			put_color(255, 255, 255); //black
			name = "white";
		}
		else
		{
			name = "Err!Overflow";
		}
	}
};

namespace app_utils
{
	enum ClusteringType {
		NoClustering = 0,
		NormalClustering,
		SphereClustering
	};
	enum View {
		HORIZONTAL = 0,
		VERTICAL,
		INPUT_ONLY,
		OUTPUT_ONLY_0
	};
	enum HighlightFaces {
		HOVERED_FACE,
		LOCAL_SPHERE,
		GLOBAL_SPHERE,
		LOCAL_NORMALS,
		GLOBAL_NORMALS
	};
	enum UserInterfaceOptions { 
		NONE,
		FIX_VERTICES,
		FIX_FACES,
		GROUPING_BY_BRUSH,
		GROUPING_BY_ADJ
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

class UniqueColors {
private:
	std::vector<Eigen::Vector3f> colors;
	int index;
	float add(const float x, const float num) {
		float r = x * 255.0f;
		r += num;
		while (r < 0) {
			r += 255;
		}
		r = (int)r % 255;
		return r / 255.0f;
	}
public:
	UniqueColors() {
		auto put_color = [&](const float r, const float g, const float b) {
			this->colors.push_back(Eigen::Vector3f(r / 255.0f, g / 255.0f, b / 255.0f));
		};
		index = 0;
		put_color(255, 0, 0); //red
		put_color(0, 255, 0); //Lime
		put_color(0, 0, 255); //Blue
		put_color(255, 255, 0); //Yellow
		put_color(0, 255, 255); //Cyan / Aqua
		put_color(255, 0, 255); //Magenta / Fuchsia
		put_color(192, 192, 192); //Silver
		put_color(128, 128, 128); //Gray
		put_color(128, 0, 0); //Maroon
		put_color(128, 128, 0); //Olive
		put_color(0, 128, 0); //Green
		put_color(128, 0, 128); //Purple
		put_color(0, 128, 128); //Teal
		put_color(0, 0, 128); //Navy
		put_color(178, 34, 34); //firebrick
		put_color(255, 165, 0); //orange
		put_color(184, 134, 11); //dark golden rod
		put_color(218, 165, 32); //golden rod
		put_color(0, 233, 154); //medium spring green
		put_color(102, 205, 170); //medium aqua marine
		put_color(95, 158, 160); //cadet blue
		put_color(221, 160, 221); //plum
		put_color(218, 112, 214); //orchid
		put_color(245, 222, 179); //wheat
		put_color(205, 133, 63); //peru
		put_color(210, 105, 30); //chocolate
		put_color(230, 230, 250); //lavender
		put_color(240, 248, 255); //alice blue
		put_color(0, 0, 0); //black
		put_color(220, 220, 220); //gainsboro
	}
	Eigen::Vector3f getNext() 
	{
		Eigen::Vector3f c = colors[index];
		colors[index] << add(c(0), 18), add(c(1), -18), add(c(2), 60);
		if ((++index) >= colors.size())
			index = 0;
		return c;
	}
};

class OptimizationOutput {
private:
	Eigen::MatrixXd center_of_faces;
	Eigen::MatrixXd center_of_sphere;
	Eigen::MatrixXd facesNorm;
	Eigen::VectorXd radius_of_sphere;
public:
	std::shared_ptr<NewtonMinimizer> newtonMinimizer;
	std::shared_ptr<GradientDescentMinimizer> gradientDescentMinimizer;
	std::shared_ptr<AdamMinimizer> adamMinimizer;
	std::vector<std::vector<int>> clusters_indices;
	std::shared_ptr<TotalObjective> totalObjective;
	std::shared_ptr<Minimizer> activeMinimizer;
	float prev_camera_zoom;
	Eigen::Vector3f prev_camera_translation;
	Eigen::Quaternionf prev_trackball_angle;
	std::vector<int> *HandlesInd; //pointer to indices in constraitPositional
	Eigen::MatrixX3d *HandlesPosDeformed; //pointer to positions in constraitPositional
	std::vector<int> *CentersInd; //pointer to indices in constraitPositional
	Eigen::MatrixX3d *CentersPosDeformed; //pointer to positions in constraitPositional
	std::vector < std::vector<int>> *ClustersSphereInd, *ClustersNormInd;
	Eigen::MatrixXd color_per_face, Vertices_output;
	Eigen::MatrixXd color_per_sphere_center;
	Eigen::MatrixXd color_per_vertex_center;
	Eigen::MatrixXd color_per_face_norm;
	Eigen::MatrixXd color_per_sphere_edge, color_per_norm_edge;
	int ModelID, CoreID;
	ImVec2 screen_position, screen_size, results_window_position, outputs_window_position;
	bool showSphereEdges, showNormEdges, showTriangleCenters, showSphereCenters, showFacesNorm;

	//Constructor & initialization
	OptimizationOutput(
		igl::opengl::glfw::Viewer* viewer, 
		const app_utils::MinimizerType minimizer_type,
		const OptimizationUtils::LineSearch linesearchType) 
	{
		//update viewer
		CoreID = viewer->append_core(Eigen::Vector4f::Zero());
		viewer->core(CoreID).background_color = Eigen::Vector4f(1, 1, 1, 0);
		viewer->core(CoreID).is_animating = true;
		viewer->core(CoreID).lighting_factor = 0.5;
		// Initialize minimizer thread
		newtonMinimizer = std::make_shared<NewtonMinimizer>(CoreID);
		gradientDescentMinimizer = std::make_shared<GradientDescentMinimizer>(CoreID);
		adamMinimizer = std::make_shared<AdamMinimizer>(CoreID);
		newtonMinimizer->lineSearch_type = linesearchType;
		gradientDescentMinimizer->lineSearch_type = linesearchType;
		adamMinimizer->lineSearch_type = linesearchType;
		updateActiveMinimizer(minimizer_type);
		totalObjective = std::make_shared<TotalObjective>();
		showFacesNorm = showSphereEdges = showNormEdges = 
			showTriangleCenters = showSphereCenters = false;
	}

	~OptimizationOutput() = default;

	void setAuxVariables(
		const Eigen::MatrixXd& V, 
		const Eigen::MatrixXi& F, 
		const Eigen::MatrixXd& center_of_sphere,
		const Eigen::VectorXd& radius_of_sphere,
		const Eigen::MatrixXd& norm
	) {
		this->center_of_faces = OptimizationUtils::center_per_triangle(V, F);
		this->center_of_sphere = center_of_sphere;
		this->radius_of_sphere = radius_of_sphere;
		this->facesNorm = norm;
	}

	double getRadiusOfSphere(int index) {
		return this->radius_of_sphere(index);
	}

	void clustering(const double ratio, const double MSE,const bool isNormal) {
		std::vector<std::vector<int>> clusters_ind;
		std::vector<Eigen::RowVectorXd> clusters_val;
		std::vector<Eigen::RowVectorXd> clusters_center;
		std::vector<double> clusters_radius;
		clusters_init(ratio, MSE, clusters_val, clusters_center, clusters_radius, isNormal);
		
		int numFaces;
		if (isNormal)
			numFaces = facesNorm.rows();
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
						currMSE = (facesNorm.row(fi) - clusters_val[ci]).squaredNorm();
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
					if(isNormal)
						clusters_val.push_back(facesNorm.row(fi));
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
					if(isNormal)
						it_N = clusters_val.erase(it_N);
					else {
						it_C = clusters_center.erase(it_C);
						it_R = clusters_radius.erase(it_R);
					}
				}
				else {
					it_i++; 
					if(isNormal)
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
						avg += facesNorm.row(currf);
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
			std::vector<Eigen::RowVectorXd>::iterator val1,val2,cent1,cent2;
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
			auto& ind2 = ind1+1;
			while(ind1 != clusters_ind.end())
			{
				if (isNormal) {
					val2 = val1 + 1;
				}
				else {
					cent2 = cent1 + 1;
					radius2 = radius1 + 1;
				}
				for (ind2 = ind1+1; ind2 != clusters_ind.end();)
				{
					double diff;
					if (isNormal)
						diff = (*val1 - *val2).squaredNorm();
					else 
						diff = (ratio*((*cent1 - *cent2).norm()) + (1-ratio)*abs(*radius1 - *radius2));

					if (diff < MSE) {
						for (int currf : (*ind2)) {
							ind1->push_back(currf);
						}
						if (isNormal) {
							Eigen::RowVectorXd avg;
							avg.resize(3);
							avg << 0, 0, 0;
							for (int currf : (*ind1)) {
								avg += facesNorm.row(currf);
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
				
				if(isNormal)
					val1++; 
				else {
					cent1++; radius1++;
				}
				ind1++;
			}
		}
		clusters_indices = clusters_ind;
	}

	void clusters_init(
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
			clusters_val.push_back(facesNorm.row(0));
			numFaces = facesNorm.rows();
		}
		else {
			clusters_center.clear();
			clusters_center.push_back(center_of_sphere.row(0));
			clusters_radius.clear();
			clusters_radius.push_back(radius_of_sphere(0));
			numFaces = center_of_sphere.rows();
		}
		clusters_ind.push_back({0});
		
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
					currMSE = (facesNorm.row(fi) - clusters_val[ci]).squaredNorm();
				}
				else {
					currMSE = (ratio*((center_of_sphere.row(fi) - clusters_center[ci]).norm()) + (1-ratio)*abs(radius_of_sphere(fi) - clusters_radius[ci]));
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
						avg += facesNorm.row(currf);
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
				if(isNormal)
					clusters_val.push_back(facesNorm.row(fi));
				else {
					clusters_center.push_back(center_of_sphere.row(fi));
					clusters_radius.push_back(radius_of_sphere(fi));
				}
			}
		}
	}

	void translateCenterOfSphere(const int fi, const Eigen::Vector3d translateValue) {
		this->center_of_sphere.row(fi) += translateValue;
	}

	Eigen::MatrixXd getCenterOfFaces() {
		return center_of_faces;
	}

	Eigen::MatrixXd getFacesNorm() {
		return center_of_faces + facesNorm;
	}

	std::vector<int> GlobNeighSphereCenters(const int fi,const float distance) {
		std::vector<int> Neighbors; Neighbors.clear();
		for (int i = 0; i < center_of_sphere.rows(); i++)
			if (((center_of_sphere.row(fi) - center_of_sphere.row(i)).norm() + abs(radius_of_sphere(fi) - radius_of_sphere(i))) < distance)
				Neighbors.push_back(i);
		return Neighbors;
	}

	std::vector<int> FaceNeigh(const Eigen::Vector3d center, const float distance) {
		std::vector<int> Neighbors; Neighbors.clear();
		for (int i = 0; i < center_of_faces.rows(); i++)
			if ((center.transpose() - center_of_faces.row(i)).norm() < distance)
				Neighbors.push_back(i);
		return Neighbors;
	}

	std::vector<int> GlobNeighNorms(const int fi,const float distance) {
		std::vector<int> Neighbors; Neighbors.clear();
		for (int i = 0; i < facesNorm.rows(); i++)
			if ((facesNorm.row(fi) - facesNorm.row(i)).squaredNorm() < distance)
				Neighbors.push_back(i);
		return Neighbors;
	}

	std::vector<int> getNeigh(const app_utils::HighlightFaces type,const Eigen::MatrixXi& F,const int fi, const float distance) {
		std::vector<int> neigh;
		if (type == app_utils::HighlightFaces::HOVERED_FACE)
			return neigh;
		if(type == app_utils::HighlightFaces::GLOBAL_NORMALS)
			return GlobNeighNorms(fi, distance);
		if(type == app_utils::HighlightFaces::GLOBAL_SPHERE)
			return GlobNeighSphereCenters(fi, distance);
		if(type == app_utils::HighlightFaces::LOCAL_NORMALS)
			neigh = GlobNeighNorms(fi, distance);
		else if(type == app_utils::HighlightFaces::LOCAL_SPHERE)
			neigh = GlobNeighSphereCenters(fi, distance);
		
		std::vector<int> result; result.push_back(fi);
		std::vector<std::vector<std::vector<int>>> TT;
		igl::triangle_triangle_adjacency(F, TT);
		int prevSize;
		do {
			prevSize = result.size();
			result = vectorsIntersection(adjSetOfTriangles(F, result,TT), neigh);
		} while (prevSize != result.size());
		return result;
	}

	std::vector<int> adjSetOfTriangles(const Eigen::MatrixXi& F, const std::vector<int> selected, std::vector<std::vector<std::vector<int>>> TT) {
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

	std::vector<int> vectorsIntersection(const std::vector<int>& A, const std::vector<int>& B) {
		std::vector<int> intersection;
		for (int fi : A) {
			if (std::find(B.begin(), B.end(), fi) != B.end())
				intersection.push_back(fi);
		}
		return intersection;
	}

	Eigen::MatrixXd getCenterOfSphere() {
		return center_of_sphere;
	}

	Eigen::MatrixXd getSphereEdges() {
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

	void initFaceColors(
		const int numF, 
		const Eigen::Vector3f center_sphere_color,
		const Eigen::Vector3f center_vertex_color,
		const Eigen::Vector3f centers_sphere_edge_color,
		const Eigen::Vector3f centers_norm_edge_color,
		const Eigen::Vector3f face_norm_color) 
	{
		color_per_face.resize(numF, 3);
		color_per_sphere_center.resize(numF, 3);
		color_per_vertex_center.resize(numF, 3);
		color_per_face_norm.resize(numF, 3);
		color_per_sphere_edge.resize(numF, 3);
		color_per_norm_edge.resize(numF, 3);

		for (int fi = 0; fi < numF; fi++) {
			color_per_sphere_center.row(fi) = center_sphere_color.cast<double>();
			color_per_vertex_center.row(fi) = center_vertex_color.cast<double>();
			color_per_face_norm.row(fi) = face_norm_color.cast<double>();
			color_per_sphere_edge.row(fi) = centers_sphere_edge_color.cast<double>();
			color_per_norm_edge.row(fi) = centers_norm_edge_color.cast<double>();
		}
	}

	void updateFaceColors(const int fi, const Eigen::Vector3f color) {
		color_per_face.row(fi) = color.cast<double>();
		color_per_sphere_center.row(fi) = color.cast<double>();
		color_per_vertex_center.row(fi) = color.cast<double>();
		color_per_face_norm.row(fi) = color.cast<double>();
		color_per_sphere_edge.row(fi) = color.cast<double>();
		color_per_norm_edge.row(fi) = color.cast<double>();
	}

	void initMinimizers(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,const OptimizationUtils::InitSphereAuxiliaryVariables& typeAuxVar){
		Eigen::VectorXd initVertices = Eigen::Map<const Eigen::VectorXd>(V.data(), V.size());
		Eigen::MatrixX3d normals;
		igl::per_face_normals((Eigen::MatrixX3d)V, (Eigen::MatrixX3i)F, normals);
		Eigen::VectorXd initNormals = Eigen::Map<const Eigen::VectorXd>(normals.data(), F.size());
		Eigen::MatrixXd center0;
		Eigen::VectorXd Radius0;
		if (typeAuxVar == OptimizationUtils::InitSphereAuxiliaryVariables::LEAST_SQUARE_SPHERE)
			OptimizationUtils::Least_Squares_Sphere_Fit(V, F, center0, Radius0);
		else if (typeAuxVar == OptimizationUtils::InitSphereAuxiliaryVariables::MODEL_CENTER_POINT)
			OptimizationUtils::center_of_mesh(V, F, center0, Radius0);
		else if (typeAuxVar == OptimizationUtils::InitSphereAuxiliaryVariables::MINUS_NORMALS) {
			//OptimizationUtils::Least_Squares_Sphere_Fit(V, F, center0, Radius0);
			this->center_of_faces = OptimizationUtils::center_per_triangle(V, F);
			Radius0.resize(F.rows());
			center0.resize(F.rows(),3);
			Radius0.setConstant(0.1);
			
			for (int i = 0; i < center0.rows(); i++) {
				center0.row(i) = this->center_of_faces.row(i) - Radius0(i) * normals.row(i);
			}
		}
		setAuxVariables(V,F, center0, Radius0, normals);
		newtonMinimizer->init(
			totalObjective,
			initVertices,
			initNormals,
			Eigen::Map<Eigen::VectorXd>(center0.data(), F.size()),
			Radius0,
			F,
			V
		);
		gradientDescentMinimizer->init(
			totalObjective,
			initVertices,
			initNormals,
			Eigen::Map<Eigen::VectorXd>(center0.data(), F.size()),
			Radius0,
			F,
			V
		);
		adamMinimizer->init(
			totalObjective,
			initVertices,
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
