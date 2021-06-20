#include "..//..//plugins/deformation_plugin.h"
#include <igl/file_dialog_open.h>
#include <GLFW/glfw3.h>
#include <cmath>
#include <igl/writeOFF.h>
#include <igl/boundary_loop.h>
#include <igl/readOFF.h>

#define INPUT_MODEL_SCREEN -1
#define NOT_FOUND -1
#define INSERT false
#define ERASE true

#define ADDING_WEIGHT_PER_HINGE_VALUE 10.0f
#define MAX_WEIGHT_PER_HINGE_VALUE  500.0f //50.0f*ADDING_WEIGHT_PER_HINGE_VALUE
#define ADDING_SIGMOID_PER_HINGE_VALUE 0.9f
#define MAX_SIGMOID_PER_HINGE_VALUE  40.0f //50.0f*ADDING_WEIGHT_PER_HINGE_VALUE


deformation_plugin::deformation_plugin() :
	igl::opengl::glfw::imgui::ImGuiMenu(){}

IGL_INLINE void deformation_plugin::init(igl::opengl::glfw::Viewer *_viewer)
{
	ImGuiMenu::init(_viewer);
	if (!_viewer)
		return;
	for (int i = 0; i < 7; i++)
		CollapsingHeader_prev[i] = CollapsingHeader_curr[i] = false;
	Brush_face_index = Brush_output_index = NOT_FOUND;
	UserInterface_UpdateAllOutputs = false;
	CollapsingHeader_change = false;
	neighbor_distance = brush_radius = 0.3;
	initSphereAuxVariables = OptimizationUtils::InitSphereAuxVariables::MINUS_NORMALS;
	isLoadNeeded = false;
	IsMouseDraggingAnyWindow = false;
	isMinimizerRunning = false;
	energies_window = results_window = outputs_window = true;
	neighbor_Type = app_utils::Neighbor_Type::CURR_FACE;
	IsChoosingGroups = false;
	isModelLoaded = false;
	isUpdateAll = true;
	UserInterface_colorInputModelIndex = 1;
	clustering_Type = app_utils::Clustering_Type::NO_CLUSTERING;
	clustering_w = 0.65;
	faceColoring_type = 1;
	curr_highlighted_output = curr_highlighted_face = NOT_FOUND;
	optimizer_type = Cuda::OptimizerType::Adam;
	linesearch_type = OptimizationUtils::LineSearch::FUNCTION_VALUE;
	UserInterface_groupNum = 0;
	UserInterface_option = app_utils::UserInterfaceOptions::NONE;
	view = app_utils::View::HORIZONTAL;
	Max_Distortion = 5;
	down_mouse_x = down_mouse_y = NOT_FOUND;
	Vertex_Energy_color = RED_COLOR;
	Highlighted_face_color = Eigen::Vector3f(153 / 255.0f, 0, 153 / 255.0f);
	Neighbors_Highlighted_face_color = Eigen::Vector3f(1, 102 / 255.0f, 1);
	center_sphere_color = Eigen::Vector3f(0, 1, 1);
	center_vertex_color = Eigen::Vector3f(128 / 255.0f, 128 / 255.0f, 128 / 255.0f);
	Color_sphere_edges = Color_normal_edge = Eigen::Vector3f(0 / 255.0f, 100 / 255.0f, 100 / 255.0f);
	face_norm_color = Eigen::Vector3f(0, 1, 1);
	Fixed_vertex_color = Fixed_face_color = BLUE_COLOR;
	Dragged_vertex_color = Dragged_face_color = GREEN_COLOR;
	model_color = GREY_COLOR;
	text_color = BLACK_COLOR;
	//update input viewer
	inputCoreID = viewer->core_list[0].id;
	viewer->core(inputCoreID).background_color = Eigen::Vector4f(1, 1, 1, 0);
	viewer->core(inputCoreID).is_animating = true;
	viewer->core(inputCoreID).lighting_factor = 0.5;
	//Load multiple views
	Outputs.push_back(OptimizationOutput(viewer, optimizer_type, linesearch_type));
	core_size = 1.0 / (Outputs.size() + 1.0);
	//maximize window
	glfwMaximizeWindow(viewer->window);
}

void deformation_plugin::load_new_model(const std::string modelpath) 
{
	if (isModelLoaded)
		clear_sellected_faces_and_vertices();
	modelPath = modelpath;
	if (modelPath.length() == 0)
		return;
	modelName = app_utils::ExtractModelName(modelPath);
	stop_minimizer_thread();
	if (isModelLoaded) 
	{
		//remove previous data
		while (Outputs.size() > 0)
			remove_output(0);
		viewer->load_mesh_from_file(modelPath.c_str());
		viewer->erase_mesh(0);
	}
	else 
		viewer->load_mesh_from_file(modelPath.c_str());
	inputModelID = viewer->data_list[0].id;
	for (int i = 0; i < Outputs.size(); i++)
	{
		viewer->load_mesh_from_file(modelPath.c_str());
		Outputs[i].ModelID = viewer->data_list[i + 1].id;
		init_objective_functions(i);
	}
	if (isModelLoaded)
		add_output();
	viewer->core(inputCoreID).align_camera_center(InputModel().V, InputModel().F);
	for (int i = 0; i < Outputs.size(); i++)
		viewer->core(Outputs[i].CoreID).align_camera_center(OutputModel(i).V, OutputModel(i).F);
	
	//set rotation type to 3D mode
	viewer->core(inputCoreID).trackball_angle = Eigen::Quaternionf::Identity();
	viewer->core(inputCoreID).orthographic = false;
	viewer->core(inputCoreID).set_rotation_type(igl::opengl::ViewerCore::RotationType(1));
	isModelLoaded = true;
	isLoadNeeded = false;
}

IGL_INLINE void deformation_plugin::draw_viewer_menu()
{
	if (isModelLoaded && UserInterface_option != app_utils::UserInterfaceOptions::NONE)
	{
		CollapsingHeader_user_interface();
		Draw_output_window();
		Draw_results_window();
		Draw_energies_window();
		return;
	}
	float w = ImGui::GetContentRegionAvailWidth();
	float p = ImGui::GetStyle().FramePadding.x;
	if (ImGui::Button("Load##Mesh", ImVec2((w - p) / 2.f, 0)))
	{
		modelPath = igl::file_dialog_open();
		isLoadNeeded = true;
	}
	if (isLoadNeeded)
	{
		load_new_model(modelPath);
		isLoadNeeded = false;
	}
	if (!isModelLoaded)
		return;
	ImGui::SameLine();
	if (ImGui::Button("Save##Mesh", ImVec2((w - p) / 2.f, 0)))
		viewer->open_dialog_save_mesh();

	if (ImGui::DragInt("save output index", &save_output_index)) {
		if (save_output_index >= Outputs.size() || save_output_index < 0)
			save_output_index = 0;
	}
	
	
	if (ImGui::Button("save Sphere", ImVec2((w - p) / 2.f, 0)) && Outputs[save_output_index].clustering_faces_indices.size()) {
		// Multiply all the mesh by "factor". Relevant only for spheres. 
		double factor = 1;
		for (auto& obj : Outputs[save_output_index].totalObjective->objectiveList) {
			auto fR = std::dynamic_pointer_cast<fixRadius>(obj);
			if (fR != NULL && fR->w != 0)
				factor = fR->alpha;
		}
		// Get mesh data
		OptimizationOutput O = Outputs[save_output_index];
		Eigen::MatrixXd colors = O.clustering_faces_colors;
		Eigen::MatrixXd V_OUT = factor * OutputModel(save_output_index).V;
		Eigen::MatrixXd V_IN = factor * InputModel().V;
		Eigen::MatrixXi F = OutputModel(save_output_index).F;
		Eigen::VectorXd Radiuses = factor * Outputs[save_output_index].getRadiusOfSphere();
		Eigen::MatrixXd Centers = factor * Outputs[save_output_index].getCenterOfSphere();
		
		// Create new Directory for saving the data
		std::string main_file_path = OptimizationUtils::ProjectPath() + "models\\OutputModels\\" + modelName + app_utils::CurrentTime() + "\\";
		std::string aux_file_path = main_file_path + "Auxiliary_Variables\\";
		std::string parts_file_path = main_file_path + "Sphere_Parts\\";
		std::string parts_color_file_path = main_file_path + "Sphere_Parts_With_Colors\\";
		std::string file_name = modelName + std::to_string(save_output_index);
		if (mkdir(main_file_path.c_str()) == -1 ||
			mkdir(parts_file_path.c_str()) == -1 ||
			mkdir(aux_file_path.c_str()) == -1 ||
			mkdir(parts_color_file_path.c_str()) == -1)
		{
			std::cerr << "Error :  " << strerror(errno) << std::endl;
			exit(1);
		}

		// Save each cluster in the new directory
		for (int clus_index = 0; clus_index < O.clustering_faces_indices.size(); clus_index++)
		{
			std::vector<int> clus_faces_index = O.clustering_faces_indices[clus_index];
			Eigen::MatrixX3i clus_faces_val(clus_faces_index.size(), 3);
			Eigen::MatrixX3d clus_faces_color(clus_faces_index.size(), 3);

			double sumRadius = 0;
			Eigen::RowVector3d sumCenters(0, 0, 0);
			for (int fi = 0; fi < clus_faces_index.size(); fi++)
			{
				sumRadius += Radiuses(clus_faces_index[fi]);
				sumCenters += Centers.row(clus_faces_index[fi]);
				clus_faces_val.row(fi) = F.row(clus_faces_index[fi]);
				clus_faces_color.row(fi) = colors.row(clus_faces_index[fi]);
			}
			Eigen::RowVector3d avgCenter = sumCenters / clus_faces_index.size();
			double avgRadius = sumRadius / clus_faces_index.size();

			Eigen::MatrixX3d clus_vertices(V_OUT.rows(), 3);
			for (int vi = 0; vi < V_OUT.rows(); vi++)
				clus_vertices.row(vi) = V_OUT.row(vi);
			// Save the current cluster in "off" file format
			std::string clus_file_name = parts_file_path + file_name + "_sphere_" + std::to_string(clus_index) + ".off";
			std::string clus_file_name_colors = parts_color_file_path + file_name + "_sphere_" + std::to_string(clus_index) + "_withColors.off";
			app_utils::writeOFFwithColors(clus_file_name_colors, clus_vertices, clus_faces_val, clus_faces_color);
			igl::writeOFF(clus_file_name, clus_vertices, clus_faces_val);
		}
		// Save the final mesh in "off" file format
		igl::writeOFF(main_file_path + file_name + "_Output.off", V_OUT, F);
		igl::writeOFF(main_file_path + file_name + "_Input.off", V_IN, F);
		app_utils::writeOFFwithColors(main_file_path + file_name + "_Output_withColors.off", V_OUT, F, colors);
		app_utils::writeOFFwithColors(main_file_path + file_name + "_Input_withColors.off", V_IN, F, colors);
		app_utils::writeTXTFile(main_file_path + file_name + "ReadMe.txt", modelName, true,
			O.clustering_faces_indices, V_OUT, F, colors, Radiuses, Centers);
		//save auxiliary variables
		Eigen::MatrixXi temp(1, 3);
		temp << 1, 3, 2;
		igl::writeOFF(aux_file_path + file_name + "_Aux_Centers.off", Centers, temp);
		Eigen::MatrixXd mat_radiuses(Radiuses.size(), 3);
		mat_radiuses.setZero();
		mat_radiuses.col(0) = Radiuses;
		igl::writeOFF(aux_file_path + file_name + "_Aux_Radiuses.off", mat_radiuses, temp);
	}
	ImGui::SameLine();
	if (ImGui::Button("Save Planar", ImVec2((w - p) / 2.f, 0)) && Outputs[save_output_index].clustering_faces_indices.size()) {
		// Get mesh data
		OptimizationOutput O = Outputs[save_output_index];
		Eigen::MatrixXd colors = O.clustering_faces_colors;
		Eigen::MatrixXd V_OUT = OutputModel(save_output_index).V;
		Eigen::MatrixXd V_IN = InputModel().V;
		Eigen::MatrixXi F = OutputModel(save_output_index).F;
		Eigen::VectorXd Radiuses = Outputs[save_output_index].getRadiusOfSphere();
		Eigen::MatrixXd Centers = Outputs[save_output_index].getCenterOfSphere();
		Eigen::MatrixXd Normals = Outputs[save_output_index].getFacesNormals();

		// Create new Directory for saving the data
		std::string main_file_path = OptimizationUtils::ProjectPath() + "models\\OutputModels\\" + modelName + app_utils::CurrentTime() + "\\";
		std::string aux_file_path = main_file_path + "Auxiliary_Variables\\";
		std::string parts_file_path = main_file_path + "Polygon_Parts\\";
		std::string parts_color_file_path = main_file_path + "Polygon_Parts_With_Colors\\";
		std::string file_name = modelName + std::to_string(save_output_index);
		if (mkdir(main_file_path.c_str()) == -1 ||
			mkdir(parts_file_path.c_str()) == -1 ||
			mkdir(aux_file_path.c_str()) == -1 ||
			mkdir(parts_color_file_path.c_str()) == -1)
		{
			std::cerr << "Error :  " << strerror(errno) << std::endl;
			exit(1);
		}
		
		// Save each cluster in the new directory
		for (int polygon_index = 0; polygon_index < O.clustering_faces_indices.size(); polygon_index++)
		{
			std::vector<int> clus_F_indices = O.clustering_faces_indices[polygon_index];
			const int clus_Num_Faces = clus_F_indices.size();
			Eigen::MatrixX3i clus_F(clus_Num_Faces, 3);
			Eigen::MatrixX3d clus_color(clus_Num_Faces, 3);

			for (int fi = 0; fi < clus_Num_Faces; fi++)
			{
				clus_F.row(fi) = F.row(clus_F_indices[fi]);
				clus_color.row(fi) = colors.row(clus_F_indices[fi]);
			}
			// Save the current cluster in "off" file format
			std::string clus_file_name = parts_file_path + file_name + "_polygon_" + std::to_string(polygon_index) + ".off";
			std::string clus_file_name_colors = parts_color_file_path + file_name + "_polygon_" + std::to_string(polygon_index) + "_withColors.off";
			igl::writeOFF(clus_file_name, V_OUT, clus_F);
			app_utils::writeOFFwithColors(clus_file_name_colors, V_OUT, clus_F, clus_color);
		}
		// Save the final mesh in "off" file format
		igl::writeOFF(main_file_path + file_name + "_Input.off", V_IN, F);
		igl::writeOFF(main_file_path + file_name + "_Output.off", V_OUT, F);
		app_utils::writeOFFwithColors(main_file_path + file_name + "_Input_withColors.off", V_IN, F, colors);
		app_utils::writeOFFwithColors(main_file_path + file_name + "_Output_withColors.off", V_OUT, F, colors);
		app_utils::writeTXTFile(main_file_path + file_name + "ReadMe.txt", modelName, false,
			O.clustering_faces_indices, V_OUT, F, colors, Radiuses, Centers);
		//save auxiliary variables
		Eigen::MatrixXi temp(1, 3);
		temp << 1, 3, 2;
		igl::writeOFF(aux_file_path + file_name + "_Aux_Normals.off", Normals, temp);
	}

	ImGui::Checkbox("Outputs window", &outputs_window);
	ImGui::Checkbox("Results window", &results_window);
	ImGui::Checkbox("Energy window", &energies_window);
	CollapsingHeader_face_coloring();
	CollapsingHeader_screen();
	CollapsingHeader_clustering();
	CollapsingHeader_minimizer();
	CollapsingHeader_cores(viewer->core(inputCoreID), viewer->data(inputModelID));
	CollapsingHeader_models(viewer->data(inputModelID));
	CollapsingHeader_colors();
	Draw_output_window();
	Draw_results_window();
	Draw_energies_window();
	CollapsingHeader_update();
}

void deformation_plugin::CollapsingHeader_update()
{
	CollapsingHeader_change = false;
	int changed_index = NOT_FOUND;
	for (int i = 0; i < 7; i++)
	{
		if (CollapsingHeader_curr[i] && !CollapsingHeader_prev[i])
		{
			changed_index = i;
			CollapsingHeader_change = true;
		}
	}
	if (CollapsingHeader_change)
	{
		for (int i = 0; i < 7; i++)
			CollapsingHeader_prev[i] = CollapsingHeader_curr[i] = false;
		CollapsingHeader_prev[changed_index] = CollapsingHeader_curr[changed_index] = true;
	}
}

void deformation_plugin::CollapsingHeader_colors()
{
	if (CollapsingHeader_change)
		ImGui::SetNextTreeNodeOpen(CollapsingHeader_curr[0]);
	if (ImGui::CollapsingHeader("colors"))
	{
		CollapsingHeader_curr[0] = true;
		ImGui::ColorEdit3("Highlighted face", Highlighted_face_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
		ImGui::ColorEdit3("Center sphere", center_sphere_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
		ImGui::ColorEdit3("Center vertex", center_vertex_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
		ImGui::ColorEdit3("Sphere edge", Color_sphere_edges.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
		
		ImGui::ColorEdit3("Normal edge", Color_normal_edge.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
		ImGui::ColorEdit3("Face norm", face_norm_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
		ImGui::ColorEdit3("Neighbors Highlighted face", Neighbors_Highlighted_face_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
		ImGui::ColorEdit3("Fixed face", Fixed_face_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
		ImGui::ColorEdit3("Dragged face", Dragged_face_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
		ImGui::ColorEdit3("Fixed vertex", Fixed_vertex_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
		ImGui::ColorEdit3("Dragged vertex", Dragged_vertex_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
		ImGui::ColorEdit3("Model", model_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
		ImGui::ColorEdit3("Vertex Energy", Vertex_Energy_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
		ImGui::ColorEdit4("Text", text_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
	}
}

void deformation_plugin::CollapsingHeader_face_coloring()
{
	if (CollapsingHeader_change)
		ImGui::SetNextTreeNodeOpen(CollapsingHeader_curr[1]);
	if (ImGui::CollapsingHeader("Face coloring"))
	{
		CollapsingHeader_curr[1] = true;
		ImGui::Combo("type", (int *)(&faceColoring_type), app_utils::build_color_energies_list(Outputs[0].totalObjective));
		ImGui::PushItemWidth(80 * menu_scaling());
		ImGui::DragFloat("Max Distortion", &Max_Distortion, 0.05f, 0.01f, 10000.0f);
		ImGui::PopItemWidth();
	}
}

void deformation_plugin::CollapsingHeader_screen()
{
	if (CollapsingHeader_change)
		ImGui::SetNextTreeNodeOpen(CollapsingHeader_curr[2]);
	if (ImGui::CollapsingHeader("Screen options"))
	{
		CollapsingHeader_curr[2] = true;
		if (ImGui::Combo("View type", (int *)(&view), app_utils::build_view_names_list(Outputs.size())))
		{
			int frameBufferWidth, frameBufferHeight;
			glfwGetFramebufferSize(viewer->window, &frameBufferWidth, &frameBufferHeight);
			post_resize(frameBufferWidth, frameBufferHeight);
		}
		if (view == app_utils::View::HORIZONTAL ||
			view == app_utils::View::VERTICAL)
		{
			if (ImGui::SliderFloat("Core Size", &core_size, 0, 1.0 / Outputs.size(), std::to_string(core_size).c_str(), 1))
			{
				int frameBufferWidth, frameBufferHeight;
				glfwGetFramebufferSize(viewer->window, &frameBufferWidth, &frameBufferHeight);
				post_resize(frameBufferWidth, frameBufferHeight);
			}
		}
	}
}

void deformation_plugin::CollapsingHeader_user_interface()
{
	if (!ImGui::CollapsingHeader("User Interface"))
	{
		ImGui::Combo("Coloring Input", (int*)(&UserInterface_colorInputModelIndex), app_utils::build_inputColoring_list(Outputs.size()));
		ImGui::Checkbox("Update All", &UserInterface_UpdateAllOutputs);
		if (UserInterface_option == app_utils::UserInterfaceOptions::ADJ_WEIGHTS || UserInterface_option == app_utils::UserInterfaceOptions::ADJ_SIGMOID)
			ImGui::Combo("Neighbor type", (int *)(&neighbor_Type), "Curr Face\0Local Sphere\0Global Sphere\0Local Normals\0Global Normals\0\0");
		if (UserInterface_option == app_utils::UserInterfaceOptions::ADJ_WEIGHTS || UserInterface_option == app_utils::UserInterfaceOptions::ADJ_SIGMOID)
			ImGui::DragFloat("Neighbors Distance", &neighbor_distance, 0.0005f, 0.00001f, 10000.0f,"%.5f");
		if (UserInterface_option == app_utils::UserInterfaceOptions::BRUSH_WEIGHTS_INCR || 
		    UserInterface_option == app_utils::UserInterfaceOptions::BRUSH_WEIGHTS_DECR || 
			UserInterface_option == app_utils::UserInterfaceOptions::BRUSH_SIGMOID)
			ImGui::DragFloat("Brush Radius", &brush_radius);
		if (ImGui::Button("Clear sellected faces & vertices"))
			clear_sellected_faces_and_vertices();
	}
}

void deformation_plugin::CollapsingHeader_clustering()
{
	if (CollapsingHeader_change)
		ImGui::SetNextTreeNodeOpen(CollapsingHeader_curr[3]);
	if (ImGui::CollapsingHeader("Clustering"))
	{
		CollapsingHeader_curr[3] = true;
		if (ImGui::Button("Copy")) {
			int ind;
			for (int f : Outputs[0].UserInterface_FixedFaces) {
				ind = f;
			}
			copy_index.push_back(ind);
			
			std::vector<Eigen::Vector2d>& hinge_to_face_mapping = Outputs[0].Energy_auxSpherePerHinge->hinges_faceIndex;
			int num_hinges = Outputs[0].Energy_auxSpherePerHinge->mesh_indices.num_hinges;
			double* hinge_val = Outputs[0].Energy_auxSpherePerHinge->weight_PerHinge.host_arr;
			std::set<int> chosen_faces;
			for (int hi = 0; hi < num_hinges; hi++) {
				if (hinge_val[hi] > 1) {
					chosen_faces.insert(hinge_to_face_mapping[hi][0]);
					chosen_faces.insert(hinge_to_face_mapping[hi][1]);
				}
			}
			paste_index.push_back(chosen_faces);
		}
		ImGui::SameLine();
		if (ImGui::Button("Reset")) {
			copy_index.clear();
			paste_index.clear();
		}
		
		ImGui::Combo("Clus. Type", (int*)(&clustering_Type), "No Clustering\0Normals\0Spheres\0\0");
		ImGui::DragFloat("Bright. Weight", &clustering_w, 0.001f, 0, 1);
			
		if (clustering_Type != app_utils::Clustering_Type::NO_CLUSTERING)
		{
			ImGui::Checkbox("Use HashMap", &clustering_hashMap);
			if (clustering_hashMap)
				ImGui::DragFloat("Min Distance", &Clustering_MinDistance, 0.000001f, 0, 1,"%.8f");	
		}
	}
}

void deformation_plugin::CollapsingHeader_minimizer()
{
	if (CollapsingHeader_change)
		ImGui::SetNextTreeNodeOpen(CollapsingHeader_curr[4]);
	if (ImGui::CollapsingHeader("Minimizer"))
	{
		CollapsingHeader_curr[4] = true;
		if (ImGui::Button("Run one iter"))
			run_one_minimizer_iter();
		if (ImGui::Checkbox("Run Minimizer", &isMinimizerRunning))
			isMinimizerRunning ? start_minimizer_thread() : stop_minimizer_thread();
		if (ImGui::Combo("Optimizer", (int *)(&optimizer_type), "Gradient Descent\0Adam\0\0"))
			change_minimizer_type(optimizer_type);
		if (ImGui::Combo("init sphere var", (int *)(&initSphereAuxVariables), "Sphere Fit\0Mesh Center\0Minus Normal\0\0"))
			init_aux_variables();
		if (initSphereAuxVariables == OptimizationUtils::InitSphereAuxVariables::MINUS_NORMALS &&
			ImGui::DragFloat("radius length", &radius_length_minus_normal, 0.01f, 0.0f, 1000.0f, "%.7f"))
			init_aux_variables();
		if (initSphereAuxVariables == OptimizationUtils::InitSphereAuxVariables::SPHERE_FIT) 
		{
			if (ImGui::DragInt("Neigh From", &(InitMinimizer_NeighLevel_From), 1, 1, 200))
				init_aux_variables();
			if (ImGui::DragInt("Neigh To", &(InitMinimizer_NeighLevel_To), 1, 1, 200))
				init_aux_variables();
		}

		if (ImGui::Combo("line search", (int *)(&linesearch_type), "Gradient Norm\0Function Value\0Constant Step\0\0")) {
			for (auto& o : Outputs)
				o.minimizer->lineSearch_type = linesearch_type;
		}
		if (linesearch_type == OptimizationUtils::LineSearch::CONSTANT_STEP && ImGui::DragFloat("Step value", &constantStep_LineSearch, 0.0001f, 0.0f, 1.0f, "%.7f")) {
			for (auto& o : Outputs)
				o.minimizer->constantStep_LineSearch = constantStep_LineSearch;	
		}
		if (ImGui::Button("Check gradients"))
			checkGradients();
	}
}

void deformation_plugin::CollapsingHeader_cores(igl::opengl::ViewerCore& core, igl::opengl::ViewerData& data)
{
	if (!outputs_window)
		return;
	ImGui::PushID(core.id);
	if (CollapsingHeader_change)
		ImGui::SetNextTreeNodeOpen(CollapsingHeader_curr[5]);
	if (ImGui::CollapsingHeader(("Core " + std::to_string(data.id)).c_str()))
	{
		CollapsingHeader_curr[5] = true;
		if (ImGui::Button("Center object", ImVec2(-1, 0)))
			core.align_camera_center(data.V, data.F);
		if (ImGui::Button("Snap canonical view", ImVec2(-1, 0)))
			viewer->snap_to_canonical_quaternion();
		// Zoom & Lightining factor
		ImGui::PushItemWidth(80 * menu_scaling());
		ImGui::DragFloat("Zoom", &(core.camera_zoom), 0.05f, 0.1f, 100000.0f);
		ImGui::DragFloat("Lighting factor", &(core.lighting_factor), 0.05f, 0.1f, 20.0f);
		// Select rotation type
		int rotation_type = static_cast<int>(core.rotation_type);
		static Eigen::Quaternionf trackball_angle = Eigen::Quaternionf::Identity();
		static bool orthographic = true;
		if (ImGui::Combo("Camera Type", &rotation_type, "Trackball\0Two Axes\02D Mode\0\0"))
		{
			using RT = igl::opengl::ViewerCore::RotationType;
			auto new_type = static_cast<RT>(rotation_type);
			if (new_type != core.rotation_type)
			{
				if (new_type == RT::ROTATION_TYPE_NO_ROTATION)
				{
					trackball_angle = core.trackball_angle;
					orthographic = core.orthographic;
					core.trackball_angle = Eigen::Quaternionf::Identity();
					core.orthographic = true;
				}
				else if (core.rotation_type == RT::ROTATION_TYPE_NO_ROTATION)
				{
					core.trackball_angle = trackball_angle;
					core.orthographic = orthographic;
				}
				core.set_rotation_type(new_type);
			}
		}
		if(ImGui::Checkbox("Orthographic view", &(core.orthographic)) && isUpdateAll)
			for (auto& c : viewer->core_list)
				c.orthographic = core.orthographic;
		ImGui::PopItemWidth();
		if (ImGui::ColorEdit4("Background", core.background_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel) && isUpdateAll)
			for (auto& c : viewer->core_list)
				c.background_color = core.background_color;
	}
	ImGui::PopID();
}

void deformation_plugin::CollapsingHeader_models(igl::opengl::ViewerData& data)
{
	if (!outputs_window)
		return;
	auto make_checkbox = [&](const char *label, unsigned int &option) {
		bool temp = option;
		bool res = ImGui::Checkbox(label, &temp);
		option = temp;
		return res;
	};
	ImGui::PushID(data.id);
	if (CollapsingHeader_change)
		ImGui::SetNextTreeNodeOpen(CollapsingHeader_curr[6]);
	if (ImGui::CollapsingHeader((modelName + " " + std::to_string(data.id)).c_str()))
	{
		CollapsingHeader_curr[6] = true;
		if (ImGui::Checkbox("Face-based", &(data.face_based)))
		{
			data.dirty = igl::opengl::MeshGL::DIRTY_ALL;
			if(isUpdateAll)
			{
				for (auto& d : viewer->data_list)
				{
					d.dirty = igl::opengl::MeshGL::DIRTY_ALL;
					d.face_based = data.face_based;
				}
			}
		}
		if (make_checkbox("Show texture", data.show_texture) && isUpdateAll)
			for (auto& d : viewer->data_list)
				d.show_texture = data.show_texture;
		if (ImGui::Checkbox("Invert normals", &(data.invert_normals))) {
			if (isUpdateAll)
			{
				for (auto& d : viewer->data_list)
				{
					d.dirty |= igl::opengl::MeshGL::DIRTY_NORMAL;
					d.invert_normals = data.invert_normals;
				}
			}
			else
				data.dirty |= igl::opengl::MeshGL::DIRTY_NORMAL;
		}
		if (make_checkbox("Show overlay", data.show_overlay) && isUpdateAll)
			for (auto& d : viewer->data_list)
				d.show_overlay = data.show_overlay;
		if (make_checkbox("Show overlay depth", data.show_overlay_depth) && isUpdateAll)
			for (auto& d : viewer->data_list)
				d.show_overlay_depth = data.show_overlay_depth;
		if (ImGui::ColorEdit4("Line color", data.line_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel) && isUpdateAll)
			for (auto& d : viewer->data_list)
				d.line_color = data.line_color;
		ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.3f);
		if (ImGui::DragFloat("Shininess", &(data.shininess), 0.05f, 0.0f, 100.0f) && isUpdateAll)
			for (auto& d : viewer->data_list)
				d.shininess = data.shininess;
		ImGui::PopItemWidth();
		if (make_checkbox("Wireframe", data.show_lines) && isUpdateAll)
			for (auto& d : viewer->data_list)
				d.show_lines = data.show_lines;
		if (make_checkbox("Fill", data.show_faces) && isUpdateAll)
			for(auto& d: viewer->data_list)
				d.show_faces = data.show_faces;
		if (ImGui::Checkbox("Show vertex labels", &(data.show_vertid)) && isUpdateAll)
			for (auto& d : viewer->data_list)
				d.show_vertid = data.show_vertid;
		if (ImGui::Checkbox("Show faces labels", &(data.show_faceid)) && isUpdateAll)
			for (auto& d : viewer->data_list)
				d.show_faceid = data.show_faceid;
	}
	ImGui::PopID();
}

void deformation_plugin::Draw_energies_window()
{
	if (!energies_window)
		return;
	ImGui::SetNextWindowPos(energies_window_position);
	ImGui::Begin("Energies & Timing", NULL, ImGuiWindowFlags_AlwaysAutoResize);
	int id = 0;
	ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.0f, 0.6f, 0.0f, 1.0f));
	if (ImGui::Button(("Add one more " + modelName).c_str()))
		add_output();
	ImGui::PopStyleColor();
	
	//add automatic lambda change
	if (ImGui::BeginTable("Lambda table", 10, ImGuiTableFlags_Resizable))
	{
		ImGui::TableSetupColumn("ID", ImGuiTableColumnFlags_WidthAlwaysAutoResize);
		ImGui::TableSetupColumn("On/Off", ImGuiTableColumnFlags_WidthAlwaysAutoResize);
		ImGui::TableSetupColumn("Start from", ImGuiTableColumnFlags_WidthAlwaysAutoResize);
		ImGui::TableSetupColumn("Stop at", ImGuiTableColumnFlags_WidthAlwaysAutoResize);
		ImGui::TableSetupColumn("#iter//lambda", ImGuiTableColumnFlags_WidthAlwaysAutoResize);
		ImGui::TableSetupColumn("#iter", ImGuiTableColumnFlags_WidthAlwaysAutoResize);
		ImGui::TableSetupColumn("Time [ms]", ImGuiTableColumnFlags_WidthAlwaysAutoResize);
		ImGui::TableSetupColumn("Avg Time [ms]", ImGuiTableColumnFlags_WidthAlwaysAutoResize);
		ImGui::TableSetupColumn("lineSearch step size", ImGuiTableColumnFlags_WidthAlwaysAutoResize);
		ImGui::TableSetupColumn("lineSearch #iter", ImGuiTableColumnFlags_WidthAlwaysAutoResize);
		ImGui::TableAutoHeaders();
		ImGui::Separator();
		ImGui::TableNextRow();
		ImGui::PushItemWidth(80);
		for (auto&out : Outputs) {
			ImGui::PushID(id++);
			const int  i64_zero = 0, i64_max = 100000.0;
			ImGui::Text((modelName + std::to_string(out.ModelID)).c_str());
			ImGui::TableNextCell();
			ImGui::Checkbox("##On/Off", &out.minimizer->isAutoLambdaRunning);
			ImGui::TableNextCell();
			ImGui::DragInt("##From", &(out.minimizer->autoLambda_from), 1, i64_zero, i64_max);
			ImGui::TableNextCell();
			ImGui::DragInt("##count", &(out.minimizer->autoLambda_count), 1, i64_zero, i64_max, "2^%d");
			ImGui::TableNextCell();
			ImGui::DragInt("##jump", &(out.minimizer->autoLambda_jump), 1, 1, i64_max);
			
			ImGui::TableNextCell();
			ImGui::Text(std::to_string(out.minimizer->getNumiter()).c_str());
			ImGui::TableNextCell();
			ImGui::Text(std::to_string(out.minimizer->timer_curr).c_str());
			ImGui::TableNextCell();
			ImGui::Text(std::to_string(out.minimizer->timer_avg).c_str());
			ImGui::TableNextCell();
			ImGui::Text(("2^" + std::to_string(int(log2(out.minimizer->init_step_size)))).c_str());
			ImGui::TableNextCell();
			ImGui::Text(std::to_string(out.minimizer->linesearch_numiterations).c_str());
			ImGui::PopID();
			ImGui::TableNextRow();
		}
		ImGui::PopItemWidth();
		ImGui::EndTable();
	}
	
	if (Outputs.size() != 0) {
		if (ImGui::BeginTable("Unconstrained weights table", Outputs[0].totalObjective->objectiveList.size() + 2, ImGuiTableFlags_Resizable))
		{
			ImGui::TableSetupColumn("ID", ImGuiTableColumnFlags_WidthAlwaysAutoResize);
			for (auto& obj : Outputs[0].totalObjective->objectiveList) {
				ImGui::TableSetupColumn(obj->name.c_str(), ImGuiTableColumnFlags_WidthAlwaysAutoResize);
			}
			ImGui::TableAutoHeaders();
			ImGui::Separator();

			ImGui::TableNextRow();
			for (int i = 0; i < Outputs.size(); i++) 
			{
				ImGui::Text((modelName + std::to_string(Outputs[i].ModelID)).c_str());
				ImGui::TableNextCell();
				ImGui::PushItemWidth(80);
				for (auto& obj : Outputs[i].totalObjective->objectiveList) {
					ImGui::PushID(id++);
					ImGui::DragFloat("##w", &(obj->w), 0.05f, 0.0f, 100000.0f);
					auto SD = std::dynamic_pointer_cast<SDenergy>(obj);
					auto fR = std::dynamic_pointer_cast<fixRadius>(obj);
					auto ABN = std::dynamic_pointer_cast<AuxBendingNormal>(obj);
					auto AS = std::dynamic_pointer_cast<AuxSpherePerHinge>(obj);
					if (obj->w) {
						if (fR != NULL) {

							ImGui::DragInt("min", &(fR->min));
							fR->min = fR->min < 1 ? 1 : fR->min;
							ImGui::DragInt("max", &(fR->max));
							fR->max = fR->max > fR->min ? fR->max : fR->min + 1;
							
							ImGui::DragFloat("alpha", &(fR->alpha), 0.001);

							

							Eigen::VectorXd Radiuses = Outputs[save_output_index].getRadiusOfSphere();
							if (ImGui::Button("update Alpha")) {
								fR->alpha = fR->max / Radiuses.maxCoeff();
							}
							ImGui::Text(("R max: " + std::to_string(Radiuses.maxCoeff() * fR->alpha)).c_str());
							ImGui::Text(("R min: " + std::to_string(Radiuses.minCoeff() * fR->alpha)).c_str());
							
						}

						if (ABN != NULL)
							ImGui::Combo("Function", (int*)(&(ABN->penaltyFunction)), "Quadratic\0Exponential\0Sigmoid\0\0");
						if (AS != NULL)
							ImGui::Combo("Function", (int*)(&(AS->penaltyFunction)), "Quadratic\0Exponential\0Sigmoid\0\0");
						
						if (ABN != NULL && ABN->penaltyFunction == Cuda::PenaltyFunction::SIGMOID) {
							ImGui::Text(("2^" + std::to_string(int(log2(ABN->get_SigmoidParameter())))).c_str());
							ImGui::SameLine();
							if (ImGui::Button("*", ImVec2(ImGui::GetFrameHeight(), ImGui::GetFrameHeight())))
							{
								ABN->Inc_SigmoidParameter();
							}
							ImGui::SameLine();
							if (ImGui::Button("/", ImVec2(ImGui::GetFrameHeight(), ImGui::GetFrameHeight())))
							{
								ABN->Dec_SigmoidParameter();
							}
							const double  f64_zero = 0, f64_max = 100000.0;
							ImGui::DragScalar("w1", ImGuiDataType_Double, &(ABN->w1), 0.05f, &f64_zero, &f64_max);
							ImGui::DragScalar("w2", ImGuiDataType_Double, &(ABN->w2), 0.05f, &f64_zero, &f64_max);
							ImGui::DragScalar("w3", ImGuiDataType_Double, &(ABN->w3), 0.05f, &f64_zero, &f64_max);
						}
						if (AS != NULL && AS->penaltyFunction == Cuda::PenaltyFunction::SIGMOID) {
							ImGui::Text(("2^" + std::to_string(int(log2(AS->get_SigmoidParameter())))).c_str());
							ImGui::SameLine();
							if (ImGui::Button("*", ImVec2(ImGui::GetFrameHeight(), ImGui::GetFrameHeight())))
							{
								AS->Inc_SigmoidParameter();
							}
							ImGui::SameLine();
							if (ImGui::Button("/", ImVec2(ImGui::GetFrameHeight(), ImGui::GetFrameHeight())))
							{
								AS->Dec_SigmoidParameter();
							}
							const double  f64_zero = 0, f64_max = 100000.0;
							ImGui::DragScalar("w1", ImGuiDataType_Double, &(AS->w1), 0.05f, &f64_zero, &f64_max);
							ImGui::DragScalar("w2", ImGuiDataType_Double, &(AS->w2), 0.05f, &f64_zero, &f64_max);
						}
					}
					ImGui::TableNextCell();
					ImGui::PopID();
				}
				ImGui::PushID(id++);
				ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.6f, 0.0f, 0.0f, 1.0f));
				if (Outputs.size() > 1 && ImGui::Button("Remove"))
					remove_output(i);
				ImGui::PopStyleColor();
				ImGui::PopID();
				ImGui::PopItemWidth();
				ImGui::TableNextRow();
			}	
			ImGui::EndTable();
		}
	}
	ImVec2 w_size = ImGui::GetWindowSize();
	energies_window_position = ImVec2(0.5 * global_screen_size[0] - 0.5 * w_size[0], global_screen_size[1] - w_size[1]);
	//close the window
	ImGui::End();
}

void deformation_plugin::Draw_output_window()
{
	if (!outputs_window)
		return;
	for (auto& out : Outputs) 
	{
		ImGui::SetNextWindowSize(ImVec2(200, 300));
		ImGui::SetNextWindowPos(out.outputs_window_position);
		ImGui::Begin(("Output settings " + std::to_string(out.CoreID)).c_str(),
			NULL,
			ImGuiWindowFlags_NoTitleBar |
			ImGuiWindowFlags_NoResize |
			ImGuiWindowFlags_NoMove
		);
		ImGui::Checkbox("Update all models together", &isUpdateAll);
		CollapsingHeader_cores(viewer->core(out.CoreID), viewer->data(out.ModelID));
		CollapsingHeader_models(viewer->data(out.ModelID));

		ImGui::Text("Show:");
		if (ImGui::Checkbox("Norm", &(out.showFacesNorm)) && isUpdateAll)
			for (auto&oi : Outputs)
				oi.showFacesNorm = out.showFacesNorm;
		ImGui::SameLine();
		if (ImGui::Checkbox("Norm Edges", &(out.showNormEdges)) && isUpdateAll)
			for (auto&oi : Outputs)
				oi.showNormEdges = out.showNormEdges;
		if (ImGui::Checkbox("Sphere", &(out.showSphereCenters)) && isUpdateAll)
			for (auto&oi : Outputs)
				oi.showSphereCenters = out.showSphereCenters;
		ImGui::SameLine();
		if (ImGui::Checkbox("Sphere Edges", &(out.showSphereEdges)) && isUpdateAll)
			for (auto&oi : Outputs)
				oi.showSphereEdges = out.showSphereEdges;
		if (ImGui::Checkbox("Face Centers", &(out.showTriangleCenters)) && isUpdateAll)
			for (auto&oi : Outputs)
				oi.showTriangleCenters = out.showTriangleCenters;
		ImGui::End();
	}
}

void deformation_plugin::Draw_results_window()
{
	if (!results_window)
		return;
	for (auto& out : Outputs)
	{
		bool bOpened2(true);
		ImColor c(text_color[0], text_color[1], text_color[2], 1.0f);
		ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 0));
		ImGui::Begin(("Text " + std::to_string(out.CoreID)).c_str(), &bOpened2,
			ImGuiWindowFlags_NoTitleBar |
			ImGuiWindowFlags_NoResize |
			ImGuiWindowFlags_NoMove |
			ImGuiWindowFlags_NoScrollbar |
			ImGuiWindowFlags_NoScrollWithMouse |
			ImGuiWindowFlags_NoBackground |
			ImGuiWindowFlags_NoCollapse |
			ImGuiWindowFlags_NoSavedSettings |
			ImGuiWindowFlags_NoInputs |
			ImGuiWindowFlags_NoFocusOnAppearing |
			ImGuiWindowFlags_NoBringToFrontOnFocus);
		ImGui::SetWindowPos(out.results_window_position);
		ImGui::SetWindowSize(out.screen_size);
		ImGui::SetWindowCollapsed(false);
		
		
		ImGui::TextColored(c, (
			std::string("Num Faces: ") +
			std::to_string(InputModel().F.rows()) +
			std::string("\tNum Vertices: ") +
			std::to_string(InputModel().V.rows()) +
			std::string("\nGrad Size: ") +
			std::to_string(out.totalObjective->objectiveList[0]->grad.size) +
			std::string("\tNum Clusters: ") +
			std::to_string(out.clustering_faces_indices.size())
			).c_str());
		ImGui::TextColored(c, (std::string(out.totalObjective->name) + std::string(" energy ") + std::to_string(out.totalObjective->energy_value)).c_str());
		ImGui::TextColored(c, (std::string(out.totalObjective->name) + std::string(" gradient ") + std::to_string(out.totalObjective->gradient_norm)).c_str());
		for (auto& obj : out.totalObjective->objectiveList) {
			if (obj->w)
			{
				ImGui::TextColored(c, (std::string(obj->name) + std::string(" energy ") + std::to_string(obj->energy_value)).c_str());
				ImGui::TextColored(c, (std::string(obj->name) + std::string(" gradient ") + std::to_string(obj->gradient_norm)).c_str());
			}
		}
		ImGui::End();
		ImGui::PopStyleColor();
	}
}

void deformation_plugin::clear_sellected_faces_and_vertices() 
{
	for (auto&o : Outputs)
	{
		o.Energy_auxSpherePerHinge->Clear_HingesWeights();
		o.Energy_auxSpherePerHinge->Clear_HingesSigmoid();
		o.Energy_auxBendingNormal->Clear_HingesWeights();
		o.Energy_auxBendingNormal->Clear_HingesSigmoid();
		
		o.UserInterface_FixedFaces.clear();
		for (auto& c : o.UserInterface_facesGroups)
			c.faces.clear();
		o.UserInterface_FixedVertices.clear();
		o.printNormals_saveVertices.clear();
	}
	update_ext_fixed_vertices();
}

void deformation_plugin::update_parameters_for_all_cores() 
{
	if (!isUpdateAll)
		return;
	for (auto& core : viewer->core_list) 
	{
		int output_index = NOT_FOUND;
		for (int i = 0; i < Outputs.size(); i++)
			if (core.id == Outputs[i].CoreID)
				output_index = i;
		if (output_index == NOT_FOUND)
		{
			if (this->prev_camera_zoom != core.camera_zoom ||
				this->prev_camera_translation != core.camera_translation ||
				this->prev_trackball_angle.coeffs() != core.trackball_angle.coeffs()
				) 
			{
				for (auto& c : viewer->core_list) 
				{
					c.camera_zoom = core.camera_zoom;
					c.camera_translation = core.camera_translation;
					c.trackball_angle = core.trackball_angle;
				}	
				this->prev_camera_zoom = core.camera_zoom;
				this->prev_camera_translation = core.camera_translation;
				this->prev_trackball_angle = core.trackball_angle;
				for (auto&o : Outputs)
				{
					o.prev_camera_zoom = core.camera_zoom;
					o.prev_camera_translation = core.camera_translation;
					o.prev_trackball_angle = core.trackball_angle;
				}
			}
		}
		else 
		{
			if (Outputs[output_index].prev_camera_zoom != core.camera_zoom ||
				Outputs[output_index].prev_camera_translation != core.camera_translation ||
				Outputs[output_index].prev_trackball_angle.coeffs() != core.trackball_angle.coeffs()
				) 
			{
				for (auto& c : viewer->core_list) 
				{
					c.camera_zoom = core.camera_zoom;
					c.camera_translation = core.camera_translation;
					c.trackball_angle = core.trackball_angle;
				}	
				this->prev_camera_zoom = core.camera_zoom;
				this->prev_camera_translation = core.camera_translation;
				this->prev_trackball_angle = core.trackball_angle;
				for (auto&o : Outputs) 
				{
					o.prev_camera_zoom = core.camera_zoom;
					o.prev_camera_translation = core.camera_translation;
					o.prev_trackball_angle = core.trackball_angle;
				}	
			}
		}
	}
}

void deformation_plugin::remove_output(const int output_index) 
{
	stop_minimizer_thread();
	viewer->erase_core(1 + output_index);
	viewer->erase_mesh(1 + output_index);
	Outputs.erase(Outputs.begin() + output_index);
	
	core_size = 1.0 / (Outputs.size() + 1.0);
	int frameBufferWidth, frameBufferHeight;
	glfwGetFramebufferSize(viewer->window, &frameBufferWidth, &frameBufferHeight);
	post_resize(frameBufferWidth, frameBufferHeight);
}

void deformation_plugin::add_output() 
{
	stop_minimizer_thread();
	Outputs.push_back(OptimizationOutput(viewer, optimizer_type,linesearch_type));
	viewer->load_mesh_from_file(modelPath.c_str());
	Outputs[Outputs.size() - 1].ModelID = viewer->data_list[Outputs.size()].id;
	init_objective_functions(Outputs.size() - 1);
	//Update the scene
	viewer->core(inputCoreID).align_camera_center(InputModel().V, InputModel().F);
	viewer->core(inputCoreID).is_animating = true;
	for (int i = 0; i < Outputs.size(); i++) 
	{
		viewer->core(Outputs[i].CoreID).align_camera_center(OutputModel(i).V, OutputModel(i).F);
		viewer->core(Outputs[i].CoreID).is_animating = true;
	}
	core_size = 1.0 / (Outputs.size() + 1.0);
	int frameBufferWidth, frameBufferHeight;
	glfwGetFramebufferSize(viewer->window, &frameBufferWidth, &frameBufferHeight);
	post_resize(frameBufferWidth, frameBufferHeight);
}

IGL_INLINE void deformation_plugin::post_resize(int w, int h)
{
	if (!viewer)
		return;
	if (view == app_utils::View::HORIZONTAL) 
	{
		viewer->core(inputCoreID).viewport = Eigen::Vector4f(0, 0, w - w * Outputs.size() * core_size, h);
		for (int i = 0; i < Outputs.size(); i++) 
		{
			Outputs[i].screen_position = ImVec2(w - w * (Outputs.size() - i) * core_size, 0);
			Outputs[i].screen_size = ImVec2(w * core_size, h);
			Outputs[i].results_window_position = Outputs[i].screen_position;
			Outputs[i].outputs_window_position = ImVec2(w - w * (Outputs.size() - (i + 1)) * core_size - 200, 0);
		}
	}
	if (view == app_utils::View::VERTICAL) 
	{
		viewer->core(inputCoreID).viewport = Eigen::Vector4f(0, Outputs.size() * h * core_size, w, h - Outputs.size() * h * core_size);
		for (int i = 0; i < Outputs.size(); i++) 
		{
			Outputs[i].screen_position = ImVec2(0, (Outputs.size() - i - 1) * h * core_size);
			Outputs[i].screen_size = ImVec2(w, h * core_size);
			Outputs[i].outputs_window_position = ImVec2(w-205, h - Outputs[i].screen_position[1] - Outputs[i].screen_size[1]);
			Outputs[i].results_window_position = ImVec2(0, Outputs[i].outputs_window_position[1]);
		}
	}
	if (view == app_utils::View::SHOW_INPUT_SCREEN_ONLY) 
	{
		viewer->core(inputCoreID).viewport = Eigen::Vector4f(0, 0, w, h);
		for (auto&o : Outputs) 
		{
			o.screen_position = ImVec2(w, h);
			o.screen_size = ImVec2(0, 0);
			o.results_window_position = o.screen_position;
			//o.outputs_window_position = 
		}
	}
 	if (view >= app_utils::View::SHOW_OUTPUT_SCREEN_ONLY_0) 
	{
 		viewer->core(inputCoreID).viewport = Eigen::Vector4f(0, 0, 0, 0);
 		for (auto&o : Outputs) 
		{
 			o.screen_position = ImVec2(w, h);
 			o.screen_size = ImVec2(0, 0);
 			o.results_window_position = o.screen_position;
 		}
 		// what does this means?
 		Outputs[view - app_utils::View::SHOW_OUTPUT_SCREEN_ONLY_0].screen_position = ImVec2(0, 0);
 		Outputs[view - app_utils::View::SHOW_OUTPUT_SCREEN_ONLY_0].screen_size = ImVec2(w, h);
 		Outputs[view - app_utils::View::SHOW_OUTPUT_SCREEN_ONLY_0].results_window_position = ImVec2(w*0.8, 0);
 	}		
	for (auto& o : Outputs)
		viewer->core(o.CoreID).viewport = Eigen::Vector4f(o.screen_position[0], o.screen_position[1], o.screen_size[0] + 1, o.screen_size[1] + 1);
	energies_window_position = ImVec2(0.1 * w, 0.8 * h);
	global_screen_size = ImVec2(w, h);
}

void deformation_plugin::brush_erase_or_insert() 
{
	if (pick_face(&Brush_output_index, &Brush_face_index, intersec_point))
	{
		if (UserInterface_option == app_utils::UserInterfaceOptions::BRUSH_WEIGHTS_INCR) {
			double add = ADDING_WEIGHT_PER_HINGE_VALUE;
			if (UI_status == ERASE)
				add = -ADDING_WEIGHT_PER_HINGE_VALUE;
			const std::vector<int> brush_faces = Outputs[Brush_output_index].FaceNeigh(intersec_point.cast<double>(), brush_radius);
			if (UserInterface_UpdateAllOutputs) {
				for (auto& out : Outputs) {
					out.Energy_auxBendingNormal->Incr_HingesWeights(brush_faces, add);
					out.Energy_auxSpherePerHinge->Incr_HingesWeights(brush_faces, add);
					out.Energy_auxBendingNormal->Reset_HingesSigmoid(brush_faces);
					out.Energy_auxSpherePerHinge->Reset_HingesSigmoid(brush_faces);
				}
			}
			else {
				Outputs[Brush_output_index].Energy_auxBendingNormal->Incr_HingesWeights(brush_faces, add);
				Outputs[Brush_output_index].Energy_auxSpherePerHinge->Incr_HingesWeights(brush_faces, add);
				Outputs[Brush_output_index].Energy_auxBendingNormal->Reset_HingesSigmoid(brush_faces);
				Outputs[Brush_output_index].Energy_auxSpherePerHinge->Reset_HingesSigmoid(brush_faces);
			}
		}
		else if (UserInterface_option == app_utils::UserInterfaceOptions::BRUSH_WEIGHTS_DECR) {
			double value = 0;
			if (UI_status == ERASE)
				value = 1;
			const std::vector<int> brush_faces = Outputs[Brush_output_index].FaceNeigh(intersec_point.cast<double>(), brush_radius);
			if (UserInterface_UpdateAllOutputs) {
				for (auto& out : Outputs) {
					out.Energy_auxBendingNormal->Set_HingesWeights(brush_faces, value);
					out.Energy_auxSpherePerHinge->Set_HingesWeights(brush_faces, value);
				}
			}
			else {
				Outputs[Brush_output_index].Energy_auxBendingNormal->Set_HingesWeights(brush_faces, value);
				Outputs[Brush_output_index].Energy_auxSpherePerHinge->Set_HingesWeights(brush_faces, value);
			}
		}
		else if(UserInterface_option == app_utils::UserInterfaceOptions::BRUSH_SIGMOID) {
			double factor = ADDING_SIGMOID_PER_HINGE_VALUE;
			if (UI_status != INSERT)
				factor = 1 / ADDING_SIGMOID_PER_HINGE_VALUE;
			const std::vector<int> brush_faces = Outputs[Brush_output_index].FaceNeigh(intersec_point.cast<double>(), brush_radius);
			if (UserInterface_UpdateAllOutputs) {
				for (auto& out : Outputs) {
					out.Energy_auxBendingNormal->Update_HingesSigmoid(brush_faces, factor);
					out.Energy_auxSpherePerHinge->Update_HingesSigmoid(brush_faces, factor);
				}
			}
			else {
				Outputs[Brush_output_index].Energy_auxBendingNormal->Update_HingesSigmoid(brush_faces, factor);
				Outputs[Brush_output_index].Energy_auxSpherePerHinge->Update_HingesSigmoid(brush_faces, factor);
			}
		}
	}
}

IGL_INLINE bool deformation_plugin::mouse_move(int mouse_x, int mouse_y)
{
	if (!isModelLoaded || IsMouseDraggingAnyWindow)
		return true;	
	if (IsChoosingGroups && (UserInterface_option == app_utils::UserInterfaceOptions::ADJ_SIGMOID || UserInterface_option == app_utils::UserInterfaceOptions::ADJ_WEIGHTS))
	{
		Eigen::Vector3f _;
		pick_face(&curr_highlighted_output, &curr_highlighted_face, _);
		return true;
	}

	bool returnTrue = false;
	for (int i = 0; i < Outputs.size(); i++)
	{
		auto& out = Outputs[i];
		if (out.UserInterface_IsTranslate && UserInterface_option == app_utils::UserInterfaceOptions::FIX_FACES)
		{
			Eigen::RowVector3d face_avg_pt = app_utils::get_face_avg(OutputModel(Output_Translate_ID), out.UserInterface_TranslateIndex);
			Eigen::Vector3f translation = app_utils::computeTranslation(mouse_x, down_mouse_x, mouse_y, down_mouse_y, face_avg_pt, OutputCore(Output_Translate_ID));
			if (UserInterface_UpdateAllOutputs)
				for (auto& o : Outputs)
					o.translateFaces(out.UserInterface_TranslateIndex, translation.cast<double>());
			else
				out.translateFaces(out.UserInterface_TranslateIndex, translation.cast<double>());
			down_mouse_x = mouse_x;
			down_mouse_y = mouse_y;
			returnTrue = true;
		}
		if (out.UserInterface_IsTranslate && UserInterface_option == app_utils::UserInterfaceOptions::FIX_VERTICES)
		{
			Eigen::RowVector3d vertex_pos = OutputModel(Output_Translate_ID).V.row(out.UserInterface_TranslateIndex);
			Eigen::Vector3f translation = app_utils::computeTranslation(mouse_x, down_mouse_x, mouse_y, down_mouse_y, vertex_pos, OutputCore(Output_Translate_ID));
			if (UserInterface_UpdateAllOutputs)
				for (int io=0;io<Outputs.size();io++)
					OutputModel(io).V.row(Outputs[io].UserInterface_TranslateIndex) += translation.cast<double>();
			else
				OutputModel(i).V.row(out.UserInterface_TranslateIndex) += translation.cast<double>();
			down_mouse_x = mouse_x;
			down_mouse_y = mouse_y;
			update_ext_fixed_vertices();
			returnTrue = true;
		}
		if (out.UserInterface_IsTranslate && (UserInterface_option == app_utils::UserInterfaceOptions::BRUSH_WEIGHTS_INCR || UserInterface_option == app_utils::UserInterfaceOptions::BRUSH_WEIGHTS_DECR || UserInterface_option == app_utils::UserInterfaceOptions::BRUSH_SIGMOID))
		{
			brush_erase_or_insert();
			returnTrue = true;
		}
	}
	if (returnTrue)
		return true;
	return false;
}

IGL_INLINE bool deformation_plugin::mouse_scroll(float delta_y) 
{
	if (!isModelLoaded || IsMouseDraggingAnyWindow || ImGui::IsAnyWindowHovered())
		return true;
	for (auto&out : Outputs)
	{
		if (out.UserInterface_IsTranslate && (UserInterface_option == app_utils::UserInterfaceOptions::BRUSH_WEIGHTS_INCR || UserInterface_option == app_utils::UserInterfaceOptions::BRUSH_WEIGHTS_DECR || UserInterface_option == app_utils::UserInterfaceOptions::BRUSH_SIGMOID))
		{
			brush_radius += delta_y * 0.005;
			brush_radius = std::max<float>(0.005, brush_radius);
			return true;
		}
	}
	if (IsChoosingGroups && (UserInterface_option == app_utils::UserInterfaceOptions::ADJ_WEIGHTS || UserInterface_option == app_utils::UserInterfaceOptions::ADJ_SIGMOID))
	{
		neighbor_distance += delta_y * 0.05;
		neighbor_distance = std::max<float>(0.005, neighbor_distance);
		return true;
	}
	return false;
}

IGL_INLINE bool deformation_plugin::mouse_up(int button, int modifier) 
{
	for (auto&out : Outputs)
		out.UserInterface_IsTranslate = false;
	IsMouseDraggingAnyWindow = false;

	if (IsChoosingGroups) 
	{
		IsChoosingGroups = false;
		curr_highlighted_output = curr_highlighted_face = NOT_FOUND;
		Eigen::Vector3f _;
		int face_index, output_index;
		if (pick_face(&output_index, &face_index, _))
		{
			std::vector<int> neigh_faces = Outputs[output_index].getNeigh(neighbor_Type, InputModel().F, face_index, neighbor_distance);
			if (UserInterface_option == app_utils::UserInterfaceOptions::ADJ_WEIGHTS) {
				double add = 5*ADDING_WEIGHT_PER_HINGE_VALUE;
				if (UI_status != INSERT)
					add = -5*ADDING_WEIGHT_PER_HINGE_VALUE;
				if (UserInterface_UpdateAllOutputs) {
					for (auto& out : Outputs) {
						out.Energy_auxBendingNormal->Incr_HingesWeights(neigh_faces, add);
						out.Energy_auxSpherePerHinge->Incr_HingesWeights(neigh_faces, add);
						out.Energy_auxBendingNormal->Reset_HingesSigmoid(neigh_faces);
						out.Energy_auxSpherePerHinge->Reset_HingesSigmoid(neigh_faces);
						
					}
				}
				else {
					Outputs[output_index].Energy_auxBendingNormal->Incr_HingesWeights(neigh_faces, add);
					Outputs[output_index].Energy_auxSpherePerHinge->Incr_HingesWeights(neigh_faces, add);
					Outputs[output_index].Energy_auxBendingNormal->Reset_HingesSigmoid(neigh_faces);
					Outputs[output_index].Energy_auxSpherePerHinge->Reset_HingesSigmoid(neigh_faces);
				}
			}
			else if (UserInterface_option == app_utils::UserInterfaceOptions::ADJ_SIGMOID) {
				double factor = pow(ADDING_SIGMOID_PER_HINGE_VALUE, 5);
				if (UI_status != INSERT)
					factor = 1 / pow(ADDING_SIGMOID_PER_HINGE_VALUE, 5);
				if (UserInterface_UpdateAllOutputs) {
					for (auto& out : Outputs) {
						out.Energy_auxBendingNormal->Update_HingesSigmoid(neigh_faces, factor);
						out.Energy_auxSpherePerHinge->Update_HingesSigmoid(neigh_faces, factor);
					}
				}
				else {
					Outputs[output_index].Energy_auxBendingNormal->Update_HingesSigmoid(neigh_faces, factor);
					Outputs[output_index].Energy_auxSpherePerHinge->Update_HingesSigmoid(neigh_faces, factor);
				}
			}
		}
	}
	return false;
}

IGL_INLINE bool deformation_plugin::mouse_down(int button, int modifier) 
{
	if (ImGui::IsAnyWindowHovered())
		IsMouseDraggingAnyWindow = true;
	down_mouse_x = viewer->current_mouse_x;
	down_mouse_y = viewer->current_mouse_y;
	
	if (UserInterface_option == app_utils::UserInterfaceOptions::FIX_FACES && button == GLFW_MOUSE_BUTTON_LEFT)
	{
		Eigen::Vector3f _;
		int face_index, output_index;
		if (pick_face(&output_index, &face_index, _, true) && output_index != INPUT_MODEL_SCREEN)
		{
			if (UserInterface_UpdateAllOutputs)
			{
				for (auto& out : Outputs)
				{
					out.UserInterface_FixedFaces.insert(face_index);
					out.UserInterface_IsTranslate = true;
					out.UserInterface_TranslateIndex = face_index;
				}
			}
			else
			{
				Outputs[output_index].UserInterface_FixedFaces.insert(face_index);
				Outputs[output_index].UserInterface_IsTranslate = true;
				Outputs[output_index].UserInterface_TranslateIndex = face_index;
			}
		}
	}
	else if (UserInterface_option == app_utils::UserInterfaceOptions::FIX_FACES && button == GLFW_MOUSE_BUTTON_MIDDLE)
	{
		Eigen::Vector3f _;
		int face_index, output_index;
		if (pick_face(&output_index, &face_index, _) && output_index != INPUT_MODEL_SCREEN)
		{
			if (UserInterface_UpdateAllOutputs)
				for (auto& out : Outputs)
					out.UserInterface_FixedFaces.erase(face_index);
			else
				Outputs[output_index].UserInterface_FixedFaces.erase(face_index);
		}
	}
	else if (UserInterface_option == app_utils::UserInterfaceOptions::FIX_VERTICES && button == GLFW_MOUSE_BUTTON_LEFT)
	{
		int output_index, vertex_index;
		if (pick_vertex(&output_index, &vertex_index, true) && output_index != INPUT_MODEL_SCREEN)
		{
			if (UserInterface_UpdateAllOutputs)
			{
				for (auto& out : Outputs)
				{
					out.UserInterface_FixedVertices.insert(vertex_index);
					out.printNormals_saveVertices.push_back(vertex_index);
					out.UserInterface_IsTranslate = true;
					out.UserInterface_TranslateIndex = vertex_index;
				}
			}
			else
			{
				Outputs[output_index].UserInterface_FixedVertices.insert(vertex_index);
				Outputs[output_index].printNormals_saveVertices.push_back(vertex_index);
				Outputs[output_index].UserInterface_IsTranslate = true;
				Outputs[output_index].UserInterface_TranslateIndex = vertex_index;
			}
			update_ext_fixed_vertices();
		}
	}
	else if (UserInterface_option == app_utils::UserInterfaceOptions::FIX_VERTICES && button == GLFW_MOUSE_BUTTON_MIDDLE)
	{
		int output_index, vertex_index;
		if (pick_vertex(&output_index, &vertex_index, true) && output_index != INPUT_MODEL_SCREEN)
		{
			if (UserInterface_UpdateAllOutputs)
				for (auto& out : Outputs)
					out.UserInterface_FixedVertices.erase(vertex_index);
			else
				Outputs[output_index].UserInterface_FixedVertices.erase(vertex_index);
			update_ext_fixed_vertices();
		}
	}
	else if ((UserInterface_option == app_utils::UserInterfaceOptions::BRUSH_SIGMOID || UserInterface_option == app_utils::UserInterfaceOptions::BRUSH_WEIGHTS_INCR || UserInterface_option == app_utils::UserInterfaceOptions::BRUSH_WEIGHTS_DECR) && button == GLFW_MOUSE_BUTTON_LEFT)
	{
		if (pick_face(&Brush_output_index, &Brush_face_index, intersec_point))
		{
			UI_status = INSERT;
			Outputs[Brush_output_index].UserInterface_IsTranslate = true;
		}
	}
	else if ((UserInterface_option == app_utils::UserInterfaceOptions::BRUSH_SIGMOID || UserInterface_option == app_utils::UserInterfaceOptions::BRUSH_WEIGHTS_INCR || UserInterface_option == app_utils::UserInterfaceOptions::BRUSH_WEIGHTS_DECR) && button == GLFW_MOUSE_BUTTON_MIDDLE)
	{
		if (pick_face(&Brush_output_index, &Brush_face_index, intersec_point))
		{
			UI_status = ERASE;
			Outputs[Brush_output_index].UserInterface_IsTranslate = true;
		}
	}
	else if ((UserInterface_option == app_utils::UserInterfaceOptions::ADJ_SIGMOID || UserInterface_option == app_utils::UserInterfaceOptions::ADJ_WEIGHTS) && button == GLFW_MOUSE_BUTTON_LEFT)
	{
		IsChoosingGroups = true;
		UI_status = INSERT;
		Eigen::Vector3f _;
		pick_face(&curr_highlighted_output, &curr_highlighted_face, _);
	}
	else if ((UserInterface_option == app_utils::UserInterfaceOptions::ADJ_SIGMOID || UserInterface_option == app_utils::UserInterfaceOptions::ADJ_WEIGHTS) && button == GLFW_MOUSE_BUTTON_MIDDLE)
	{
		IsChoosingGroups = true;
		UI_status = ERASE;
		Eigen::Vector3f _;
		pick_face(&curr_highlighted_output, &curr_highlighted_face, _);
	}

	return false;
}

IGL_INLINE bool deformation_plugin::key_pressed(unsigned int key, int modifiers) 
{
	if ((key == 'c' || key == 'C') && modifiers == 1)
		clear_sellected_faces_and_vertices();
	if ((key == 'x' || key == 'X') && modifiers == 1) {
		if (clustering_Type != app_utils::Clustering_Type::NO_CLUSTERING)
			clustering_Type = app_utils::Clustering_Type::NO_CLUSTERING;
		else {
			clustering_Type = app_utils::Clustering_Type::SPHERES;
			if (neighbor_Type == app_utils::Neighbor_Type::LOCAL_NORMALS)
				clustering_Type = app_utils::Clustering_Type::NORMALS;
		}
	}
	if ((key == 'a' || key == 'A') && modifiers == 1) 
	{
		modelPath = OptimizationUtils::ProjectPath() + 
			"\\models\\InputModels\\from_2k_to_10k\\island.off";
		isLoadNeeded = true;
	}
	if ((key == 's' || key == 'S') && modifiers == 1) {
		modelPath = OptimizationUtils::ProjectPath() + 
			"\\models\\InputModels\\Bear_without_eyes.off";
		isLoadNeeded = true;
	}
	if (isModelLoaded && (key == 'q' || key == 'Q') && modifiers == 1) 
	{
		neighbor_Type = app_utils::Neighbor_Type::LOCAL_NORMALS;
		clustering_Type = app_utils::Clustering_Type::NORMALS;
		for (auto&out : Outputs) {
			out.showFacesNorm = true;
			out.showSphereEdges = out.showNormEdges = 
				out.showTriangleCenters = out.showSphereCenters = false;
		}
		for (OptimizationOutput& out : Outputs) 
		{
			for (auto& obj : out.totalObjective->objectiveList) 
			{
				std::shared_ptr<AuxSpherePerHinge> AS = std::dynamic_pointer_cast<AuxSpherePerHinge>(obj);
				std::shared_ptr<AuxBendingNormal> ABN = std::dynamic_pointer_cast<AuxBendingNormal>(obj);
				if(ABN != NULL)
					ABN->w = 1.6;
				if (AS != NULL)
					AS->w = 0;
			}
		}
	}
	if (isModelLoaded && (key == 'w' || key == 'W') && modifiers == 1) 
	{
		neighbor_Type = app_utils::Neighbor_Type::LOCAL_SPHERE;
		clustering_Type = app_utils::Clustering_Type::SPHERES;
		initSphereAuxVariables = OptimizationUtils::InitSphereAuxVariables::MINUS_NORMALS;
		init_aux_variables();
		for (auto&out : Outputs) {
			out.showSphereCenters = true;
			out.showSphereEdges = out.showNormEdges =
				out.showTriangleCenters = out.showFacesNorm = false;
		}
		for (OptimizationOutput& out : Outputs) 
		{
			for (auto& obj : out.totalObjective->objectiveList) 
			{
				std::shared_ptr<AuxSpherePerHinge> AS = std::dynamic_pointer_cast<AuxSpherePerHinge>(obj);
				std::shared_ptr<AuxBendingNormal> ABN = std::dynamic_pointer_cast<AuxBendingNormal>(obj);
				if (ABN != NULL)
					ABN->w = 0;
				if (AS != NULL)
					AS->w = 1.6;
			}
		}
	}
	
	if ((key == ' ') && modifiers == 1 && isModelLoaded)
		isMinimizerRunning ? stop_minimizer_thread() : start_minimizer_thread();
	
	return ImGuiMenu::key_pressed(key, modifiers);
}

IGL_INLINE bool deformation_plugin::key_down(int key, int modifiers)
{
	if (key == '1')
		UserInterface_option = app_utils::UserInterfaceOptions::FIX_VERTICES;
	else if (key == '2')
		UserInterface_option = app_utils::UserInterfaceOptions::BRUSH_WEIGHTS_INCR;
	else if (key == '3')
		UserInterface_option = app_utils::UserInterfaceOptions::BRUSH_WEIGHTS_DECR;
	else if (key == '4')
		UserInterface_option = app_utils::UserInterfaceOptions::ADJ_WEIGHTS;
	else if (key == '5')
		UserInterface_option = app_utils::UserInterfaceOptions::BRUSH_SIGMOID;
	else if (key == '6')
		UserInterface_option = app_utils::UserInterfaceOptions::ADJ_SIGMOID;
	else if (key == '7')
		UserInterface_option = app_utils::UserInterfaceOptions::FIX_FACES;
	
	return ImGuiMenu::key_down(key, modifiers);
}

IGL_INLINE bool deformation_plugin::key_up(int key, int modifiers)
{
	UserInterface_option = app_utils::UserInterfaceOptions::NONE;
	return ImGuiMenu::key_up(key, modifiers);
}

IGL_INLINE void deformation_plugin::shutdown()
{
	stop_minimizer_thread();
	ImGuiMenu::shutdown();
}

void deformation_plugin::draw_brush_sphere() 
{
	if (!(Brush_face_index != NOT_FOUND && Outputs[Brush_output_index].UserInterface_IsTranslate))
		return;
	//prepare color
	Eigen::MatrixXd c(1, 3);
	if (UserInterface_option == app_utils::UserInterfaceOptions::BRUSH_WEIGHTS_INCR) {
		if (UI_status == INSERT)
			c.row(0) = Outputs[0].Energy_auxBendingNormal->colorP.cast<double>();
		else if (UI_status == ERASE)
			c.row(0) = model_color.cast<double>();
	}
	else if (UserInterface_option == app_utils::UserInterfaceOptions::BRUSH_WEIGHTS_DECR) {
		if (UI_status == INSERT)
			c.row(0) = Outputs[0].Energy_auxBendingNormal->colorM.cast<double>();
		else if (UI_status == ERASE)
			c.row(0) = model_color.cast<double>();
	}
	else if (UserInterface_option == app_utils::UserInterfaceOptions::BRUSH_SIGMOID) {
		if (UI_status == INSERT)
			c.row(0) = Outputs[0].Energy_auxBendingNormal->colorP.cast<double>();
		else if (UI_status == ERASE)
			c.row(0) = model_color.cast<double>();
	}
	else return;
	
	//prepare brush sphere
	const int samples = 100;
	Eigen::MatrixXd sphere(samples * samples, 3);
	Eigen::RowVector3d center = intersec_point.cast<double>().transpose();
	int i, j;
	for (double alfa = 0, i = 0; alfa < 360; i++, alfa += int(360/samples)) 
	{
		for (double beta = 0, j = 0; beta < 360; j++, beta += int(360 / samples))
		{
			Eigen::RowVector3d dir;
			dir << sin(alfa), cos(alfa)*cos(beta), sin(beta)*cos(alfa);
			if (i + samples * j < sphere.rows())
				sphere.row(i + samples * j) = dir * brush_radius + center;
		}
	}
	
	//update data for cores
	OutputModel(Brush_output_index).add_points(sphere, c);
}

IGL_INLINE bool deformation_plugin::pre_draw() 
{
	follow_and_mark_selected_faces();
	Update_view();
	update_parameters_for_all_cores();
	for (auto& out : Outputs)
		if (out.minimizer->progressed)
			update_data_from_minimizer();
	//Update the model's faces colors in the screens
	InputModel().set_colors(model_color.cast <double>().replicate(1, InputModel().F.rows()).transpose());
	for (int i = 0; i < Outputs.size(); i++) {
		if (Outputs[i].color_per_face.size()) {
			OutputModel(i).set_colors(Outputs[i].color_per_face);
			if ((UserInterface_colorInputModelIndex - 1) == i)
				InputModel().set_colors(Outputs[i].color_per_face);
		}
	}

	//Update the model's vertex colors in screens
	if (isModelLoaded) {
		InputModel().point_size = 10;
		OptimizationOutput& IOm = Outputs[UserInterface_colorInputModelIndex - 1];
		InputModel().set_points(IOm.fixed_vertices_positions, IOm.color_per_vertex);
		for (int oi = 0; oi < Outputs.size(); oi++) {
			auto& m = OutputModel(oi);
			auto& o = Outputs[oi];
			auto& AS = Outputs[oi].Energy_auxSpherePerHinge;
			m.point_size = 10;
			m.set_points(o.fixed_vertices_positions, o.color_per_vertex);
			m.clear_edges();

			if (o.showFacesNorm)
				m.add_points(o.getFacesNorm(), o.color_per_face_norm);
			if (o.showTriangleCenters)
				m.add_points(o.getCenterOfFaces(), o.color_per_vertex_center);
			if (o.showSphereCenters)
				m.add_points(o.getCenterOfSphere(), o.color_per_sphere_center);
			if (o.showSphereEdges)
				m.add_edges(o.getCenterOfFaces(), o.getSphereEdges(), o.color_per_sphere_edge);
			if (o.showNormEdges)
				m.add_edges(o.getCenterOfFaces(), o.getFacesNorm(), o.color_per_norm_edge);
			
			// Update Vertices colors for UI sigmoid weights
			int num_hinges = AS->mesh_indices.num_hinges;
			const Eigen::VectorXi& x0_index = AS->x0_GlobInd;
			const Eigen::VectorXi& x1_index = AS->x1_GlobInd;
			double* hinge_val = AS->weight_PerHinge.host_arr;
			std::set<int> points_indices;
			for (int hi = 0; hi < num_hinges; hi++) {
				if (hinge_val[hi] < 1) {
					points_indices.insert(x0_index[hi]);
					points_indices.insert(x1_index[hi]);
				}
			}
			Eigen::MatrixXd points_pos(points_indices.size(), 3);
			auto& iter = points_indices.begin();
			for (int i = 0; i < points_pos.rows(); i++) {
				int v_index = *(iter++);
				points_pos.row(i) = m.V.row(v_index);
			}
			auto color = Outputs[oi].Energy_auxBendingNormal->colorM.cast<double>().replicate(1, points_indices.size()).transpose();
			m.add_points(points_pos, color);
		}
	}
	draw_brush_sphere();
	
	return ImGuiMenu::pre_draw();
}

void deformation_plugin::change_minimizer_type(Cuda::OptimizerType type)
{
	optimizer_type = type;
	stop_minimizer_thread();
	init_aux_variables();
	for (int i = 0; i < Outputs.size(); i++)
		Outputs[i].updateActiveMinimizer(optimizer_type);
}

void deformation_plugin::update_ext_fixed_vertices() 
{
	for (int i = 0; i < Outputs.size(); i++)
	{
		std::vector<int> CurrHandlesInd; CurrHandlesInd.clear();
		Eigen::MatrixX3d CurrHandlesPosDeformed;
		//First, we push each vertices index to the handles
		for (auto vi : Outputs[i].UserInterface_FixedVertices)
			CurrHandlesInd.push_back(vi);
		//Here we update the positions for each handle
		CurrHandlesPosDeformed = Eigen::MatrixX3d::Zero(CurrHandlesInd.size(), 3);
		int idx = 0;
		for (auto hi : CurrHandlesInd)
			CurrHandlesPosDeformed.row(idx++) <<
			OutputModel(i).V(hi, 0),
			OutputModel(i).V(hi, 1),
			OutputModel(i).V(hi, 2);
		set_vertices_for_mesh(OutputModel(i).V, i);
		//Finally, we update the handles in the constraints positional object
		if (isModelLoaded)
			Outputs[i].Energy_FixChosenVertices->updateExtConstraints(CurrHandlesInd, CurrHandlesPosDeformed);
	}
}

void deformation_plugin::Update_view() 
{
	for (auto& data : viewer->data_list)
		for (auto& out : Outputs)
			data.copy_options(viewer->core(inputCoreID), viewer->core(out.CoreID));
	for (auto& core : viewer->core_list)
		for (auto& data : viewer->data_list)
			viewer->data(data.id).set_visible(false, core.id);
	InputModel().set_visible(true, inputCoreID);
	for (int i = 0; i < Outputs.size(); i++)
		OutputModel(i).set_visible(true, Outputs[i].CoreID);
	for (auto& core : viewer->core_list)
		core.is_animating = true;
}

void deformation_plugin::follow_and_mark_selected_faces() 
{
	if (!isModelLoaded)
		return;
	for (int i = 0; i < Outputs.size(); i++) 
	{
		Outputs[i].initFaceColors(
			InputModel().F.rows(),
			center_sphere_color,
			center_vertex_color, 
			Color_sphere_edges, 
			Color_normal_edge, 
			face_norm_color);

		UpdateEnergyColors(i);
		//Mark the Groups faces
		for (FacesGroup cluster : Outputs[i].UserInterface_facesGroups)
			for (int fi : cluster.faces)
				Outputs[i].setFaceColors(fi, cluster.color);
		//Mark the fixed faces
		for (int fi : Outputs[i].UserInterface_FixedFaces)
			Outputs[i].setFaceColors(fi, Fixed_face_color);
		//Mark the selected faces by brush
		{
			std::vector<Eigen::Vector2d>& hinge_to_face_mapping = Outputs[i].Energy_auxSpherePerHinge->hinges_faceIndex;
			int num_hinges = Outputs[i].Energy_auxSpherePerHinge->mesh_indices.num_hinges;
			
			if (UserInterface_option == app_utils::UserInterfaceOptions::ADJ_SIGMOID || UserInterface_option == app_utils::UserInterfaceOptions::BRUSH_SIGMOID) {
				double* hinge_val = Outputs[i].Energy_auxBendingNormal->Sigmoid_PerHinge.host_arr;
				for (int hi = 0; hi < num_hinges; hi++) {
					const int f0 = hinge_to_face_mapping[hi][0];
					const int f1 = hinge_to_face_mapping[hi][1];
					const double log_minus_w = -log2(hinge_val[hi]);
					if (log_minus_w > 0) {
						const double alpha = log_minus_w / MAX_SIGMOID_PER_HINGE_VALUE;
						Outputs[i].shiftFaceColors(f0, alpha, model_color, Outputs[i].Energy_auxBendingNormal->colorP);
						Outputs[i].shiftFaceColors(f1, alpha, model_color, Outputs[i].Energy_auxBendingNormal->colorP);
					}
				}
			}
			else {
				double* hinge_val = Outputs[i].Energy_auxBendingNormal->weight_PerHinge.host_arr;
				for (int hi = 0; hi < num_hinges; hi++) {
					const int f0 = hinge_to_face_mapping[hi][0];
					const int f1 = hinge_to_face_mapping[hi][1];
					if (hinge_val[hi] > 1) {
						const double alpha = (hinge_val[hi] - 1.0f) / MAX_WEIGHT_PER_HINGE_VALUE;
						Outputs[i].shiftFaceColors(f0, alpha, model_color, Outputs[i].Energy_auxBendingNormal->colorP);
						Outputs[i].shiftFaceColors(f1, alpha, model_color, Outputs[i].Energy_auxBendingNormal->colorP);
					}
				}
			}
		}
		//Mark the highlighted face & neighbors
		if (curr_highlighted_face != NOT_FOUND && curr_highlighted_output == i)
		{
			std::vector<int> neigh = Outputs[i].getNeigh(neighbor_Type, InputModel().F, curr_highlighted_face, neighbor_distance);
			for (int fi : neigh)
				Outputs[i].setFaceColors(fi, Neighbors_Highlighted_face_color);
			Outputs[i].setFaceColors(curr_highlighted_face, Highlighted_face_color);
		}
		//Mark the Dragged face
		if (Outputs[i].UserInterface_IsTranslate && (UserInterface_option == app_utils::UserInterfaceOptions::FIX_FACES))
			Outputs[i].setFaceColors(Outputs[i].UserInterface_TranslateIndex, Dragged_face_color);
		//Mark the vertices
		int idx = 0;
		Outputs[i].fixed_vertices_positions.resize(Outputs[i].UserInterface_FixedVertices.size(), 3);
		Outputs[i].color_per_vertex.resize(Outputs[i].UserInterface_FixedVertices.size(), 3);
		//Mark the dragged vertex
		if (Outputs[i].UserInterface_IsTranslate && (UserInterface_option == app_utils::UserInterfaceOptions::FIX_VERTICES))
		{
			Outputs[i].fixed_vertices_positions.resize(Outputs[i].UserInterface_FixedVertices.size() + 1, 3);
			Outputs[i].color_per_vertex.resize(Outputs[i].UserInterface_FixedVertices.size() + 1, 3);
			Outputs[i].color_per_vertex.row(idx) = Dragged_vertex_color.cast<double>();
			Outputs[i].fixed_vertices_positions.row(idx) = OutputModel(i).V.row(Outputs[i].UserInterface_TranslateIndex);
			idx++;
		}
		//Mark the fixed vertices
		for (auto vi : Outputs[i].UserInterface_FixedVertices)
		{
			Outputs[i].fixed_vertices_positions.row(idx) = OutputModel(i).V.row(vi);
			Outputs[i].color_per_vertex.row(idx++) = Fixed_vertex_color.cast<double>();
		}

		//Mark the clusters if needed
		if (clustering_Type != app_utils::Clustering_Type::NO_CLUSTERING) {
			Eigen::MatrixXd P;
			if (clustering_Type == app_utils::Clustering_Type::NORMALS) {
				P = Outputs[i].getFacesNormals();
			}
			if (clustering_Type == app_utils::Clustering_Type::SPHERES) {
				Eigen::MatrixXd C = Outputs[i].getCenterOfSphere();
				Eigen::VectorXd R = Outputs[i].getRadiusOfSphere();
				P.resize(C.rows(), 3);
				for (int fi = 0; fi < C.rows(); fi++) {
					P(fi, 0) = C(fi, 0) * R(fi);
					P(fi, 1) = C(fi, 1);
					P(fi, 2) = C(fi, 2);
				}
			}

			Eigen::RowVector3d Pmin = P.row(0);
			Eigen::RowVector3d Pmax = P.row(0);
			for (int fi = 0; fi < P.rows(); fi++) {
				Pmin(0) = Pmin(0) < P(fi, 0) ? Pmin(0) : P(fi, 0);
				Pmin(1) = Pmin(1) < P(fi, 1) ? Pmin(1) : P(fi, 1);
				Pmin(2) = Pmin(2) < P(fi, 2) ? Pmin(2) : P(fi, 2);
				Pmax(0) = Pmax(0) > P(fi, 0) ? Pmax(0) : P(fi, 0);
				Pmax(1) = Pmax(1) > P(fi, 1) ? Pmax(1) : P(fi, 1);
				Pmax(2) = Pmax(2) > P(fi, 2) ? Pmax(2) : P(fi, 2);
			}

			Outputs[i].clustering_faces_colors.resize(P.rows(), 3);
			ColorsHashMap DataColors(Clustering_MinDistance, &ColorsHashMap_colors);
			for (int fi = 0; fi < P.rows(); fi++) {
				for (int xyz = 0; xyz < 3; xyz++) {
					P(fi, xyz) = P(fi, xyz) - Pmin(xyz);
					P(fi, xyz) = P(fi, xyz) / (Pmax(xyz) - Pmin(xyz));
				}
				Outputs[i].clustering_faces_colors.row(fi) = P.row(fi);
				if (clustering_hashMap)
					Outputs[i].clustering_faces_colors.row(fi) = DataColors.getColor(P.row(fi).transpose(), fi).transpose();
				Outputs[i].setFaceColors(fi, Eigen::Vector3f(
					clustering_w * Outputs[i].clustering_faces_colors(fi, 0) + (1 - clustering_w),
					clustering_w * Outputs[i].clustering_faces_colors(fi, 1) + (1 - clustering_w),
					clustering_w * Outputs[i].clustering_faces_colors(fi, 2) + (1 - clustering_w)
				));
			}
			Outputs[i].clustering_faces_indices = DataColors.face_index;
		}
	}
}
	
igl::opengl::ViewerData& deformation_plugin::InputModel() 
{
	return viewer->data(inputModelID);
}

igl::opengl::ViewerData& deformation_plugin::OutputModel(const int index) 
{
	return viewer->data(Outputs[index].ModelID);
}

igl::opengl::ViewerCore& deformation_plugin::InputCore()
{
	return viewer->core(inputCoreID);
}

igl::opengl::ViewerCore& deformation_plugin::OutputCore(const int index) 
{
	return viewer->core(Outputs[index].CoreID);
}



bool deformation_plugin::pick_face(
	int* output_index, 
	int* face_index,
	Eigen::Vector3f& intersec_point, 
	const bool update_fixfaces) 
{
	*face_index = pick_face_per_core(InputModel().V, InputModel().F, app_utils::View::SHOW_INPUT_SCREEN_ONLY, intersec_point);
	*output_index = INPUT_MODEL_SCREEN;
	for (int i = 0; i < Outputs.size(); i++) 
	{
		if (*face_index == NOT_FOUND)
		{
			*face_index = pick_face_per_core(OutputModel(i).V, OutputModel(i).F, app_utils::View::SHOW_OUTPUT_SCREEN_ONLY_0 + i, intersec_point);
			*output_index = i;
		}
	}
	if (update_fixfaces)
	{
		Output_Translate_ID = *output_index;
	}
	return (*face_index != NOT_FOUND);
}

int deformation_plugin::pick_face_per_core(
	Eigen::MatrixXd& V, 
	Eigen::MatrixXi& F, 
	int CoreIndex, 
	Eigen::Vector3f& intersec_point) 
{
	// Cast a ray in the view direction starting from the mouse position
	int CoreID;
	if (CoreIndex == app_utils::View::SHOW_INPUT_SCREEN_ONLY)
		CoreID = inputCoreID;
	else
		CoreID = Outputs[CoreIndex - app_utils::View::SHOW_OUTPUT_SCREEN_ONLY_0].CoreID;
	double x = viewer->current_mouse_x;
	double y = viewer->core(CoreID).viewport(3) - viewer->current_mouse_y;
	if (view == app_utils::View::VERTICAL) 
	{
		y = (viewer->core(inputCoreID).viewport(3) / core_size) - viewer->current_mouse_y;
	}
	Eigen::RowVector3d pt;
	Eigen::Matrix4f modelview = viewer->core(CoreID).view;
	int vi = NOT_FOUND;
	std::vector<igl::Hit> hits;
	igl::unproject_in_mesh(Eigen::Vector2f(x, y), viewer->core(CoreID).view,
		viewer->core(CoreID).proj, viewer->core(CoreID).viewport, V, F, pt, hits);
	Eigen::Vector3f s, dir;
	igl::unproject_ray(Eigen::Vector2f(x, y), viewer->core(CoreID).view,
		viewer->core(CoreID).proj, viewer->core(CoreID).viewport, s, dir);
	int fi = NOT_FOUND;
	if (hits.size() > 0) 
	{
		fi = hits[0].id;
		intersec_point = s + dir * hits[0].t;
	}
	return fi;
}

bool deformation_plugin::pick_vertex(
	int* output_index, 
	int* vertex_index, 
	const bool update)
{
	*vertex_index = pick_vertex_per_core(InputModel().V, InputModel().F, app_utils::View::SHOW_INPUT_SCREEN_ONLY);
	*output_index = INPUT_MODEL_SCREEN;
	for (int i = 0; i < Outputs.size(); i++) 
	{
		if (*vertex_index == NOT_FOUND)
		{
			*vertex_index = pick_vertex_per_core(OutputModel(i).V, OutputModel(i).F, app_utils::View::SHOW_OUTPUT_SCREEN_ONLY_0 + i);
			*output_index = i;
		}
	}
	if (update)
	{
		Output_Translate_ID = *output_index;
	}
	return (*vertex_index != NOT_FOUND);
}

int deformation_plugin::pick_vertex_per_core(
	Eigen::MatrixXd& V, 
	Eigen::MatrixXi& F, 
	int CoreIndex) 
{
	// Cast a ray in the view direction starting from the mouse position
	int CoreID;
	if (CoreIndex == app_utils::View::SHOW_INPUT_SCREEN_ONLY)
		CoreID = inputCoreID;
	else
		CoreID = Outputs[CoreIndex - app_utils::View::SHOW_OUTPUT_SCREEN_ONLY_0].CoreID;
	double x = viewer->current_mouse_x;
	double y = viewer->core(CoreID).viewport(3) - viewer->current_mouse_y;
	if (view == app_utils::View::VERTICAL) {
		y = (viewer->core(inputCoreID).viewport(3) / core_size) - viewer->current_mouse_y;
	}
	Eigen::Matrix<double, 3, 1, 0, 3, 1> pt;
	Eigen::Matrix4f modelview = viewer->core(CoreID).view;
	int vi = NOT_FOUND;
	std::vector<igl::Hit> hits;
	unproject_in_mesh(
		Eigen::Vector2f(x, y), 
		viewer->core(CoreID).view,
		viewer->core(CoreID).proj, 
		viewer->core(CoreID).viewport, 
		V, 
		F, 
		pt, 
		hits
	);
	if (hits.size() > 0) 
	{
		int fi = hits[0].id;
		Eigen::RowVector3d bc;
		bc << 1.0 - hits[0].u - hits[0].v, hits[0].u, hits[0].v;
		bc.maxCoeff(&vi);
		vi = F(fi, vi);
	}
	return vi;
}

void deformation_plugin::set_vertices_for_mesh(
	Eigen::MatrixXd& V_uv, 
	const int index) 
{
	Eigen::MatrixXd V_uv_3D(V_uv.rows(),3);
	if (V_uv.cols() == 2) 
	{
		V_uv_3D.leftCols(2) = V_uv.leftCols(2);
		V_uv_3D.rightCols(1).setZero();
	}
	else if (V_uv.cols() == 3) 
	{
		V_uv_3D = V_uv;
	}
	OutputModel(index).set_vertices(V_uv_3D);
	OutputModel(index).compute_normals();
}
	
void deformation_plugin::checkGradients()
{
	stop_minimizer_thread();
	for (auto& o: Outputs) 
	{
		if (!isModelLoaded) 
		{
			isMinimizerRunning = false;
			return;
		}
		Eigen::VectorXd testX = Eigen::VectorXd::Random(o.totalObjective->objectiveList[0]->grad.size);
		o.totalObjective->checkGradient(testX);
		for (auto const &objective : o.totalObjective->objectiveList)
			objective->checkGradient(testX);
	}
}

void deformation_plugin::update_data_from_minimizer()
{
	const unsigned int out_size = Outputs.size();
	std::vector<Eigen::MatrixXd> V(out_size), center(out_size), norm(out_size);
	std::vector<Eigen::VectorXd> radius(out_size);
	for (int i = 0; i < out_size; i++)
	{
		auto& out = Outputs[i];
		out.minimizer->get_data(V[i], center[i], radius[i], norm[i]);
		
		if (out.UserInterface_IsTranslate && UserInterface_option == app_utils::UserInterfaceOptions::FIX_VERTICES)
			V[i].row(out.UserInterface_TranslateIndex) = OutputModel(i).V.row(out.UserInterface_TranslateIndex);
		else if (out.UserInterface_IsTranslate && UserInterface_option == app_utils::UserInterfaceOptions::FIX_FACES && out.getCenterOfSphere().size())
			center[i].row(out.UserInterface_TranslateIndex) = out.getCenterOfSphere().row(out.UserInterface_TranslateIndex);
		
		out.setAuxVariables(V[i], InputModel().F, center[i], radius[i], norm[i]);
		set_vertices_for_mesh(V[i],i);
	}
}

void deformation_plugin::stop_minimizer_thread() 
{
	isMinimizerRunning = false;
	for (auto&o : Outputs) 
	{
		if (o.minimizer->is_running) 
		{
			o.minimizer->stop();
		}
		while (o.minimizer->is_running);
	}
}

void deformation_plugin::init_aux_variables() 
{
	stop_minimizer_thread();
	if (InitMinimizer_NeighLevel_From < 1)
		InitMinimizer_NeighLevel_From = 1;
	if (InitMinimizer_NeighLevel_From > InitMinimizer_NeighLevel_To)
		InitMinimizer_NeighLevel_To = InitMinimizer_NeighLevel_From;
	for (int i = 0; i < Outputs.size(); i++)
		Outputs[i].initMinimizers(
			OutputModel(i).V,
			OutputModel(i).F,
			initSphereAuxVariables,
			InitMinimizer_NeighLevel_From,
			InitMinimizer_NeighLevel_To,
			copy_index,
			paste_index,
			radius_length_minus_normal);
}

void deformation_plugin::run_one_minimizer_iter() 
{
	stop_minimizer_thread();
	for (auto& o : Outputs)
		o.minimizer->run_one_iteration();
}

void deformation_plugin::start_minimizer_thread() 
{
	stop_minimizer_thread();
	for (auto& o : Outputs)
	{
		minimizer_thread = std::thread(&Minimizer::run, o.minimizer.get());
		minimizer_thread.detach();
	}
	isMinimizerRunning = true;
}

void deformation_plugin::init_objective_functions(const int index)
{
	Eigen::MatrixXd V = OutputModel(index).V;
	Eigen::MatrixX3i F = OutputModel(index).F;
	stop_minimizer_thread();
	if (V.rows() == 0 || F.rows() == 0)
		return;
	// initialize the energy
	std::cout << console_color::yellow << "-------Energies, begin-------" << std::endl;
	std::shared_ptr <AuxBendingNormal> auxBendingNormal = std::make_unique<AuxBendingNormal>(V, F, Cuda::PenaltyFunction::SIGMOID);
	Outputs[index].Energy_auxBendingNormal = auxBendingNormal;
	std::shared_ptr <AuxSpherePerHinge> auxSpherePerHinge = std::make_unique<AuxSpherePerHinge>(V, F, Cuda::PenaltyFunction::SIGMOID);
	Outputs[index].Energy_auxSpherePerHinge = auxSpherePerHinge;
	std::shared_ptr <STVK> stvk = std::make_unique<STVK>(V, F);
	std::shared_ptr <SDenergy> sdenergy = std::make_unique<SDenergy>(V, F);
	std::shared_ptr <FixAllVertices> fixAllVertices = std::make_unique<FixAllVertices>(V, F);
	std::shared_ptr <fixRadius> FixRadius = std::make_unique<fixRadius>(V, F);
	std::shared_ptr <UniformSmoothness> uniformSmoothness = std::make_unique<UniformSmoothness>(V, F);
	
	//Add User Interface Energies
	auto fixChosenVertices = std::make_shared<FixChosenConstraints>(V, F);
	Outputs[index].Energy_FixChosenVertices = fixChosenVertices;

	//init total objective
	Outputs[index].totalObjective->init_mesh(V, F);
	Outputs[index].totalObjective->objectiveList.clear();
	auto add_obj = [&](std::shared_ptr< ObjectiveFunction> obj) 
	{
		Outputs[index].totalObjective->objectiveList.push_back(move(obj));
	};
	add_obj(auxSpherePerHinge);
	add_obj(auxBendingNormal);
	add_obj(stvk);
	add_obj(sdenergy);
	add_obj(fixAllVertices);
	add_obj(fixChosenVertices);
	add_obj(FixRadius);
	add_obj(uniformSmoothness);
	std::cout  << "-------Energies, end-------" << console_color::white << std::endl;
	init_aux_variables();
}

void deformation_plugin::UpdateEnergyColors(const int index) 
{
	int numF = OutputModel(index).F.rows();
	Eigen::VectorXd DistortionPerFace(numF);
	DistortionPerFace.setZero();
	if (faceColoring_type == 0) 
	{ // No colors
		DistortionPerFace.setZero();
	}
	else if (faceColoring_type == 1) 
	{ // total energy
		for (auto& obj: Outputs[index].totalObjective->objectiveList) 
		{
			// calculate the distortion over all the energies
			if ((obj->Efi.size() != 0) && (obj->w != 0))
				DistortionPerFace += obj->Efi * obj->w;
		}
	}
	else 
	{
		auto& obj = Outputs[index].totalObjective->objectiveList[faceColoring_type - 2];
		if ((obj->Efi.size() != 0) && (obj->w != 0))
			DistortionPerFace = obj->Efi * obj->w;
	}
	Eigen::VectorXd alpha_vec = DistortionPerFace / (Max_Distortion+1e-8);
	Eigen::VectorXd beta_vec = Eigen::VectorXd::Ones(numF) - alpha_vec;
	Eigen::MatrixXd alpha(numF, 3), beta(numF, 3);
	alpha = alpha_vec.replicate(1, 3);
	beta = beta_vec.replicate(1, 3);
	//calculate low distortion color matrix
	Eigen::MatrixXd LowDistCol = model_color.cast <double>().replicate(1, numF).transpose();
	//calculate high distortion color matrix
	Eigen::MatrixXd HighDistCol = Vertex_Energy_color.cast <double>().replicate(1, numF).transpose();
	Outputs[index].color_per_face = beta.cwiseProduct(LowDistCol) + alpha.cwiseProduct(HighDistCol);
}
