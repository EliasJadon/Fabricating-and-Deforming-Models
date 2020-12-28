#include "..//..//plugins/deformation_plugin.h"
#include <igl/file_dialog_open.h>
#include <GLFW/glfw3.h>

#define INPUT_MODEL_SCREEN -1
#define NOT_FOUND -1
#define INSERT false
#define ERASE true

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
	typeSphereAuxVar = OptimizationUtils::InitSphereAuxiliaryVariables::LEAST_SQUARE_SPHERE;
	isLoadNeeded = false;
	IsMouseDraggingAnyWindow = false;
	isMinimizerRunning = false;
	energies_window = results_window = outputs_window = true;
	tips_window = false;
	neighborType = app_utils::NeighborType::LOCAL_NORMALS;
	IsChoosingGroups = false;
	isModelLoaded = false;
	isUpdateAll = true;
	UserInterface_colorInputModelIndex = 0;
	clusteringType = app_utils::ClusteringType::NoClustering;
	clusteringMSE = 0.1;
	clusteringRatio = 0.5;
	faceColoring_type = 1;
	curr_highlighted_output = curr_highlighted_face = NOT_FOUND;
	minimizer_type = MinimizerType::ADAM_MINIMIZER;
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
	Color_sphere_edges = Color_normal_edge = Eigen::Vector3f(0 / 255.0f, 100 / 255.0f, 100 / 255.0f);;
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
	Outputs.push_back(OptimizationOutput(viewer, minimizer_type,linesearch_type));
	core_size = 1.0 / (Outputs.size() + 1.0);
	//maximize window
	glfwMaximizeWindow(viewer->window);
}

void deformation_plugin::load_new_model(const std::string modelpath) 
{
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
		initializeMinimizer(i);
	}
	if (isModelLoaded)
		add_output();
	viewer->core(inputCoreID).align_camera_center(InputModel().V, InputModel().F);
	for (int i = 0; i < Outputs.size(); i++)
		viewer->core(Outputs[i].CoreID).align_camera_center(OutputModel(i).V, OutputModel(i).F);
	isModelLoaded = true;
	//set rotation type to 3D mode
	viewer->core(inputCoreID).trackball_angle = Eigen::Quaternionf::Identity();
	viewer->core(inputCoreID).orthographic = false;
	viewer->core(inputCoreID).set_rotation_type(igl::opengl::ViewerCore::RotationType(1));
}

IGL_INLINE void deformation_plugin::draw_viewer_menu()
{
	Draw_tips_window();
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
		if (UserInterface_option == app_utils::UserInterfaceOptions::GROUPING_BY_ADJ)
			ImGui::Combo("Neighbor type", (int *)(&neighborType), "Curr Face\0Local Sphere\0Global Sphere\0Local Normals\0Global Normals\0\0");
		if (UserInterface_option == app_utils::UserInterfaceOptions::GROUPING_BY_ADJ)
			ImGui::DragFloat("Neighbors Distance", &neighbor_distance, 0.05f, 0.01f, 10000.0f);
		if (UserInterface_option == app_utils::UserInterfaceOptions::GROUPING_BY_BRUSH)
			ImGui::DragFloat("Brush Radius", &brush_radius, 0.05f, 0.01f, 10000.0f);
		if (UserInterface_option == app_utils::UserInterfaceOptions::GROUPING_BY_BRUSH ||
			UserInterface_option == app_utils::UserInterfaceOptions::GROUPING_BY_ADJ)
			ImGui::Combo("Group Color", (int *)(&UserInterface_groupNum), app_utils::build_groups_names_list(Outputs[0].UserInterface_facesGroups));
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
		bool AnyChange = false;
		if (ImGui::Combo("Type", (int *)(&clusteringType), "None\0Normals\0Spheres\0\0"))
			AnyChange = true;
		if (ImGui::DragFloat("Tolerance", &clusteringMSE, 0.001f, 0.001f, 100.0f))
			AnyChange = true;
		if (clusteringType == app_utils::ClusteringType::SphereClustering && 
			ImGui::DragFloat("Ratio [cent/rad]", &clusteringRatio, 0.001f, 0.0f, 1.0f))
			AnyChange = true;
		if(AnyChange)
		{
			if (clusteringType == app_utils::ClusteringType::NormalClustering)
			{
				for (auto& out : Outputs)
					out.clustering(clusteringRatio, clusteringMSE, true);
			}
			else if (clusteringType == app_utils::ClusteringType::SphereClustering)
			{
				for (auto& out : Outputs)
					out.clustering(clusteringRatio, clusteringMSE, false);
			}
			else
			{
				for (auto& out : Outputs)
					out.clusters_indices.clear();
			}
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
		if (ImGui::Combo("Minimizer type", (int *)(&minimizer_type), "Newton\0Gradient Descent\0Adam\0\0"))
			change_minimizer_type(minimizer_type);
		if (ImGui::Combo("init sphere var", (int *)(&typeSphereAuxVar), "Sphere Fit\0Mesh Center\0Minus Normal\0\0"))
			init_minimizer_thread();
		
		if (ImGui::Combo("line search", (int *)(&linesearch_type), "Gradient Norm\0Function Value\0Constant Step\0\0")) {
			for (auto& o : Outputs)
				o.minimizer->lineSearch_type = linesearch_type;
		}
		if (linesearch_type == OptimizationUtils::LineSearch::CONSTANT_STEP && ImGui::DragFloat("Step value", &constantStep_LineSearch, 0.0001f, 0.0f, 1.0f)) {
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
	if (ImGui::BeginTable("Lambda table", 8, ImGuiTableFlags_Resizable))
	{
		ImGui::TableSetupColumn("ID", ImGuiTableColumnFlags_WidthAlwaysAutoResize);
		ImGui::TableSetupColumn("On/Off", ImGuiTableColumnFlags_WidthAlwaysAutoResize);
		ImGui::TableSetupColumn("Start from iter", ImGuiTableColumnFlags_WidthAlwaysAutoResize);
		ImGui::TableSetupColumn("Stop at", ImGuiTableColumnFlags_WidthAlwaysAutoResize);
		ImGui::TableSetupColumn("number of iter per lambda reduction", ImGuiTableColumnFlags_WidthAlwaysAutoResize);
		ImGui::TableSetupColumn("Curr iter", ImGuiTableColumnFlags_WidthAlwaysAutoResize);
		ImGui::TableSetupColumn("Time per iter [ms]", ImGuiTableColumnFlags_WidthAlwaysAutoResize);
		ImGui::TableSetupColumn("Avg time [ms]", ImGuiTableColumnFlags_WidthAlwaysAutoResize);
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
					auto ABN = std::dynamic_pointer_cast<AuxBendingNormal>(obj);
					auto AS = std::dynamic_pointer_cast<AuxSpherePerHinge>(obj);
					if (obj->w) {
						if (ABN != NULL)
							ImGui::Combo("Function", (int*)(&(ABN->cuda_ABN->functionType)), "Quadratic\0Exponential\0Sigmoid\0\0");
						if (AS != NULL)
							ImGui::Combo("Function", (int*)(&(AS->cuda_ASH->functionType)), "Quadratic\0Exponential\0Sigmoid\0\0");

						if (ABN != NULL && ABN->cuda_ABN->functionType == FunctionType::SIGMOID) {
							ImGui::Text(("2^" + std::to_string(int(log2(ABN->cuda_ABN->planarParameter)))).c_str());
							ImGui::SameLine();
							if (ImGui::Button("*", ImVec2(ImGui::GetFrameHeight(), ImGui::GetFrameHeight())))
							{
								ABN->cuda_ABN->planarParameter = (ABN->cuda_ABN->planarParameter * 2) > 1 ? 1 : ABN->cuda_ABN->planarParameter * 2;
							}
							ImGui::SameLine();
							if (ImGui::Button("/", ImVec2(ImGui::GetFrameHeight(), ImGui::GetFrameHeight())))
							{
								ABN->cuda_ABN->planarParameter /= 2;
							}
							const double  f64_zero = 0, f64_max = 100000.0;
							ImGui::DragScalar("w1", ImGuiDataType_Double, &(ABN->cuda_ABN->w1), 0.05f, &f64_zero, &f64_max);
							ImGui::DragScalar("w2", ImGuiDataType_Double, &(ABN->cuda_ABN->w2), 0.05f, &f64_zero, &f64_max);
							ImGui::DragScalar("w3", ImGuiDataType_Double, &(ABN->cuda_ABN->w3), 0.05f, &f64_zero, &f64_max);
						}
						if (AS != NULL && AS->cuda_ASH->functionType == FunctionType::SIGMOID) {
							ImGui::Text(("2^" + std::to_string(int(log2(AS->cuda_ASH->planarParameter)))).c_str());
							ImGui::SameLine();
							if (ImGui::Button("*", ImVec2(ImGui::GetFrameHeight(), ImGui::GetFrameHeight())))
							{
								AS->cuda_ASH->planarParameter = (AS->cuda_ASH->planarParameter * 2) > 1 ? 1 : AS->cuda_ASH->planarParameter * 2;
							}
							ImGui::SameLine();
							if (ImGui::Button("/", ImVec2(ImGui::GetFrameHeight(), ImGui::GetFrameHeight())))
							{
								AS->cuda_ASH->planarParameter /= 2;
							}
							const double  f64_zero = 0, f64_max = 100000.0;
							ImGui::DragScalar("w0", ImGuiDataType_Double, &(AS->cuda_ASH->w1), 0.05f, &f64_zero, &f64_max);
							ImGui::DragScalar("w1", ImGuiDataType_Double, &(AS->cuda_ASH->w2), 0.05f, &f64_zero, &f64_max);
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

void deformation_plugin::Draw_tips_window()
{
	if (!tips_window)
		return;
	ImGui::SetNextWindowSize(tips_window_size);
	ImGui::SetNextWindowPos(tips_window_position);
	ImGui::Begin("Tips & shortcuts",
		NULL,
		ImGuiWindowFlags_NoTitleBar |
		ImGuiWindowFlags_NoResize |
		ImGuiWindowFlags_NoMove
	);

	ImGui::SetWindowFontScale(2);
	ImGui::Text("Hello :)");
	ImGui::Text("You have some useful Tips for using this interface in the folowing paraghraphs.");
	ImGui::Text("\nHow to start:");
	ImGui::Text("\t1. Choose .OFF or .OBJ 3D model by \"Load\" button (see shortcuts also)");
	ImGui::Text("\t2. Choose Physical={Planar, Spherical} mode (see shortcuts also):");
	ImGui::Text("\t\t2.1. Choose \"Neighbor type\" under \"user interface\" header");
	ImGui::Text("\t\t     (you need to ***hold*** '2' first on the keyboard)");
	ImGui::Text("\t\t\t\"Local Normals\"\t- for planar mode");
	ImGui::Text("\t\t\t\"Global Normals\"\t- for planar mode");
	ImGui::Text("\t\t\t\"Local Spheres\"\t-  for spherical mode");
	ImGui::Text("\t\t\t\"Global Spheres\"\t-  for spherical mode");
	ImGui::Text("\t\t\t\"Curr Face\"\t- for both");
	ImGui::Text("\t\t2.2. Show normals or spheres centers (optional)");
	ImGui::Text("\t\t2.2. update the weight for the suitable energy");
	ImGui::Text("\t\t     (you need to choose only one energy from  the first for in the table)");
	ImGui::Text("\t3. Optional - Update minimizer settings");
	ImGui::Text("\t   e.g. line-search, Newton\\adam\\Gradient descent, etc");
	ImGui::Text("\t4. Run the solver (see also shortcuts)");
	ImGui::Text("\t5. Optional - You can change lambda manually from the energies window ");
	ImGui::Text("\t   by \"*\" button or \"\\\" button");
	ImGui::Text("\t   Or change it automatically by the first table in \"Energies Window\"");
	ImGui::Text("\t6. Optional - You can add external energies (see user energy paraghraph).");
	ImGui::Text("\t7. Optional - Finally, You can cluster the final results");

	ImGui::Text("\n\nUser external energies:");
	ImGui::Text("Pay attention, Left-click for adding");
	ImGui::Text("               Right-click for removing");
	ImGui::Text("\t- Fix vertices - ***Hold*** '1' and then choose vertices by the mouse");
	ImGui::Text("\t- Choose Groups by Brush - ***Hold*** '2' and then choose vertices by the mouse");
	ImGui::Text("\t                           you can scroll to change the size of the brush!");
	ImGui::Text("\t- Choose Groups by Neighbors - ***Hold*** '3' and then choose vertices by the mouse");
	ImGui::Text("\t                               you can scroll to change the neighbor distance!");
	
	ImGui::Text("\t- Fix Faces - ***Hold*** '4' and then choose vertices by the mouse");


	ImGui::Text("\n\nShortcuts:");
	ImGui::Text("\t- Shift + space - Run the solver (which replace step 4 in the above list)");
	ImGui::Text("\t- Shift + C - clear all chosen vertices and faces");
	ImGui::Text("\t- Shift + A - load island.off model (which replace step 1 in the above list)");
	ImGui::Text("\t- Shift + S - load spot.off model (which replace step 1 in the above list)");
	ImGui::Text("\t- Shift + Q - set planar mode (which replace step 2 in the above list)");
	ImGui::Text("\t- Shift + W - set spherical mode (which replace step 2 in the above list)");

	ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.6f, 0.0f, 0.0f, 1.0f));
	if (ImGui::Button("Close"))
		tips_window = false;
	ImGui::PopStyleColor();
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
		
		/*if (clusteri!!!!!!ngType != app_utils::ClusteringType::NoClustering && out.clusters_indices.size())
		{
			Eigen::Vector3f _;
			int highlightedFi = pick_face(_);
			ImGui::TextColored(c, (std::string("Number of clusters:\t") + std::to_string(out.clusters_indices.size())).c_str());
			for (int ci = 0; ci < out.clusters_indices.size(); ci++)
			{
				if (std::find(out.clusters_indices[ci].begin(), out.clusters_indices[ci].end(), highlightedFi) != out.clusters_indices[ci].end())
				{
					ImGui::TextColored(c, (std::string("clusters:\t") + std::to_string(ci)).c_str());
				}
			}
		}
		if (IsChoosingGroups!!!!!!!) {
			double r = out.getRadiusOfSphere(curr_highlighted_face)
			ImGui::TextColored(c, std::to_string(r).c_str())
		}*/
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
		o.UserInterface_FixedFaces.clear();
		for (auto& c : o.UserInterface_facesGroups)
			c.faces.clear();
		o.UserInterface_FixedVertices.clear();
	}
	update_ext_fixed_vertices();
	update_ext_fixed_faces();
	update_ext_fixed_group_faces();
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
	Outputs.push_back(OptimizationOutput(viewer, minimizer_type,linesearch_type));
	viewer->load_mesh_from_file(modelPath.c_str());
	Outputs[Outputs.size() - 1].ModelID = viewer->data_list[Outputs.size()].id;
	initializeMinimizer(Outputs.size() - 1);
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
	tips_window_position = ImVec2(0.1 * w, 0.1 * h);
	tips_window_size = ImVec2(0.8 * w, 0.8 * h);
}

void deformation_plugin::brush_erase_or_insert() 
{
	if (pick_face(&Brush_output_index, &Brush_face_index, intersec_point))
	{
		std::vector<int> brush_faces = Outputs[Brush_output_index].FaceNeigh(intersec_point.cast<double>(), brush_radius);
		if (EraseOrInsert == INSERT) 
		{
			for (int fi : brush_faces)
			{
				if (UserInterface_UpdateAllOutputs)
					for (auto& out : Outputs)
						out.UserInterface_facesGroups[UserInterface_groupNum].faces.insert(fi);
				else
					Outputs[Brush_output_index].UserInterface_facesGroups[UserInterface_groupNum].faces.insert(fi);
			}
		}
		else
		{
			if (UserInterface_UpdateAllOutputs)
				for (auto& out : Outputs)
					for (FacesGroup& clusterI : out.UserInterface_facesGroups)
						for (int fi : brush_faces)
							clusterI.faces.erase(fi);
			else
				for (FacesGroup& clusterI : Outputs[Brush_output_index].UserInterface_facesGroups)
					for (int fi : brush_faces)
						clusterI.faces.erase(fi);
		}
		update_ext_fixed_group_faces();
	}
}

IGL_INLINE bool deformation_plugin::mouse_move(int mouse_x, int mouse_y)
{
	if (!isModelLoaded || IsMouseDraggingAnyWindow)
		return true;	
	if (clusteringType != app_utils::ClusteringType::NoClustering && cluster_index != NOT_FOUND)
	{
		Eigen::Vector3f _;
		int face_index, output_index;
		
		if (pick_face(&output_index, &face_index, _) && output_index != NOT_FOUND && Outputs[output_index].clusters_indices.size())
		{
			for (int ci = 0; ci < Outputs[output_index].clusters_indices.size(); ci++)
			{
				for (auto& it = Outputs[output_index].clusters_indices[ci].begin(); it != Outputs[output_index].clusters_indices[ci].end(); ++it)
				{
					if (face_index == *it)
					{
						//found
						if (cluster_index != ci && cluster_index != NOT_FOUND)
						{
							Outputs[output_index].clusters_indices[cluster_index].push_back(*it);
							Outputs[output_index].clusters_indices[ci].erase(it);
							break;
						}
					}
				}
			}
		}
		return true;
	}
	if (IsChoosingGroups && UserInterface_option == app_utils::UserInterfaceOptions::GROUPING_BY_ADJ)
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
			update_ext_fixed_faces();
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
		if (out.UserInterface_IsTranslate && UserInterface_option == app_utils::UserInterfaceOptions::GROUPING_BY_BRUSH)
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
		if (out.UserInterface_IsTranslate && UserInterface_option == app_utils::UserInterfaceOptions::GROUPING_BY_BRUSH)
		{
			brush_radius += delta_y * 0.05;
			brush_radius = std::max<float>(0.005, brush_radius);
			return true;
		}
	}
	if (IsChoosingGroups && UserInterface_option == app_utils::UserInterfaceOptions::GROUPING_BY_ADJ)
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
	cluster_index = NOT_FOUND;

	if (IsChoosingGroups) 
	{
		IsChoosingGroups = false;
		curr_highlighted_output = curr_highlighted_face = NOT_FOUND;
		Eigen::Vector3f _;
		int face_index, output_index;
		if (pick_face(&output_index, &face_index, _))
		{
			std::vector<int> neigh = Outputs[output_index].getNeigh(neighborType, InputModel().F, face_index, neighbor_distance);
			if (EraseOrInsert == ERASE)
			{
				if (UserInterface_UpdateAllOutputs)
					for (auto& out : Outputs)
						for (FacesGroup& clusterI : out.UserInterface_facesGroups)
							for (int currF : neigh)
								clusterI.faces.erase(currF);
				else
					for (FacesGroup& clusterI : Outputs[output_index].UserInterface_facesGroups)
						for (int currF : neigh)
							clusterI.faces.erase(currF);
			}
			else if (EraseOrInsert == INSERT)
			{
				if (UserInterface_UpdateAllOutputs)
					for (auto& out : Outputs)
						for (int currF : neigh)
							out.UserInterface_facesGroups[UserInterface_groupNum].faces.insert(currF);
				else
					for (int currF : neigh)
						Outputs[output_index].UserInterface_facesGroups[UserInterface_groupNum].faces.insert(currF);

			}
				
			update_ext_fixed_group_faces();
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
	
	if (clusteringType != app_utils::ClusteringType::NoClustering && button == GLFW_MOUSE_BUTTON_LEFT && modifier == 2)
	{
		Eigen::Vector3f _;
		int face_index,output_index;
		pick_face(&output_index, &face_index, _, false);
		if (Outputs[output_index].clusters_indices.size())
			for (int ci = 0; ci < Outputs[output_index].clusters_indices.size(); ci++)
				if (std::find(Outputs[output_index].clusters_indices[ci].begin(), Outputs[output_index].clusters_indices[ci].end(), face_index) != Outputs[output_index].clusters_indices[ci].end())
					cluster_index = ci;
	}
	else if (UserInterface_option == app_utils::UserInterfaceOptions::FIX_FACES && button == GLFW_MOUSE_BUTTON_LEFT)
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
			
			update_ext_fixed_faces();
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
			update_ext_fixed_faces();
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
					out.UserInterface_IsTranslate = true;
					out.UserInterface_TranslateIndex = vertex_index;
				}
			}
			else
			{
				Outputs[output_index].UserInterface_FixedVertices.insert(vertex_index);
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
	else if (UserInterface_option == app_utils::UserInterfaceOptions::GROUPING_BY_BRUSH && button == GLFW_MOUSE_BUTTON_LEFT)
	{
		if (pick_face(&Brush_output_index, &Brush_face_index, intersec_point))
		{
			EraseOrInsert = INSERT;
			Outputs[Brush_output_index].UserInterface_IsTranslate = true;
		}
	}
	else if (UserInterface_option == app_utils::UserInterfaceOptions::GROUPING_BY_BRUSH && button == GLFW_MOUSE_BUTTON_MIDDLE)
	{
		if (pick_face(&Brush_output_index, &Brush_face_index, intersec_point))
		{
			EraseOrInsert = ERASE;
			Outputs[Brush_output_index].UserInterface_IsTranslate = true;
		}
	}
	else if (UserInterface_option == app_utils::UserInterfaceOptions::GROUPING_BY_ADJ && button == GLFW_MOUSE_BUTTON_LEFT)
	{
		IsChoosingGroups = true;
		EraseOrInsert = INSERT;
		Eigen::Vector3f _;
		pick_face(&curr_highlighted_output, &curr_highlighted_face, _);
	}
	else if (UserInterface_option == app_utils::UserInterfaceOptions::GROUPING_BY_ADJ && button == GLFW_MOUSE_BUTTON_MIDDLE)
	{
		IsChoosingGroups = true;
		EraseOrInsert = ERASE;
		Eigen::Vector3f _;
		pick_face(&curr_highlighted_output, &curr_highlighted_face, _);
	}

	return false;
}

IGL_INLINE bool deformation_plugin::key_pressed(unsigned int key, int modifiers) 
{
	if ((key == 'c' || key == 'C') && modifiers == 1)
		clear_sellected_faces_and_vertices();
	if ((key == 'a' || key == 'A') && modifiers == 1) 
	{
		modelPath = OptimizationUtils::ProjectPath() + 
			"\\models\\InputModels\\from_2k_to_10k\\island.off";
		isLoadNeeded = true;
	}
	if ((key == 's' || key == 'S') && modifiers == 1) {
		modelPath = OptimizationUtils::ProjectPath() + 
			"\\models\\InputModels\\from_2k_to_10k\\spot.obj";
		isLoadNeeded = true;
	}
	if (isModelLoaded && (key == 'q' || key == 'Q') && modifiers == 1) 
	{
		neighborType = app_utils::NeighborType::LOCAL_NORMALS;
		for (auto&out : Outputs) {
			out.showFacesNorm = true;
			out.showSphereEdges = out.showNormEdges = out.showTriangleCenters = out.showSphereCenters = false;
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
		neighborType = app_utils::NeighborType::LOCAL_SPHERE;
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
	
	if ((key == ' ') && modifiers == 1)
		isMinimizerRunning ? stop_minimizer_thread() : start_minimizer_thread();
	
	return ImGuiMenu::key_pressed(key, modifiers);
}

IGL_INLINE bool deformation_plugin::key_down(int key, int modifiers)
{

	if (key == '1')
		UserInterface_option = app_utils::UserInterfaceOptions::FIX_VERTICES;
	if (key == '2')
		UserInterface_option = app_utils::UserInterfaceOptions::GROUPING_BY_ADJ;
	if (key == '3')
		UserInterface_option = app_utils::UserInterfaceOptions::GROUPING_BY_BRUSH;
	if (key == '4')
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
	if (!(Brush_face_index != NOT_FOUND && 
		Outputs[Brush_output_index].UserInterface_IsTranslate &&
		UserInterface_option == app_utils::UserInterfaceOptions::GROUPING_BY_BRUSH))
		return;
	//prepare brush sphere
	Eigen::MatrixXd sphere(36 * 36, 3);
	Eigen::RowVector3d center = intersec_point.cast<double>().transpose();
	int i, j;
	for (double alfa = 0, i = 0; alfa < 360; i++, alfa += 10) 
	{
		for (double beta = 0, j = 0; beta < 360; j++, beta += 10) 
		{
			Eigen::RowVector3d dir;
			dir << sin(alfa), cos(alfa)*cos(beta), sin(beta)*cos(alfa);
			sphere.row(i + 36 * j) = dir * brush_radius + center;
		}
	}
	//prepare color
	Eigen::MatrixXd c(1, 3);
	if (EraseOrInsert == INSERT) {
		c.row(0) = Outputs[Brush_output_index].UserInterface_facesGroups[UserInterface_groupNum].color.cast<double>();
	}
	else if (EraseOrInsert == ERASE) { 
		c.row(0) << 1, 1, 1; // white color for erasing
	}
	//update data for cores
	OutputModel(Brush_output_index).point_size = 10;
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
	InputModel().points.resize(0, 0);
	for (int i = 0; i < Outputs.size(); i++) {
		OutputModel(i).point_size = 10;
		OutputModel(i).set_points(Outputs[i].fixed_vertices_positions, Outputs[i].color_per_vertex);
		if ((UserInterface_colorInputModelIndex - 1) == i)
		{
			InputModel().point_size = 10;
			InputModel().set_points(Outputs[i].fixed_vertices_positions, Outputs[i].color_per_vertex);
		}
	}
	
	draw_brush_sphere();

	for (int oi = 0; oi < Outputs.size(); oi++) {
		OutputModel(oi).clear_edges();
		OutputModel(oi).point_size = 10;
			
		if (Outputs[oi].showFacesNorm && Outputs[oi].getFacesNorm().size() != 0)
			OutputModel(oi).add_points(Outputs[oi].getFacesNorm(), Outputs[oi].color_per_face_norm);
		if (Outputs[oi].showTriangleCenters && Outputs[oi].getCenterOfFaces().size() != 0)
			OutputModel(oi).add_points(Outputs[oi].getCenterOfFaces(), Outputs[oi].color_per_vertex_center);
		if (Outputs[oi].showSphereCenters && Outputs[oi].getCenterOfSphere().size() != 0)
			OutputModel(oi).add_points(Outputs[oi].getCenterOfSphere(), Outputs[oi].color_per_sphere_center);
		if (Outputs[oi].showSphereEdges && Outputs[oi].getCenterOfFaces().size() != 0)
			OutputModel(oi).add_edges(Outputs[oi].getCenterOfFaces(), Outputs[oi].getSphereEdges(), Outputs[oi].color_per_sphere_edge);
		if (Outputs[oi].showNormEdges && Outputs[oi].getCenterOfFaces().size() != 0)
			OutputModel(oi).add_edges(Outputs[oi].getCenterOfFaces(), Outputs[oi].getFacesNorm(), Outputs[oi].color_per_norm_edge);
	}
	return ImGuiMenu::pre_draw();
}

void deformation_plugin::change_minimizer_type(MinimizerType type) 
{
	minimizer_type = type;
	stop_minimizer_thread();
	init_minimizer_thread();
	for (int i = 0; i < Outputs.size(); i++)
		Outputs[i].updateActiveMinimizer(minimizer_type);
}

void deformation_plugin::update_ext_fixed_faces() 
{
	for (auto&out : Outputs) {
		std::vector<int> CurrFacesInd; CurrFacesInd.clear();
		Eigen::MatrixX3d CurrCentersPos, CurrFacesNormals;
		for (auto fi : out.UserInterface_FixedFaces)
			CurrFacesInd.push_back(fi);
		CurrCentersPos = Eigen::MatrixX3d::Zero(CurrFacesInd.size(), 3);
		CurrFacesNormals = Eigen::MatrixX3d::Zero(CurrFacesInd.size(), 3);
		int idx = 0;
		for (auto ci : CurrFacesInd)
		{
			if (out.getCenterOfSphere().size() != 0)
				CurrCentersPos.row(idx) = out.getCenterOfSphere().row(ci);
			if (out.getFacesNormals().size() != 0)
				CurrFacesNormals.row(idx) = out.getFacesNormals().row(ci);
			idx++;
		}
			
		//Finally, we update the handles in the constraints positional object
		if (isModelLoaded) {
			out.Energy_FixChosenNormals->updateExtConstraints(CurrFacesInd, CurrFacesNormals);
			out.Energy_FixChosenSpheres->updateExtConstraints(CurrFacesInd, CurrCentersPos);
		}
	}
}

void deformation_plugin::update_ext_fixed_group_faces() 
{
	for (auto&out : Outputs)
	{
		std::vector < std::vector<int>> ind(out.UserInterface_facesGroups.size());
		for (int ci = 0; ci < out.UserInterface_facesGroups.size(); ci++)
			for (int fi : out.UserInterface_facesGroups[ci].faces)
				ind[ci].push_back(fi);
		if (isModelLoaded)
		{
			out.Energy_GroupNormals->updateExtConstraints(ind);
			out.Energy_GroupSpheres->updateExtConstraints(ind);
		}
	}
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
	if (!InputModel().F.size())
		return;
	for (int i = 0; i < Outputs.size(); i++) 
	{
		Outputs[i].initFaceColors(InputModel().F.rows(),center_sphere_color,center_vertex_color, Color_sphere_edges, Color_normal_edge, face_norm_color);
		UpdateEnergyColors(i);
		//Mark the Groups faces
		for (FacesGroup cluster : Outputs[i].UserInterface_facesGroups)
			for (int fi : cluster.faces)
				Outputs[i].updateFaceColors(fi, cluster.color);
		//Mark the fixed faces
		for (int fi : Outputs[i].UserInterface_FixedFaces)
			Outputs[i].updateFaceColors(fi, Fixed_face_color);
		//Mark the highlighted face & neighbors
		if (curr_highlighted_face != NOT_FOUND && curr_highlighted_output == i)
		{
			std::vector<int> neigh = Outputs[i].getNeigh(neighborType,InputModel().F, curr_highlighted_face, neighbor_distance);
			for (int fi : neigh)
				Outputs[i].updateFaceColors(fi, Neighbors_Highlighted_face_color);
			Outputs[i].updateFaceColors(curr_highlighted_face, Highlighted_face_color);
		}
		//Mark the Dragged face
		if (Outputs[i].UserInterface_IsTranslate && (UserInterface_option == app_utils::UserInterfaceOptions::FIX_FACES))
			Outputs[i].updateFaceColors(Outputs[i].UserInterface_TranslateIndex, Dragged_face_color);
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

		if (clusteringType != app_utils::ClusteringType::NoClustering && 
			Outputs[i].clusters_indices.size()) 
		{
			UniqueColors uniqueColors;
			for(std::vector<int> clus: Outputs[i].clusters_indices)
			{
				Eigen::Vector3f clusColor = uniqueColors.getNext();
				for (int fi : clus)
				{
					Outputs[i].updateFaceColors(fi, clusColor);
				}
			}
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
		Eigen::VectorXd testX = Eigen::VectorXd::Random(InputModel().V.size() + 7*InputModel().F.rows());
		o.totalObjective->checkGradient(testX);
		for (auto const &objective : o.totalObjective->objectiveList)
			objective->checkGradient(testX);
	}
}

void deformation_plugin::update_data_from_minimizer()
{
	std::vector<Eigen::MatrixXd> V(Outputs.size()),center(Outputs.size()),norm(Outputs.size());
	std::vector<Eigen::VectorXd> radius(Outputs.size());
	for (int i = 0; i < Outputs.size(); i++)
	{
		Outputs[i].minimizer->get_data(V[i], center[i],radius[i],norm[i]);
		if (Outputs[i].UserInterface_IsTranslate && UserInterface_option == app_utils::UserInterfaceOptions::FIX_VERTICES)
			V[i].row(Outputs[i].UserInterface_TranslateIndex) = OutputModel(i).V.row(Outputs[i].UserInterface_TranslateIndex);
		else if (Outputs[i].UserInterface_IsTranslate && UserInterface_option == app_utils::UserInterfaceOptions::FIX_FACES && Outputs[i].getCenterOfSphere().size())
			center[i].row(Outputs[i].UserInterface_TranslateIndex) = Outputs[i].getCenterOfSphere().row(Outputs[i].UserInterface_TranslateIndex);
		Outputs[i].setAuxVariables(V[i], InputModel().F, center[i],radius[i],norm[i]);
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

void deformation_plugin::init_minimizer_thread() 
{
	stop_minimizer_thread();
	for (int i = 0; i < Outputs.size(); i++)
		Outputs[i].initMinimizers(OutputModel(i).V, OutputModel(i).F, typeSphereAuxVar);
}

void deformation_plugin::run_one_minimizer_iter() 
{
	stop_minimizer_thread();
	static int iteration_counter = 0;
	static int lambda_counter = 0;
	if(iteration_counter == 0)
		init_minimizer_thread();
	for (int i = 0; i < Outputs.size(); i++) 
	{
		minimizer_thread = std::thread(&Minimizer::run_one_iteration, Outputs[i].minimizer.get(), iteration_counter++, &lambda_counter, true);
		minimizer_thread.join();
	}
}

void deformation_plugin::start_minimizer_thread() 
{
	stop_minimizer_thread();
	init_minimizer_thread();
	for (int i = 0; i < Outputs.size();i++) 
	{
		minimizer_thread = std::thread(&Minimizer::run, Outputs[i].minimizer.get());
		minimizer_thread.detach();
	}
	isMinimizerRunning = true;
}

void deformation_plugin::initializeMinimizer(const int index)
{
	Eigen::MatrixXd V = OutputModel(index).V;
	Eigen::MatrixX3i F = OutputModel(index).F;
	stop_minimizer_thread();
	if (V.rows() == 0 || F.rows() == 0)
		return;
	// initialize the energy
	std::cout << console_color::yellow << "-------Energies, begin-------" << std::endl;
	std::shared_ptr <AuxBendingNormal> auxBendingNormal = std::make_unique<AuxBendingNormal>(V, F, FunctionType::SIGMOID);
	std::shared_ptr <AuxSpherePerHinge> auxSpherePerHinge = std::make_unique<AuxSpherePerHinge>(V, F, FunctionType::SIGMOID);
	std::shared_ptr <STVK> stvk = std::make_unique<STVK>(V, F);
	std::shared_ptr <FixAllVertices> fixAllVertices = std::make_unique<FixAllVertices>(V, F);
	
	//Add User Interface Energies
	auto fixChosenNormals = std::make_shared<FixChosenConstraints>(F.rows(), V.rows(), ConstraintsType::NORMALS);
	Outputs[index].Energy_FixChosenNormals = fixChosenNormals;
	
	auto fixChosenVertices = std::make_shared<FixChosenConstraints>(F.rows(), V.rows(), ConstraintsType::VERTICES);
	Outputs[index].Energy_FixChosenVertices = fixChosenVertices;

	auto fixChosenSpheres = std::make_shared<FixChosenConstraints>(F.rows(), V.rows(), ConstraintsType::SPHERES);
	Outputs[index].Energy_FixChosenSpheres = fixChosenSpheres;
	
	std::shared_ptr< Grouping> groupSpheres = std::make_shared<Grouping>(V, F, ConstraintsType::SPHERES);
	Outputs[index].Energy_GroupSpheres = groupSpheres;

	std::shared_ptr< Grouping> groupNormals = std::make_shared<Grouping>(V, F, ConstraintsType::NORMALS);
	Outputs[index].Energy_GroupNormals = groupNormals;

	//init total objective
	Outputs[index].totalObjective->objectiveList.clear();
	auto add_obj = [&](std::shared_ptr< ObjectiveFunction> obj) 
	{
		Outputs[index].totalObjective->objectiveList.push_back(move(obj));
	};
	add_obj(auxSpherePerHinge);
	add_obj(auxBendingNormal);
	add_obj(stvk);
	add_obj(fixAllVertices);
	add_obj(fixChosenVertices);
	add_obj(fixChosenNormals);
	add_obj(fixChosenSpheres);
	add_obj(groupSpheres);
	add_obj(groupNormals);
	std::cout  << "-------Energies, end-------" << console_color::white << std::endl;
	init_minimizer_thread();
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
