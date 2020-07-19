#include "plugins/deformation_plugin/include/deformation_plugin.h"
#include <igl/file_dialog_open.h>
#include <GLFW/glfw3.h>

deformation_plugin::deformation_plugin() :
	igl::opengl::glfw::imgui::ImGuiMenu(){}

IGL_INLINE void deformation_plugin::init(igl::opengl::glfw::Viewer *_viewer)
{
	ImGuiMenu::init(_viewer);
	if (_viewer)
	{
		brush_radius = 0.3;
		showSphereEdges = showNormEdges = showTriangleCenters = showSphereCenters = false;
		showFacesNorm = false;
		typeAuxVar = OptimizationUtils::InitAuxVariables::SPHERE;
		isLoadNeeded = false;
		IsMouseDraggingAnyWindow = false;
		IsMouseHoveringAnyWindow = false;
		isMinimizerRunning = false;
		Outputs_Settings = false;
		highlightFacesType = app_utils::HighlightFaces::NO_HIGHLIGHT;
		IsTranslate = false;
		IsChoosingCluster = false;
		isModelLoaded = false;
		isUpdateAll = true;
		minimizer_settings = true;
		show_text = true;
		runOneIteration = false;
		faceColoring_type = 1;
		curr_highlighted_face = -1;
		minimizer_type = app_utils::MinimizerType::NEWTON;
		linesearch_type = OptimizationUtils::LineSearch::FUNCTION_VALUE;
		mouse_mode = app_utils::MouseMode::FACE_CLUSTERING_0;
		view = app_utils::View::HORIZONTAL;

		Max_Distortion = 5;
		texture_scaling_input = texture_scaling_output = 1;
		down_mouse_x = down_mouse_y = -1;

		Vertex_Energy_color = RED_COLOR;
		Highlighted_face_color = Eigen::Vector3f(153 / 255.0f, 0, 153 / 255.0f);
		Neighbors_Highlighted_face_color = Eigen::Vector3f(1, 102 / 255.0f, 1);
		center_sphere_color = Eigen::Vector3f(0, 1, 1);
		center_vertex_color = Eigen::Vector3f(128 / 255.0f, 128 / 255.0f, 128 / 255.0f);
		centers_sphere_edge_color = centers_norm_edge_color = Eigen::Vector3f(0 / 255.0f, 100 / 255.0f, 100 / 255.0f);;
		face_norm_color = Eigen::Vector3f(0, 1, 1);
		Fixed_vertex_color = Fixed_face_color = BLUE_COLOR;
		Dragged_vertex_color = Dragged_face_color = GREEN_COLOR;
		model_color = GREY_COLOR;
		text_color = BLACK_COLOR;

		for (int i = 0; i < 5; i++)
			faceClusters.push_back(FaceClusters(faceClusters.size()));
		
		//update input viewer
		inputCoreID = viewer->core_list[0].id;
		viewer->core(inputCoreID).background_color = Eigen::Vector4f(0.9, 0.9, 0.9, 0);
		viewer->core(inputCoreID).is_animating = true;
		viewer->core(inputCoreID).lighting_factor = 0.5;

		//Load multiple views
		Outputs.push_back(OptimizationOutput(viewer, minimizer_type,linesearch_type));
		core_size = 1.0 / (Outputs.size() + 1.0);
		
		//maximize window
		glfwMaximizeWindow(viewer->window);
	}
}

void deformation_plugin::load_new_model(const std::string modelpath) {
	clear_sellected_faces_and_vertices();
	modelPath = modelpath;
	if (modelPath.length() != 0)
	{
		modelName = app_utils::ExtractModelName(modelPath);
		stop_minimizer_thread();
		if (isModelLoaded) {
			//remove previous data
			while (Outputs.size() > 0)
				remove_output(0);
			viewer->load_mesh_from_file(modelPath.c_str());
			viewer->erase_mesh(0);
		}
		else viewer->load_mesh_from_file(modelPath.c_str());
		inputModelID = viewer->data_list[0].id;
		for (int i = 0; i < Outputs.size(); i++){
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
}

IGL_INLINE void deformation_plugin::draw_viewer_menu()
{
	float w = ImGui::GetContentRegionAvailWidth();
	float p = ImGui::GetStyle().FramePadding.x;
	if (ImGui::Button("Load##Mesh", ImVec2((w - p) / 2.f, 0))){
		modelPath = igl::file_dialog_open();
		isLoadNeeded = true;
	}
	if (isLoadNeeded) {
		load_new_model(modelPath);
		isLoadNeeded = false;
	}
	ImGui::SameLine(0, p);
	if (ImGui::Button("Save##Mesh", ImVec2((w - p) / 2.f, 0)))
		viewer->open_dialog_save_mesh();
	if (ImGui::Checkbox("Outputs settings", &Outputs_Settings))
		if(Outputs_Settings)
			show_text = !Outputs_Settings;
	if (ImGui::Checkbox("Show text", &show_text))
		if(show_text)
			Outputs_Settings = !show_text;

	if (ImGui::Combo("Highlight type", (int *)(&highlightFacesType), "No Highlight\0Hovered Face\0Local Sphere\0Global Sphere\0Local Normals\0Global Normals\0\0")) {
		if(highlightFacesType == app_utils::HighlightFaces::GLOBAL_NORMALS ||
			highlightFacesType == app_utils::HighlightFaces::LOCAL_NORMALS)
			neighbor_distance = 0.03;
		if(highlightFacesType == app_utils::HighlightFaces::GLOBAL_SPHERE ||
			highlightFacesType == app_utils::HighlightFaces::LOCAL_SPHERE)
			neighbor_distance = 0.3;
	}
	if (highlightFacesType == app_utils::HighlightFaces::GLOBAL_NORMALS ||
		highlightFacesType == app_utils::HighlightFaces::LOCAL_NORMALS ||
		highlightFacesType == app_utils::HighlightFaces::GLOBAL_SPHERE ||
		highlightFacesType == app_utils::HighlightFaces::LOCAL_SPHERE)
		ImGui::DragFloat("Neighbors Distance", &neighbor_distance, 0.05f, 0.01f, 10000.0f);

	
	if (ImGui::Combo("View cores", (int *)(&view), app_utils::build_view_names_list(Outputs.size()))) {
		// That's how you get the current width/height of the frame buffer (for example, after the window was resized)
		int frameBufferWidth, frameBufferHeight;
		glfwGetFramebufferSize(viewer->window, &frameBufferWidth, &frameBufferHeight);
		post_resize(frameBufferWidth, frameBufferHeight);
	}

	if ((view == app_utils::View::HORIZONTAL) || (view == app_utils::View::VERTICAL)) {
		if (ImGui::SliderFloat("Core Size", &core_size, 0, 1.0 / Outputs.size(), std::to_string(core_size).c_str(), 1)) {
			int frameBufferWidth, frameBufferHeight;
			glfwGetFramebufferSize(viewer->window, &frameBufferWidth, &frameBufferHeight);
			post_resize(frameBufferWidth, frameBufferHeight);
		}
	}
	
	ImGui::Combo("Mouse Mode", (int *)(&mouse_mode), app_utils::build_clusters_names_list(faceClusters.size()));
	if (mouse_mode == app_utils::MouseMode::CLEAR) {
		clear_sellected_faces_and_vertices();
		mouse_mode = app_utils::MouseMode::CLEAR;
	}
	if(mouse_mode >= app_utils::MouseMode::FACE_CLUSTERING_0)
		ImGui::DragFloat("Brush Radius", &brush_radius, 0.05f, 0.01f, 10000.0f);

	if (ImGui::Button("Add Cluster"))
		faceClusters.push_back(FaceClusters(faceClusters.size()));

	ImGui::Checkbox("Update all cores together", &isUpdateAll);
	if(isModelLoaded)
		Draw_menu_for_Minimizer();
	Draw_menu_for_cores(viewer->core(inputCoreID));
	Draw_menu_for_models(viewer->data(inputModelID));
	Draw_menu_for_output_settings();
	Draw_menu_for_text_results();
	if (isModelLoaded && minimizer_settings)
		Draw_menu_for_minimizer_settings();
	follow_and_mark_selected_faces();
	Update_view();
	if (isUpdateAll)
		update_parameters_for_all_cores();
	IsMouseHoveringAnyWindow = false;
	if (ImGui::IsAnyWindowHovered() |
		ImGui::IsWindowHovered(ImGuiHoveredFlags_RootAndChildWindows) |
		ImGui::IsItemHovered(ImGuiHoveredFlags_RectOnly) |
		ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow) |
		ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByPopup | ImGuiHoveredFlags_AllowWhenBlockedByActiveItem))
		IsMouseHoveringAnyWindow = true;
}

void deformation_plugin::clear_sellected_faces_and_vertices() {
	selected_fixed_faces.clear();
	for (auto& c : faceClusters)
		c.faces.clear();
	selected_vertices.clear();
	UpdateVerticesHandles();
	UpdateCentersHandles();
	UpdateClustersHandles();
}

void deformation_plugin::update_parameters_for_all_cores() {
	for (auto& core : viewer->core_list) {
		int output_index = -1;
		for (int i = 0; i < Outputs.size(); i++)
			if (core.id == Outputs[i].CoreID)
				output_index = i;
		if (output_index == -1) {
			if (this->prev_camera_zoom != core.camera_zoom ||
				this->prev_camera_translation != core.camera_translation ||
				this->prev_trackball_angle.coeffs() != core.trackball_angle.coeffs()
				) {
				for (auto& c : viewer->core_list) {
					c.camera_zoom = core.camera_zoom;
					c.camera_translation = core.camera_translation;
					c.trackball_angle = core.trackball_angle;
				}	
				this->prev_camera_zoom = core.camera_zoom;
				this->prev_camera_translation = core.camera_translation;
				this->prev_trackball_angle = core.trackball_angle;
				for (auto&o : Outputs){
					o.prev_camera_zoom = core.camera_zoom;
					o.prev_camera_translation = core.camera_translation;
					o.prev_trackball_angle = core.trackball_angle;
				}
			}
		}
		else {
			if (Outputs[output_index].prev_camera_zoom != core.camera_zoom ||
				Outputs[output_index].prev_camera_translation != core.camera_translation ||
				Outputs[output_index].prev_trackball_angle.coeffs() != core.trackball_angle.coeffs()
				) {
				for (auto& c : viewer->core_list) {
					c.camera_zoom = core.camera_zoom;
					c.camera_translation = core.camera_translation;
					c.trackball_angle = core.trackball_angle;
				}	
				this->prev_camera_zoom = core.camera_zoom;
				this->prev_camera_translation = core.camera_translation;
				this->prev_trackball_angle = core.trackball_angle;
				for (auto&o : Outputs) {
					o.prev_camera_zoom = core.camera_zoom;
					o.prev_camera_translation = core.camera_translation;
					o.prev_trackball_angle = core.trackball_angle;
				}	
			}
		}
	}
}

void deformation_plugin::remove_output(const int output_index) {
	stop_minimizer_thread();
	viewer->erase_core(1 + output_index);
	viewer->erase_mesh(1 + output_index);
	Outputs.erase(Outputs.begin() + output_index);
	//Update the scene
	viewer->core(inputCoreID).align_camera_center(InputModel().V, InputModel().F);
	for (int i = 0; i < Outputs.size(); i++)
		viewer->core(Outputs[i].CoreID).align_camera_center(OutputModel(i).V, OutputModel(i).F);
	core_size = 1.0 / (Outputs.size() + 1.0);
	int frameBufferWidth, frameBufferHeight;
	glfwGetFramebufferSize(viewer->window, &frameBufferWidth, &frameBufferHeight);
	post_resize(frameBufferWidth, frameBufferHeight);
}

void deformation_plugin::add_output() {
	stop_minimizer_thread();
	Outputs.push_back(OptimizationOutput(viewer, minimizer_type,linesearch_type));
	viewer->load_mesh_from_file(modelPath.c_str());
	Outputs[Outputs.size() - 1].ModelID = viewer->data_list[Outputs.size()].id;
	initializeMinimizer(Outputs.size() - 1);
	//Update the scene
	viewer->core(inputCoreID).align_camera_center(InputModel().V, InputModel().F);
	for (int i = 0; i < Outputs.size(); i++)
		viewer->core(Outputs[i].CoreID).align_camera_center(OutputModel(i).V, OutputModel(i).F);
	core_size = 1.0 / (Outputs.size() + 1.0);
	int frameBufferWidth, frameBufferHeight;
	glfwGetFramebufferSize(viewer->window, &frameBufferWidth, &frameBufferHeight);
	post_resize(frameBufferWidth, frameBufferHeight);
}

IGL_INLINE void deformation_plugin::post_resize(int w, int h)
{
	if (viewer)
	{
		if (view == app_utils::View::HORIZONTAL) {
			viewer->core(inputCoreID).viewport = Eigen::Vector4f(0, 0, w - w * Outputs.size() * core_size, h);
			for (int i = 0; i < Outputs.size(); i++) {
				Outputs[i].window_position = ImVec2(w - w * (Outputs.size() - i) * core_size, 0);
				Outputs[i].window_size = ImVec2(w * core_size, h);
				Outputs[i].text_position = Outputs[i].window_position;
			}
		}
		if (view == app_utils::View::VERTICAL) {
			viewer->core(inputCoreID).viewport = Eigen::Vector4f(0, Outputs.size() * h * core_size, w, h - Outputs.size() * h * core_size);
			for (int i = 0; i < Outputs.size(); i++) {
				Outputs[i].window_position = ImVec2(0, (Outputs.size() - i - 1) * h * core_size);
				Outputs[i].window_size = ImVec2(w, h * core_size);
				Outputs[i].text_position = ImVec2(w*0.8, h - Outputs[i].window_position[1] - Outputs[i].window_size[1]);
			}
		}
		if (view == app_utils::View::INPUT_ONLY) {
			viewer->core(inputCoreID).viewport = Eigen::Vector4f(0, 0, w, h);
			for (auto&o : Outputs) {
				o.window_position = ImVec2(w, h);
				o.window_size = ImVec2(0, 0);
				o.text_position = o.window_position;
			}
		}
 		if (view >= app_utils::View::OUTPUT_ONLY_0) {
 			viewer->core(inputCoreID).viewport = Eigen::Vector4f(0, 0, 0, 0);
 			for (auto&o : Outputs) {
 				o.window_position = ImVec2(w, h);
 				o.window_size = ImVec2(0, 0);
 				o.text_position = o.window_position;
 			}
 			// what does this means?
 			Outputs[view - app_utils::View::OUTPUT_ONLY_0].window_position = ImVec2(0, 0);
 			Outputs[view - app_utils::View::OUTPUT_ONLY_0].window_size = ImVec2(w, h);
 			Outputs[view - app_utils::View::OUTPUT_ONLY_0].text_position = ImVec2(w*0.8, 0);
 		}		
		for (auto& o : Outputs)
			viewer->core(o.CoreID).viewport = Eigen::Vector4f(o.window_position[0], o.window_position[1], o.window_size[0]+1, o.window_size[1]+1);
	}
}

void deformation_plugin::brush_erase_or_insert() {
	int f = pick_face(intersec_point);
	brush_index = f;
	if (f != -1) {
		std::vector<int> brush_faces = Outputs[0].FaceNeigh(intersec_point.cast<double>(), brush_radius);
		if (EraseOrInsert == false) { //insert
			int clusterIndex = mouse_mode - app_utils::MouseMode::FACE_CLUSTERING_0;
			for (int fi : brush_faces)
				if (!(find(faceClusters[clusterIndex].faces.begin(), faceClusters[clusterIndex].faces.end(), fi) != faceClusters[clusterIndex].faces.end()))
					faceClusters[clusterIndex].faces.insert(fi);
		}
		else if (EraseOrInsert == true) { //erase
			for (FaceClusters& clusterI : faceClusters)
				for (int fi : brush_faces)
					clusterI.faces.erase(fi);
		}
		UpdateClustersHandles();
	}
}

IGL_INLINE bool deformation_plugin::mouse_move(int mouse_x, int mouse_y)
{
	if (IsMouseHoveringAnyWindow | IsMouseDraggingAnyWindow)
		return true;
	
	if (IsTranslate && mouse_mode == app_utils::MouseMode::FIX_FACES)
	{
		if (!selected_fixed_faces.empty())
		{
			Eigen::RowVector3d face_avg_pt = app_utils::get_face_avg(viewer, Model_Translate_ID, Translate_Index);
			Eigen::Vector3f translation = app_utils::computeTranslation(mouse_x, down_mouse_x, mouse_y, down_mouse_y, face_avg_pt, viewer->core(Core_Translate_ID));
			for (auto& out : Outputs) {
				out.translateCenterOfSphere(Translate_Index, translation.cast<double>());
			}
			down_mouse_x = mouse_x;
			down_mouse_y = mouse_y;
			UpdateCentersHandles();
			return true;
		}
	}
	else if (IsTranslate && mouse_mode == app_utils::MouseMode::FIX_VERTEX)
	{
		if (!selected_vertices.empty())
		{
			Eigen::RowVector3d vertex_pos = viewer->data(Model_Translate_ID).V.row(Translate_Index);
			Eigen::Vector3f translation = app_utils::computeTranslation(mouse_x, down_mouse_x, mouse_y, down_mouse_y, vertex_pos, viewer->core(Core_Translate_ID));
			if (Core_Translate_ID == inputCoreID) {
				viewer->data(Model_Translate_ID).V.row(Translate_Index) += translation.cast<double>();
				viewer->data(Model_Translate_ID).set_mesh(viewer->data(Model_Translate_ID).V, viewer->data(Model_Translate_ID).F);
			}
			else {
				for (auto& out : Outputs) {
					viewer->data(out.ModelID).V.row(Translate_Index) += translation.cast<double>();
					viewer->data(out.ModelID).set_mesh(viewer->data(out.ModelID).V, viewer->data(out.ModelID).F);
				}
			}
			down_mouse_x = mouse_x;
			down_mouse_y = mouse_y;
			UpdateVerticesHandles();
			return true;
		}
	}
	else if (IsTranslate && mouse_mode >= app_utils::MouseMode::FACE_CLUSTERING_0) {
		brush_erase_or_insert();
		return true;
	}
	else if (IsChoosingCluster && mouse_mode >= app_utils::MouseMode::FACE_CLUSTERING_0) {
		Eigen::Vector3f _;
		curr_highlighted_face = pick_face(_);
		return true;
	}

	return false;
}

IGL_INLINE bool deformation_plugin::mouse_scroll(float delta_y) {
	if (IsTranslate && mouse_mode >= app_utils::MouseMode::FACE_CLUSTERING_0) {
		brush_radius += delta_y * 0.05;
		brush_radius = std::max<float>(0.05, brush_radius);
		return true;
	}
	else if (IsChoosingCluster && mouse_mode >= app_utils::MouseMode::FACE_CLUSTERING_0) {
		neighbor_distance += delta_y * 0.05;
		neighbor_distance = std::max<float>(0.05, neighbor_distance);
		return true;
	}
	return false;
}

IGL_INLINE bool deformation_plugin::mouse_up(int button, int modifier) {
	IsTranslate = false;
	IsMouseDraggingAnyWindow = false;
	if (IsChoosingCluster) {
		IsChoosingCluster = false;
		curr_highlighted_face = -1;
		Eigen::Vector3f _;
		int f = pick_face(_);
		if (f != -1) {
			int clusterIndex = mouse_mode - app_utils::MouseMode::FACE_CLUSTERING_0;
			if (find(faceClusters[clusterIndex].faces.begin(), faceClusters[clusterIndex].faces.end(), f) != faceClusters[clusterIndex].faces.end()) {
				std::vector<int> neigh = Outputs[0].getNeigh(highlightFacesType, InputModel().F, f, neighbor_distance);
				for (int currF : neigh)
					faceClusters[clusterIndex].faces.erase(currF);
			}
			else {
				std::vector<int> neigh = Outputs[0].getNeigh(highlightFacesType, InputModel().F, f, neighbor_distance);
				for (int currF : neigh)
					if (find(faceClusters[clusterIndex].faces.begin(), faceClusters[clusterIndex].faces.end(), currF) == faceClusters[clusterIndex].faces.end())
						faceClusters[clusterIndex].faces.insert(currF);
			}
			UpdateClustersHandles();
		}
	}
	return false;
}

IGL_INLINE bool deformation_plugin::mouse_down(int button, int modifier) {
	if (IsMouseHoveringAnyWindow)
		IsMouseDraggingAnyWindow = true;
	down_mouse_x = viewer->current_mouse_x;
	down_mouse_y = viewer->current_mouse_y;
	
	//check if there faces which is selected on the left screen
	if (mouse_mode == app_utils::MouseMode::FIX_FACES && button == GLFW_MOUSE_BUTTON_LEFT && modifier == 2)
	{
		Eigen::Vector3f _;
		int f = pick_face(_);
		if (f != -1) {
			if (find(selected_fixed_faces.begin(), selected_fixed_faces.end(), f) != selected_fixed_faces.end())
				selected_fixed_faces.erase(f);
			else
				selected_fixed_faces.insert(f);
			UpdateCentersHandles();
		}
	}
	else if (mouse_mode >= app_utils::MouseMode::FACE_CLUSTERING_0 && button == GLFW_MOUSE_BUTTON_LEFT && modifier == 2)
	{
		int f = pick_face(intersec_point);
		brush_index = f;
		if (f != -1) {
			int clusterIndex = mouse_mode - app_utils::MouseMode::FACE_CLUSTERING_0;
			if (find(faceClusters[clusterIndex].faces.begin(), faceClusters[clusterIndex].faces.end(), f) != faceClusters[clusterIndex].faces.end())
				EraseOrInsert = true; //erase
			else 
				EraseOrInsert = false; // insert
			
			IsTranslate = true;
			//brush_erase_or_insert();
		}
	}
	else if (mouse_mode >= app_utils::MouseMode::FACE_CLUSTERING_0 && button == GLFW_MOUSE_BUTTON_MIDDLE && modifier == 2)
	{
		IsChoosingCluster = true;
		Eigen::Vector3f _;
		curr_highlighted_face = pick_face(_);
	}
	else if (mouse_mode == app_utils::MouseMode::FIX_VERTEX && button == GLFW_MOUSE_BUTTON_LEFT && modifier == 2)
	{
		//check if there faces which is selected on the left screen
		int v = pick_vertex();
		if (v != -1)
		{
			if (find(selected_vertices.begin(), selected_vertices.end(), v) != selected_vertices.end())
				selected_vertices.erase(v);
			else
				selected_vertices.insert(v);
			UpdateVerticesHandles();
		}
	}
	else if (mouse_mode == app_utils::MouseMode::FIX_FACES && button == GLFW_MOUSE_BUTTON_MIDDLE)
	{
		if (!selected_fixed_faces.empty())
		{
			//check if there faces which is selected on the left screen
			Eigen::Vector3f _;
			int f = pick_face(_,true);
			if (find(selected_fixed_faces.begin(), selected_fixed_faces.end(), f) != selected_fixed_faces.end()) {
				IsTranslate = true;
				Translate_Index = f;
			}
		}
	}
	else if (mouse_mode == app_utils::MouseMode::FIX_VERTEX && button == GLFW_MOUSE_BUTTON_MIDDLE)
	{
		if (!selected_vertices.empty())
		{
			//check if there faces which is selected on the left screen
			int v = pick_vertex(true);
			if (find(selected_vertices.begin(), selected_vertices.end(), v) != selected_vertices.end()) {
				IsTranslate = true;
				Translate_Index = v;
			}
		}
	}
	return false;
}

IGL_INLINE bool deformation_plugin::key_pressed(unsigned int key, int modifiers) {
	if ((key == '!') && modifiers == 1)
		mouse_mode = (app_utils::MouseMode)4;
	if ((key == '@') && modifiers == 1)
		mouse_mode = (app_utils::MouseMode)5;
	if ((key == '#') && modifiers == 1)
		mouse_mode = (app_utils::MouseMode) 6;
	if ((key == '$') && modifiers == 1)
		mouse_mode = (app_utils::MouseMode)7;
	if ((key == '%') && modifiers == 1)
		mouse_mode = (app_utils::MouseMode)8;
	if ((key == 'c' || key == 'C') && modifiers == 1)
		clear_sellected_faces_and_vertices();
	if ((key == 'a' || key == 'A') && modifiers == 1) {
		modelPath = OptimizationUtils::ProjectPath() + "\\models\\island.obj";
		isLoadNeeded = true;
	}
	if ((key == 's' || key == 'S') && modifiers == 1) {
		modelPath = OptimizationUtils::ProjectPath() + "\\models\\spot.obj";
		isLoadNeeded = true;
	}
	if (isModelLoaded && (key == 'q' || key == 'Q') && modifiers == 1) {
		highlightFacesType = app_utils::HighlightFaces::LOCAL_NORMALS;
		neighbor_distance = 0.03;
		change_minimizer_type(app_utils::MinimizerType::ADAM_MINIMIZER);
		showFacesNorm = true;
		showSphereEdges = showNormEdges = showTriangleCenters = showSphereCenters = false;
		for (OptimizationOutput& out : Outputs) {
			for (auto& obj : out.totalObjective->objectiveList) {
				std::shared_ptr<AuxSpherePerHinge> AS = std::dynamic_pointer_cast<AuxSpherePerHinge>(obj);
				std::shared_ptr<ClusterSpheres> CS = std::dynamic_pointer_cast<ClusterSpheres>(obj);
				std::shared_ptr<AuxBendingNormal> ABN = std::dynamic_pointer_cast<AuxBendingNormal>(obj);
				std::shared_ptr<ClusterNormals> CN = std::dynamic_pointer_cast<ClusterNormals>(obj);
				if(ABN != NULL)
					ABN->w = 1.6;
				if (CN != NULL)
					CN->w = 0.05;
				if (AS != NULL)
					AS->w = 0;
				if (CS != NULL)
					CS->w = 0;
			}
		}
	}
	if (isModelLoaded && (key == 'w' || key == 'W') && modifiers == 1) {
		highlightFacesType = app_utils::HighlightFaces::LOCAL_SPHERE;
		neighbor_distance = 0.3;
		change_minimizer_type(app_utils::MinimizerType::ADAM_MINIMIZER);
		showSphereCenters = true;
		showSphereEdges = showNormEdges = showTriangleCenters = showFacesNorm = false;
		for (OptimizationOutput& out : Outputs) {
			for (auto& obj : out.totalObjective->objectiveList) {
				std::shared_ptr<AuxSpherePerHinge> AS = std::dynamic_pointer_cast<AuxSpherePerHinge>(obj);
				std::shared_ptr<ClusterSpheres> CS = std::dynamic_pointer_cast<ClusterSpheres>(obj);
				std::shared_ptr<AuxBendingNormal> ABN = std::dynamic_pointer_cast<AuxBendingNormal>(obj);
				std::shared_ptr<ClusterNormals> CN = std::dynamic_pointer_cast<ClusterNormals>(obj);

				if(AS != NULL)
					AS->w = 1.6;
				if (CS != NULL)
					CS->w = 0.05;
				if (ABN != NULL)
					ABN->w = 0;
				if (CN != NULL)
					CN->w = 0;
			}
		}
	}
	
	if ((key == ' ') && modifiers == 1)
		isMinimizerRunning ? stop_minimizer_thread() : start_minimizer_thread();
	
	return ImGuiMenu::key_pressed(key, modifiers);
}

IGL_INLINE void deformation_plugin::shutdown()
{
	stop_minimizer_thread();
	ImGuiMenu::shutdown();
}

void deformation_plugin::draw_brush_sphere() {
	if (!(brush_index != -1 && IsTranslate && mouse_mode >= app_utils::MouseMode::FACE_CLUSTERING_0))
		return;
	//prepare brush sphere
	Eigen::MatrixXd sphere(36 * 36, 3);
	Eigen::RowVector3d center = intersec_point.cast<double>().transpose();
	int i, j;
	for (double alfa = 0, i = 0; alfa < 360; i++, alfa += 10) {
		for (double beta = 0, j = 0; beta < 360; j++, beta += 10) {
			Eigen::RowVector3d dir;
			dir << sin(alfa), cos(alfa)*cos(beta), sin(beta)*cos(alfa);
			sphere.row(i + 36 * j) = dir * brush_radius + center;
		}
	}

	//prepare color
	Eigen::MatrixXd c(1, 3);
	if (EraseOrInsert == false) { //insert
		int clusterIndex = mouse_mode - app_utils::MouseMode::FACE_CLUSTERING_0;
		c.row(0) = faceClusters[clusterIndex].color.cast<double>();
	}
	else if (EraseOrInsert == true) { //erase
		c.row(0) << 1, 1, 1; // white color for eraseing
	}

	//update data for cores
	InputModel().point_size = 10;
	InputModel().add_points(sphere, c);
	for (int oi = 0; oi < Outputs.size(); oi++) {
		OutputModel(oi).point_size = 10;
		OutputModel(oi).add_points(sphere, c);
	}
}

IGL_INLINE bool deformation_plugin::pre_draw() {
	//call parent function
	ImGuiMenu::pre_draw();
	for (auto& out : Outputs)
		if (out.activeMinimizer->progressed)
			update_data_from_minimizer();
	//Update the model's faces colors in the screens
	for (int i = 0; i < Outputs.size(); i++) {
		if (Outputs[i].color_per_face.size()) {
			InputModel().set_colors(Outputs[0].color_per_face);
			OutputModel(i).set_colors(Outputs[i].color_per_face);
		}
	}
	//Update the model's vertex colors in screens
	InputModel().point_size = 10;
	InputModel().set_points(Vertices_Input, color_per_vertex);
	for (int i = 0; i < Outputs.size(); i++) {
		OutputModel(i).point_size = 10;
		OutputModel(i).set_points(Outputs[i].Vertices_output, color_per_vertex);
	}
	
	draw_brush_sphere();

	for (int oi = 0; oi < Outputs.size(); oi++) {
		OutputModel(oi).clear_edges();
		OutputModel(oi).point_size = 10;
			
		if (showFacesNorm && Outputs[oi].getFacesNorm().size() != 0)
			OutputModel(oi).add_points(Outputs[oi].getFacesNorm(), Outputs[oi].color_per_face_norm);
		if (showTriangleCenters && Outputs[oi].getCenterOfFaces().size() != 0)
			OutputModel(oi).add_points(Outputs[oi].getCenterOfFaces(), Outputs[oi].color_per_vertex_center);
		if (showSphereCenters && Outputs[oi].getCenterOfSphere().size() != 0)
			OutputModel(oi).add_points(Outputs[oi].getCenterOfSphere(), Outputs[oi].color_per_sphere_center);
		if (showSphereEdges && Outputs[oi].getCenterOfFaces().size() != 0)
			OutputModel(oi).add_edges(Outputs[oi].getCenterOfFaces(), Outputs[oi].getSphereEdges(), Outputs[oi].color_per_sphere_edge);
		if (showNormEdges && Outputs[oi].getCenterOfFaces().size() != 0)
			OutputModel(oi).add_edges(Outputs[oi].getCenterOfFaces(), Outputs[oi].getFacesNorm(), Outputs[oi].color_per_norm_edge);
	}
	return false;
}

void deformation_plugin::Draw_menu_for_colors() {
	ImVec2 screen_pos = ImGui::GetCursorScreenPos();
	if (!ImGui::CollapsingHeader("colors", ImGuiTreeNodeFlags_DefaultOpen))
	{
		ImGui::ColorEdit3("Highlighted face color", Highlighted_face_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
		ImGui::ColorEdit3("Center sphere color", center_sphere_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
		ImGui::ColorEdit3("Center vertex color", center_vertex_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
		ImGui::ColorEdit3("Centers sphere edge color", centers_sphere_edge_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
		ImGui::ColorEdit3("Centers norm edge color", centers_norm_edge_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
		ImGui::ColorEdit3("Face norm color", face_norm_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
		ImGui::ColorEdit3("Neighbors Highlighted face color", Neighbors_Highlighted_face_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
		ImGui::ColorEdit3("Fixed face color", Fixed_face_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
		ImGui::ColorEdit3("Dragged face color", Dragged_face_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
		ImGui::ColorEdit3("Fixed vertex color", Fixed_vertex_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
		ImGui::ColorEdit3("Dragged vertex color", Dragged_vertex_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
		ImGui::ColorEdit3("Model color", model_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
		ImGui::ColorEdit3("Vertex Energy color", Vertex_Energy_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
		ImGui::ColorEdit4("text color", text_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
	}
}

void deformation_plugin::change_minimizer_type(app_utils::MinimizerType type) {
	minimizer_type = type;
	stop_minimizer_thread();
	init_minimizer_thread();
	for (int i = 0; i < Outputs.size(); i++)
		Outputs[i].updateActiveMinimizer(minimizer_type);
}

void deformation_plugin::Draw_menu_for_Minimizer() {
	if (ImGui::CollapsingHeader("Minimizer", ImGuiTreeNodeFlags_DefaultOpen))
	{
		if (ImGui::Checkbox("Only one iteration", &runOneIteration) && isMinimizerRunning)
			start_minimizer_thread();
		if (ImGui::Checkbox(isMinimizerRunning ? "On" : "Off", &isMinimizerRunning))
			isMinimizerRunning ? start_minimizer_thread() : stop_minimizer_thread();
		ImGui::Checkbox("Minimizer settings", &minimizer_settings);
		ImGui::Text("Show:");
		ImGui::Checkbox("Norm", &showFacesNorm);
		ImGui::SameLine();
		ImGui::Checkbox("Norm Edges", &showNormEdges);
		ImGui::Checkbox("Sphere", &showSphereCenters);
		ImGui::SameLine();
		ImGui::Checkbox("Sphere Edges", &showSphereEdges);
		ImGui::Checkbox("Face Centers", &showTriangleCenters);
		if (ImGui::Combo("Minimizer type", (int *)(&minimizer_type), "Newton\0Gradient Descent\0Adam\0\0")) {
			change_minimizer_type(minimizer_type);
		}
		if (ImGui::Combo("init Aux Var", (int *)(&typeAuxVar), "Sphere\0Mesh Center\0\0"))
			init_minimizer_thread();
		std::shared_ptr<NewtonMinimizer> newtonMinimizer = std::dynamic_pointer_cast<NewtonMinimizer>(Outputs[0].activeMinimizer);
		if (newtonMinimizer != NULL) {
			bool PD = newtonMinimizer->getPositiveDefiniteChecker();
			ImGui::Checkbox("Positive Definite check", &PD);
			for (auto& o : Outputs) {
				std::dynamic_pointer_cast<NewtonMinimizer>(o.activeMinimizer)->SwitchPositiveDefiniteChecker(PD);
			}
		}
		if (ImGui::Combo("line search", (int *)(&linesearch_type), "Gradient Norm\0Function Value\0Constant Step\0\0")) {
			for (auto& o:Outputs)
				o.activeMinimizer->lineSearch_type = linesearch_type;
		}
		if (linesearch_type == OptimizationUtils::LineSearch::CONSTANT_STEP && ImGui::DragFloat("Step value", &constantStep_LineSearch, 0.0001f, 0.0f, 1.0f)) {
			for (auto& o : Outputs)
				o.activeMinimizer->constantStep_LineSearch = constantStep_LineSearch;
		}
		ImGui::Combo("Face coloring", (int *)(&faceColoring_type), app_utils::build_color_energies_list(Outputs[0].totalObjective));
		float w = ImGui::GetContentRegionAvailWidth(), p = ImGui::GetStyle().FramePadding.x;
		if (ImGui::Button("Check gradients", ImVec2((w - p) / 2.f, 0)))
			checkGradients();
		ImGui::SameLine(0, p);
		if (ImGui::Button("Check Hessians", ImVec2((w - p) / 2.f, 0)))
			checkHessians();
	}
}

void deformation_plugin::Draw_menu_for_cores(igl::opengl::ViewerCore& core) {
	if (!Outputs_Settings)
		return;
	ImGui::PushID(core.id);
	std::stringstream ss;
	std::string name = (core.id == inputCoreID) ? "Input Core" : "Output Core " + std::to_string(core.id);
	ss << name;
	if (!ImGui::CollapsingHeader(ss.str().c_str(), ImGuiTreeNodeFlags_DefaultOpen))
	{
		int data_id;
		for (int i = 0; i < Outputs.size(); i++)
			if (core.id == Outputs[i].CoreID)
				data_id = Outputs[i].ModelID;
		if (core.id == inputCoreID)
			data_id = inputModelID;
		if (ImGui::Button("Center object", ImVec2(-1, 0)))
			core.align_camera_center(viewer->data_list[data_id].V, viewer->data_list[data_id].F);
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
		if (ImGui::Combo("Camera Type", &rotation_type, "Trackball\0Two Axes\0002D Mode\0\0"))
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
		// Orthographic view
		ImGui::Checkbox("Orthographic view", &(core.orthographic));
		ImGui::PopItemWidth();
		ImGui::ColorEdit4("Background", core.background_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
	}
	ImGui::PopID();
}

void deformation_plugin::Draw_menu_for_models(igl::opengl::ViewerData& data) {
	if (!Outputs_Settings)
		return;
	// Helper for setting viewport specific mesh options
	auto make_checkbox = [&](const char *label, unsigned int &option) {
		bool temp = option;
		bool res = ImGui::Checkbox(label, &temp);
		option = temp;
		return res;
	};
	ImGui::PushID(data.id);
	std::stringstream ss;
	if (data.id == inputModelID)
		ss << modelName;
	else
		ss << modelName + " " + std::to_string(data.id) + " (Param.)";
			
	if (!ImGui::CollapsingHeader(ss.str().c_str(), ImGuiTreeNodeFlags_DefaultOpen))
	{
		float w = ImGui::GetContentRegionAvailWidth();
		float p = ImGui::GetStyle().FramePadding.x;
		if (data.id == inputModelID)
			ImGui::SliderFloat("texture", &texture_scaling_input, 0.01, 100, std::to_string(texture_scaling_input).c_str(), 1);
		else
			ImGui::SliderFloat("texture", &texture_scaling_output, 0.01, 100, std::to_string(texture_scaling_output).c_str(), 1);	
		if (ImGui::Checkbox("Face-based", &(data.face_based)))
			data.dirty = igl::opengl::MeshGL::DIRTY_ALL;
		make_checkbox("Show texture", data.show_texture);
		if (ImGui::Checkbox("Invert normals", &(data.invert_normals)))
			data.dirty |= igl::opengl::MeshGL::DIRTY_NORMAL;
		make_checkbox("Show overlay", data.show_overlay);
		make_checkbox("Show overlay depth", data.show_overlay_depth);
		ImGui::ColorEdit4("Line color", data.line_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
		ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.3f);
		ImGui::DragFloat("Shininess", &(data.shininess), 0.05f, 0.0f, 100.0f);
		ImGui::PopItemWidth();
		make_checkbox("Wireframe", data.show_lines);
		make_checkbox("Fill", data.show_faces);
		ImGui::Checkbox("Show vertex labels", &(data.show_vertid));
		ImGui::Checkbox("Show faces labels", &(data.show_faceid));
	}
	ImGui::PopID();
}

void deformation_plugin::Draw_menu_for_minimizer_settings() {
	// 	ImGui::SetNextWindowSize(ImVec2(1000, 0), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowPos(ImVec2(200, 550), ImGuiCond_FirstUseEver);
	ImGui::Begin("minimizer settings", NULL, ImGuiWindowFlags_AlwaysAutoResize);
	//add outputs buttons
	ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.0f, 0.6f, 0.0f, 1.0f));
	if (ImGui::Button("Add Output"))
		add_output();
	ImGui::PopStyleColor();
	int id = 0;
	if (Outputs.size() != 0) {
		if (ImGui::BeginTable("Unconstrained weights table", Outputs[0].totalObjective->objectiveList.size() + 3, ImGuiTableFlags_Resizable))
		{

			ImGui::TableSetupColumn("Outputs", ImGuiTableColumnFlags_WidthFixed);
			ImGui::TableSetupColumn("##col1", ImGuiTableColumnFlags_WidthFixed);
			for (auto& obj : Outputs[0].totalObjective->objectiveList) {
				ImGui::TableSetupColumn(obj->name.c_str(), ImGuiTableColumnFlags_WidthFixed);
			}
			ImGui::TableSetupColumn("shift eigen values", ImGuiTableColumnFlags_WidthFixed);
			ImGui::TableAutoHeaders();

			ImGui::Separator();

			ImGui::TableNextRow();
			for (int i = 0; i < Outputs.size(); i++) {
				ImGui::Text(("Output " + std::to_string(Outputs[i].CoreID)).c_str());

				if (ImGui::Button("Remove"))
					remove_output(i);

				ImGui::TableNextCell();
				ImGui::Text("Weight");
				ImGui::TableNextCell();


				ImGui::PushItemWidth(100);
				for (auto& obj : Outputs[i].totalObjective->objectiveList) {
					ImGui::PushID(id++);
					ImGui::DragFloat("##w", &(obj->w), 0.05f, 0.0f, 100000.0f);
					auto BE = std::dynamic_pointer_cast<BendingEdge>(obj);
					auto BN = std::dynamic_pointer_cast<BendingNormal>(obj);
					auto ABN = std::dynamic_pointer_cast<AuxBendingNormal>(obj);
					auto AS = std::dynamic_pointer_cast<AuxSpherePerHinge>(obj);
					if (true) {
						if (BE != NULL)
							ImGui::Combo("Function", (int*)(&(BE->functionType)), "Quadratic\0Exponential\0Sigmoid\0\0");
						if (BN != NULL)
							ImGui::Combo("Function", (int*)(&(BN->functionType)), "Quadratic\0Exponential\0Sigmoid\0\0");
						if (ABN != NULL)
							ImGui::Combo("Function", (int*)(&(ABN->functionType)), "Quadratic\0Exponential\0Sigmoid\0\0");
						if (AS != NULL)
							ImGui::Combo("Function", (int*)(&(AS->functionType)), "Quadratic\0Exponential\0Sigmoid\0\0");

						if (BE != NULL && BE->functionType == OptimizationUtils::FunctionType::SIGMOID) {
							ImGui::Text((std::to_string(BE->planarParameter)).c_str());
							ImGui::SameLine();
							if (ImGui::Button("*", ImVec2(ImGui::GetFrameHeight(), ImGui::GetFrameHeight())))
							{
								BE->planarParameter = (BE->planarParameter * 2) > 1 ? 1 : BE->planarParameter * 2;
							}
							ImGui::SameLine();
							if (ImGui::Button("/", ImVec2(ImGui::GetFrameHeight(), ImGui::GetFrameHeight())))
							{
								BE->planarParameter /= 2;
							}
						}
						if (BN != NULL && BN->functionType == OptimizationUtils::FunctionType::SIGMOID) {
							ImGui::Text((std::to_string(BN->planarParameter)).c_str());
							ImGui::SameLine();
							if (ImGui::Button("*", ImVec2(ImGui::GetFrameHeight(), ImGui::GetFrameHeight())))
							{
								BN->planarParameter = (BN->planarParameter * 2) > 1 ? 1 : BN->planarParameter * 2;
							}
							ImGui::SameLine();
							if (ImGui::Button("/", ImVec2(ImGui::GetFrameHeight(), ImGui::GetFrameHeight())))
							{
								BN->planarParameter /= 2;
							}
						}
						if (ABN != NULL && ABN->functionType == OptimizationUtils::FunctionType::SIGMOID) {
							ImGui::Text((std::to_string(ABN->planarParameter)).c_str());
							ImGui::SameLine();
							if (ImGui::Button("*", ImVec2(ImGui::GetFrameHeight(), ImGui::GetFrameHeight())))
							{
								ABN->planarParameter = (ABN->planarParameter * 2) > 1 ? 1 : ABN->planarParameter * 2;
							}
							ImGui::SameLine();
							if (ImGui::Button("/", ImVec2(ImGui::GetFrameHeight(), ImGui::GetFrameHeight())))
							{
								ABN->planarParameter /= 2;
							}
							const double  f64_zero = 0, f64_max = 100000.0;
							ImGui::DragScalar("w1", ImGuiDataType_Double, &(ABN->w1), 0.05f, &f64_zero, &f64_max);
							ImGui::DragScalar("w2", ImGuiDataType_Double, &(ABN->w2), 0.05f, &f64_zero, &f64_max);
							ImGui::DragScalar("w3", ImGuiDataType_Double, &(ABN->w3), 0.05f, &f64_zero, &f64_max);
						}
						if (AS != NULL && AS->functionType == OptimizationUtils::FunctionType::SIGMOID) {
							ImGui::Text((std::to_string(AS->planarParameter)).c_str());
							ImGui::SameLine();
							if (ImGui::Button("*", ImVec2(ImGui::GetFrameHeight(), ImGui::GetFrameHeight())))
							{
								AS->planarParameter = (AS->planarParameter * 2) > 1 ? 1 : AS->planarParameter * 2;
							}
							ImGui::SameLine();
							if (ImGui::Button("/", ImVec2(ImGui::GetFrameHeight(), ImGui::GetFrameHeight())))
							{
								AS->planarParameter /= 2;
							}
							const double  f64_zero = 0, f64_max = 100000.0;
							ImGui::DragScalar("w0", ImGuiDataType_Double, &(AS->w_aux[0]), 0.05f, &f64_zero, &f64_max);
							ImGui::DragScalar("w1", ImGuiDataType_Double, &(AS->w_aux[1]), 0.05f, &f64_zero, &f64_max);
						}
					}
					ImGui::TableNextCell();
					ImGui::PopID();
				}
				ImGui::DragFloat("##ShiftEigenValues", &(Outputs[i].totalObjective->Shift_eigen_values), 0.05f, 0.0f, 100000.0f);
				ImGui::TableNextCell();
				ImGui::PopItemWidth();
			}
			ImGui::EndTable();
		}

		static bool show = false;
		if (ImGui::Button("More info")) {
			show = !show;
		}
		if (show) {
			//add more features
			Draw_menu_for_colors();
			ImGui::PushItemWidth(80 * menu_scaling());
			ImGui::DragFloat("Max Distortion", &Max_Distortion, 0.05f, 0.01f, 10000.0f);
			ImGui::PopItemWidth();
		}
	}
	//close the window
	ImGui::End();
}

void deformation_plugin::Draw_menu_for_output_settings() {
	for (auto& out : Outputs) {
		if (Outputs_Settings) {
			ImGui::SetNextWindowSize(ImVec2(200, 300), ImGuiCond_FirstUseEver);
			ImGui::Begin(("Output settings " + std::to_string(out.CoreID)).c_str(),
				NULL, 
				ImGuiWindowFlags_NoTitleBar |
				ImGuiWindowFlags_NoResize |
				ImGuiWindowFlags_NoMove
			);
			ImGui::SetWindowPos(out.text_position);
			Draw_menu_for_cores(viewer->core(out.CoreID));
			Draw_menu_for_models(viewer->data(out.ModelID));
			ImGui::End();
		}
	}
}

void deformation_plugin::Draw_menu_for_text_results() {
	for (auto& out:Outputs) {
		if (show_text) {
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
			ImGui::SetWindowPos(out.text_position);
			ImGui::SetWindowSize(out.window_size);
			ImGui::SetWindowCollapsed(false);
			//add text...
			ImGui::TextColored(c, (std::string(out.totalObjective->name) + std::string(" energy ") + std::to_string(out.totalObjective->energy_value)).c_str());
			ImGui::TextColored(c, (std::string(out.totalObjective->name) + std::string(" gradient ") + std::to_string(out.totalObjective->gradient_norm)).c_str());
			for (auto& obj : out.totalObjective->objectiveList) {
				ImGui::TextColored(c, (std::string(obj->name) + std::string(" energy ") + std::to_string(obj->energy_value)).c_str());
				ImGui::TextColored(c, (std::string(obj->name) + std::string(" gradient ") + std::to_string(obj->gradient_norm)).c_str());
			}
			ImGui::End();
			ImGui::PopStyleColor();
		}
	}
}

void deformation_plugin::UpdateCentersHandles() {
	std::vector<int> CurrCentersInd;
	std::vector<Eigen::MatrixX3d> CurrCentersPos;
	CurrCentersInd.clear();
	for (auto fi : selected_fixed_faces)
		CurrCentersInd.push_back(fi);
	for (auto& out : Outputs)
		CurrCentersPos.push_back(Eigen::MatrixX3d::Zero(CurrCentersInd.size(), 3));

	for (int i = 0; i < Outputs.size(); i++) {
		int idx = 0;
		for (auto ci : CurrCentersInd) {
			if (Outputs[i].getCenterOfSphere().size() != 0)
				CurrCentersPos[i].row(idx) = Outputs[i].getCenterOfSphere().row(ci);
			idx++;
		}
	}
	//Finally, we update the handles in the constraints positional object
	for (int i = 0; i < Outputs.size(); i++) {
		if (isModelLoaded) {
			*(Outputs[i].CentersInd) = CurrCentersInd;
			*(Outputs[i].CentersPosDeformed) = CurrCentersPos[i];
		}
	}
}

void deformation_plugin::UpdateClustersHandles() {
	std::vector < std::vector<int>> ind(faceClusters.size());
	for (int ci = 0; ci < faceClusters.size(); ci++)
		for (int fi : faceClusters[ci].faces)
			ind[ci].push_back(fi);
	for (int i = 0; i < Outputs.size(); i++)
		if (isModelLoaded) {
			*(Outputs[i].ClustersSphereInd) = ind;
			*(Outputs[i].ClustersNormInd) = ind;
		}	
}

void deformation_plugin::UpdateVerticesHandles() {
	std::vector<int> CurrHandlesInd;
	std::vector<Eigen::MatrixX3d> CurrHandlesPosDeformed;
	CurrHandlesInd.clear();

	//First, we push each vertices index to the handles
	for (auto vi : selected_vertices)
		CurrHandlesInd.push_back(vi);
	//Here we update the positions for each handle
	for (auto& out :Outputs)
		CurrHandlesPosDeformed.push_back(Eigen::MatrixX3d::Zero(CurrHandlesInd.size(),3));
	for (int i = 0; i < Outputs.size(); i++){
		int idx = 0;
		for (auto hi : CurrHandlesInd)
			CurrHandlesPosDeformed[i].row(idx++) << OutputModel(i).V(hi, 0), OutputModel(i).V(hi, 1), OutputModel(i).V(hi, 2);
		set_vertices_for_mesh(OutputModel(i).V, i);
	}
	//Finally, we update the handles in the constraints positional object
	for (int i = 0; i < Outputs.size();i++) {
		if (isModelLoaded) {
			*(Outputs[i].HandlesInd) = CurrHandlesInd;
			*(Outputs[i].HandlesPosDeformed) = CurrHandlesPosDeformed[i];
		}
	}
}

void deformation_plugin::Update_view() {
	for (auto& data : viewer->data_list)
		for (auto& out : Outputs)
			data.copy_options(viewer->core(inputCoreID), viewer->core(out.CoreID));
	for (auto& core : viewer->core_list)
		for (auto& data : viewer->data_list)
			viewer->data(data.id).set_visible(false, core.id);
	InputModel().set_visible(true, inputCoreID);
	for (int i = 0; i < Outputs.size(); i++)
		OutputModel(i).set_visible(true, Outputs[i].CoreID);
}

void deformation_plugin::follow_and_mark_selected_faces() {
	if(InputModel().F.size()){
		
		//Mark the faces
		for (int i = 0; i < Outputs.size(); i++) {
			Outputs[i].initFaceColors(InputModel().F.rows(),center_sphere_color,center_vertex_color, centers_sphere_edge_color, centers_norm_edge_color, face_norm_color);
			UpdateEnergyColors(i);
			//Mark the cluster faces
			for (FaceClusters cluster : faceClusters)
				for (int fi : cluster.faces)
					Outputs[i].updateFaceColors(fi, cluster.color);
			//Mark the fixed faces
			for (int fi : selected_fixed_faces)
				Outputs[i].updateFaceColors(fi, Fixed_face_color);
			//Mark the highlighted face & neighbors
			if (curr_highlighted_face != -1 && highlightFacesType != app_utils::HighlightFaces::NO_HIGHLIGHT) {
				std::vector<int> neigh = Outputs[i].getNeigh(highlightFacesType,InputModel().F, curr_highlighted_face, neighbor_distance);
				for (int fi : neigh)
					Outputs[i].updateFaceColors(fi, Neighbors_Highlighted_face_color);
				Outputs[i].updateFaceColors(curr_highlighted_face, Highlighted_face_color);
			}
			//Mark the Dragged face
			if (IsTranslate && (mouse_mode == app_utils::MouseMode::FIX_FACES))
				Outputs[i].updateFaceColors(Translate_Index, Dragged_face_color);
			//Mark the vertices
			int idx = 0;
			Vertices_Input.resize(selected_vertices.size(), 3);
			Outputs[i].Vertices_output.resize(selected_vertices.size(), 3);
			color_per_vertex.resize(selected_vertices.size(), 3);
			//Mark the dragged vertex
			if (IsTranslate && (mouse_mode == app_utils::MouseMode::FIX_VERTEX)) {
				Vertices_Input.resize(selected_vertices.size() + 1, 3);
				Outputs[i].Vertices_output.resize(selected_vertices.size() + 1, 3);
				color_per_vertex.resize(selected_vertices.size() + 1, 3);
				Vertices_Input.row(idx) = InputModel().V.row(Translate_Index);
				color_per_vertex.row(idx) = Dragged_vertex_color.cast<double>();
				Outputs[i].Vertices_output.row(idx) = OutputModel(i).V.row(Translate_Index);
				idx++;
			}
			//Mark the fixed vertices
			for (auto vi : selected_vertices) {
				Vertices_Input.row(idx) = InputModel().V.row(vi);
				Outputs[i].Vertices_output.row(idx) = OutputModel(i).V.row(vi);
				color_per_vertex.row(idx++) = Fixed_vertex_color.cast<double>();
			}
		}
	}
}
	
igl::opengl::ViewerData& deformation_plugin::InputModel() {
	return viewer->data(inputModelID);
}

igl::opengl::ViewerData& deformation_plugin::OutputModel(const int index) {
	return viewer->data(Outputs[index].ModelID);
}

int deformation_plugin::pick_face(Eigen::Vector3f& intersec_point, const bool update) {
	//check if there faces which is selected on the left screen
	int f = pick_face_per_core(InputModel().V, InputModel().F, app_utils::View::INPUT_ONLY, intersec_point);
	if (update) {
		Model_Translate_ID = inputModelID;
		Core_Translate_ID = inputCoreID;
	}
	for (int i = 0; i < Outputs.size(); i++) {
		if (f == -1) {
			f = pick_face_per_core(OutputModel(i).V, OutputModel(i).F, app_utils::View::OUTPUT_ONLY_0 + i, intersec_point);
			if (update) {
				Model_Translate_ID = Outputs[i].ModelID;
				Core_Translate_ID = Outputs[i].CoreID;
			}
		}
	}
	return f;
}

int deformation_plugin::pick_face_per_core(Eigen::MatrixXd& V, Eigen::MatrixXi& F, int CoreIndex, Eigen::Vector3f& intersec_point) {
	// Cast a ray in the view direction starting from the mouse position
	int CoreID;
	if (CoreIndex == app_utils::View::INPUT_ONLY)
		CoreID = inputCoreID;
	else
		CoreID = Outputs[CoreIndex - app_utils::View::OUTPUT_ONLY_0].CoreID;
	double x = viewer->current_mouse_x;
	double y = viewer->core(CoreID).viewport(3) - viewer->current_mouse_y;
	if (view == app_utils::View::VERTICAL) {
		y = (viewer->core(inputCoreID).viewport(3) / core_size) - viewer->current_mouse_y;
	}
	
	//Eigen::RowVector3d pt;
	Eigen::RowVector3d pt;
	Eigen::Matrix4f modelview = viewer->core(CoreID).view;
	int vi = -1;
	std::vector<igl::Hit> hits;
	igl::unproject_in_mesh(Eigen::Vector2f(x, y), viewer->core(CoreID).view,
		viewer->core(CoreID).proj, viewer->core(CoreID).viewport, V, F, pt, hits);
	Eigen::Vector3f s, dir;
	igl::unproject_ray(Eigen::Vector2f(x, y), viewer->core(CoreID).view,
		viewer->core(CoreID).proj, viewer->core(CoreID).viewport, s, dir);
	int fi = -1;
	if (hits.size() > 0) {
		fi = hits[0].id;
		intersec_point = s + dir * hits[0].t;
	}
	return fi;
}

int deformation_plugin::pick_vertex(const bool update) {
	//check if there faces which is selected on the left screen
	int v = pick_vertex_per_core(InputModel().V, InputModel().F, app_utils::View::INPUT_ONLY);
	if (update) {
		Model_Translate_ID = inputModelID;
		Core_Translate_ID = inputCoreID;
	}
	for (int i = 0; i < Outputs.size(); i++) {
		if (v == -1) {
			v = pick_vertex_per_core(OutputModel(i).V, OutputModel(i).F, app_utils::View::OUTPUT_ONLY_0 + i);
			if (update) {
				Model_Translate_ID = Outputs[i].ModelID;
				Core_Translate_ID = Outputs[i].CoreID;
			}
		}
	}
	return v;
}

int deformation_plugin::pick_vertex_per_core(Eigen::MatrixXd& V, Eigen::MatrixXi& F, int CoreIndex) {
	// Cast a ray in the view direction starting from the mouse position
	int CoreID;
	if (CoreIndex == app_utils::View::INPUT_ONLY)
		CoreID = inputCoreID;
	else
		CoreID = Outputs[CoreIndex - app_utils::View::OUTPUT_ONLY_0].CoreID;
	double x = viewer->current_mouse_x;
	double y = viewer->core(CoreID).viewport(3) - viewer->current_mouse_y;
	if (view == app_utils::View::VERTICAL) {
		y = (viewer->core(inputCoreID).viewport(3) / core_size) - viewer->current_mouse_y;
	}
	Eigen::Matrix<double, 3, 1, 0, 3, 1> pt;
	Eigen::Matrix4f modelview = viewer->core(CoreID).view;
	int vi = -1;
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
	if (hits.size() > 0) {
		int fi = hits[0].id;
		Eigen::RowVector3d bc;
		bc << 1.0 - hits[0].u - hits[0].v, hits[0].u, hits[0].v;
		bc.maxCoeff(&vi);
		vi = F(fi, vi);
	}
	return vi;
}

void deformation_plugin::set_vertices_for_mesh(Eigen::MatrixXd& V_uv, const int index) {
	Eigen::MatrixXd V_uv_3D(V_uv.rows(),3);
	if (V_uv.cols() == 2) {
		V_uv_3D.leftCols(2) = V_uv.leftCols(2);
		V_uv_3D.rightCols(1).setZero();
	}
	else if (V_uv.cols() == 3) {
		V_uv_3D = V_uv;
	}
	OutputModel(index).set_vertices(V_uv_3D);
	OutputModel(index).compute_normals();
}
	
void deformation_plugin::checkGradients()
{
	stop_minimizer_thread();
	for (auto& o: Outputs) {
		if (!isModelLoaded) {
			isMinimizerRunning = false;
			return;
		}
		Eigen::VectorXd xx = Eigen::VectorXd::Random(InputModel().V.size() + 7*InputModel().F.rows());
		for (auto const &objective : o.totalObjective->objectiveList)
			objective->checkGradient(xx);
	}
}

void deformation_plugin::checkHessians()
{
	stop_minimizer_thread();
	for (auto& o : Outputs) {
		if (!isModelLoaded) {
			isMinimizerRunning = false;
			return;
		}
		Eigen::VectorXd xx = Eigen::VectorXd::Random(InputModel().V.size() + 7*InputModel().F.rows());
		for (auto const &objective : o.totalObjective->objectiveList)
			objective->checkHessian(xx);
	}
}

void deformation_plugin::update_data_from_minimizer()
{
	std::vector<Eigen::MatrixXd> V(Outputs.size()),center(Outputs.size()),norm(Outputs.size());
	std::vector<Eigen::VectorXd> radius(Outputs.size());
	for (int i = 0; i < Outputs.size(); i++){
		Outputs[i].activeMinimizer->get_data(V[i], center[i],radius[i],norm[i]);
		if (IsTranslate && mouse_mode == app_utils::MouseMode::FIX_VERTEX)
			V[i].row(Translate_Index) = OutputModel(i).V.row(Translate_Index);
		else if (IsTranslate && mouse_mode == app_utils::MouseMode::FIX_FACES)
			if (Outputs[i].getCenterOfSphere().size() != 0)
				center[i].row(Translate_Index) = Outputs[i].getCenterOfSphere().row(Translate_Index);
		Outputs[i].setAuxVariables(V[i], InputModel().F, center[i],radius[i],norm[i]);
		set_vertices_for_mesh(V[i],i);
	}
}

void deformation_plugin::stop_minimizer_thread() {
	isMinimizerRunning = false;
	for (auto&o : Outputs) {
		if (o.activeMinimizer->is_running) {
			o.activeMinimizer->stop();
		}
		while (o.activeMinimizer->is_running);
	}
}

void deformation_plugin::init_minimizer_thread() {
	stop_minimizer_thread();
	for (int i = 0; i < Outputs.size(); i++)
		Outputs[i].initMinimizers(OutputModel(i).V, OutputModel(i).F, typeAuxVar);
}

void deformation_plugin::start_minimizer_thread() {
	if (!isModelLoaded) {
		isMinimizerRunning = false;
		return;
	}
	stop_minimizer_thread();
	init_minimizer_thread();
	for (int i = 0; i < Outputs.size();i++) {
		std::cout << ">> A new minimizer has been started" << std::endl;
		isMinimizerRunning = true;
		//start minimizer
		if (runOneIteration) {
			static int iteration_counter = 0;
			minimizer_thread = std::thread(&Minimizer::run_one_iteration, Outputs[i].activeMinimizer.get(), iteration_counter++,true);
			minimizer_thread.join();
		}
		else {
			minimizer_thread = std::thread(&Minimizer::run, Outputs[i].activeMinimizer.get());
			minimizer_thread.detach();
		}
	}
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
	//std::shared_ptr <BendingEdge> bendingEdge = std::make_unique<BendingEdge>(OptimizationUtils::FunctionType::SIGMOID);
	//bendingEdge->init_mesh(V, F);
	//bendingEdge->init();
	std::shared_ptr <AuxBendingNormal> auxBendingNormal = std::make_unique<AuxBendingNormal>(OptimizationUtils::FunctionType::SIGMOID);
	auxBendingNormal->init_mesh(V, F);
	auxBendingNormal->init();
	std::shared_ptr <AuxSpherePerHinge> auxSpherePerHinge = std::make_unique<AuxSpherePerHinge>(OptimizationUtils::FunctionType::SIGMOID);
	auxSpherePerHinge->init_mesh(V, F);
	auxSpherePerHinge->init();
	//std::shared_ptr <BendingNormal> bendingNormal = std::make_unique<BendingNormal>(OptimizationUtils::FunctionType::SIGMOID);
	//bendingNormal->init_mesh(V, F);
	//bendingNormal->init();
	std::shared_ptr <SymmetricDirichlet> SymmDirich = std::make_unique<SymmetricDirichlet>();
	SymmDirich->init_mesh(V, F);
	SymmDirich->init();
	std::shared_ptr <STVK> stvk = std::make_unique<STVK>();
	if (app_utils::IsMesh2D(InputModel().V)) {
		stvk->init_mesh(V, F);
		stvk->init();
	}
	std::shared_ptr <FixAllVertices> fixAllVertices = std::make_unique<FixAllVertices>();
	fixAllVertices->init_mesh(V, F);
	fixAllVertices->init();
	std::shared_ptr <FixChosenVertices> fixChosenVertices = std::make_shared<FixChosenVertices>();
	fixChosenVertices->numV = V.rows();
	fixChosenVertices->numF = F.rows();
	fixChosenVertices->init();
	Outputs[index].HandlesInd = &(fixChosenVertices->ConstrainedVerticesInd);
	Outputs[index].HandlesPosDeformed = &(fixChosenVertices->ConstrainedVerticesPos);
	std::shared_ptr< FixChosenSpheres> fixChosenSpheres = std::make_shared<FixChosenSpheres>();
	fixChosenSpheres->numV = V.rows();
	fixChosenSpheres->numF = F.rows();
	fixChosenSpheres->init();
	Outputs[index].CentersInd = &(fixChosenSpheres->ConstrainedCentersInd);
	Outputs[index].CentersPosDeformed = &(fixChosenSpheres->ConstrainedCentersPos);
	std::shared_ptr< ClusterSpheres> clusterSpheres = std::make_shared<ClusterSpheres>();
	clusterSpheres->numV = V.rows();
	clusterSpheres->numF = F.rows();
	clusterSpheres->init();
	Outputs[index].ClustersSphereInd = &(clusterSpheres->ClustersInd);
	std::shared_ptr< ClusterNormals> clusterNormals = std::make_shared<ClusterNormals>();
	clusterNormals->numV = V.rows();
	clusterNormals->numF = F.rows();
	clusterNormals->init();
	Outputs[index].ClustersNormInd = &(clusterNormals->ClustersInd);
	//init total objective
	Outputs[index].totalObjective->objectiveList.clear();
	Outputs[index].totalObjective->init_mesh(V, F);
	auto add_obj = [&](std::shared_ptr< ObjectiveFunction> obj) {
		Outputs[index].totalObjective->objectiveList.push_back(move(obj));
	};
	add_obj(auxSpherePerHinge);
	add_obj(auxBendingNormal);
	//add_obj(bendingNormal);
	//add_obj(bendingEdge);
	add_obj(SymmDirich);
	if(app_utils::IsMesh2D(InputModel().V))
		add_obj(stvk);
	add_obj(fixAllVertices);
	add_obj(fixChosenVertices);
	add_obj(fixChosenSpheres);
	add_obj(clusterSpheres);
	add_obj(clusterNormals);
	Outputs[index].totalObjective->init();
	std::cout  << "-------Energies, end-------" << console_color::white << std::endl;
	init_minimizer_thread();
}

void deformation_plugin::UpdateEnergyColors(const int index) {
	int numF = OutputModel(index).F.rows();
	Eigen::VectorXd DistortionPerFace(numF);
	DistortionPerFace.setZero();
	if (faceColoring_type == 0) { // No colors
		DistortionPerFace.setZero();
	}
	else if (faceColoring_type == 1) { // total energy
		for (auto& obj: Outputs[index].totalObjective->objectiveList) {
			// calculate the distortion over all the energies
			if ((obj->Efi.size() != 0) && (obj->w != 0))
				DistortionPerFace += obj->Efi * obj->w;
		}
	}
	else {
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
