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
		typeAuxVar = OptimizationUtils::Sphere;
		isLoadNeeded = false;
		IsMouseDraggingAnyWindow = false;
		IsMouseHoveringAnyWindow = false;
		worhp_on = false;
		solver_on = false;
		Outputs_Settings = false;
		Highlighted_face = false;
		IsTranslate = false;
		model_loaded = false;
		UpdateAll = true;
		solver_settings = true;
		show_text = true;
		step_by_step = false;
		distortion_type = app_utils::Distortion::TOTAL_DISTORTION;
		solver_type = app_utils::SolverType::NEWTON;
		linesearch_type = OptimizationUtils::LineSearch::FunctionValue;
		mouse_mode = app_utils::MouseMode::VERTEX_SELECT;
		view = app_utils::View::Horizontal;

		Max_Distortion = 5;
		texture_scaling_input = texture_scaling_output = 1;
		down_mouse_x = down_mouse_y = -1;

		Vertex_Energy_color = Highlighted_face_color = RED_COLOR;
		Fixed_vertex_color = Fixed_face_color = BLUE_COLOR;
		Dragged_vertex_color = Dragged_face_color = GREEN_COLOR;
		model_color = GREY_COLOR;
		text_color = BLACK_COLOR;
		
		//update input viewer
		inputCoreID = viewer->core_list[0].id;
		viewer->core(inputCoreID).background_color = Eigen::Vector4f(0.9, 0.9, 0.9, 0);
		viewer->core(inputCoreID).is_animating = true;
		viewer->core(inputCoreID).lighting_factor = 0.5;

		//Load multiple views
		Outputs.push_back(OptimizationOutput(viewer, solver_type,linesearch_type));
		core_size = 1.0 / (Outputs.size() + 1.0);
		
		//maximize window
		glfwMaximizeWindow(viewer->window);
	}
}

void deformation_plugin::load_new_model(const std::string modelpath) {
	modelPath = modelpath;
	if (modelPath.length() != 0)
	{
		modelName = app_utils::ExtractModelName(modelPath);

		stop_solver_thread();
		
		if (model_loaded) {
			//remove previous data
			while (Outputs.size() > 0)
				remove_output(0);

			viewer->load_mesh_from_file(modelPath.c_str());
			viewer->erase_mesh(0);
		}
		else {
			viewer->load_mesh_from_file(modelPath.c_str());
		}

		inputModelID = viewer->data_list[0].id;

		for (int i = 0; i < Outputs.size(); i++)
		{
			viewer->load_mesh_from_file(modelPath.c_str());
			Outputs[i].ModelID = viewer->data_list[i + 1].id;
			initializeSolver(i);
		}

		if (model_loaded) {
			//add new data
			add_output();
		}

		viewer->core(inputCoreID).align_camera_center(InputModel().V, InputModel().F);
		for (int i = 0; i < Outputs.size(); i++)
			viewer->core(Outputs[i].CoreID).align_camera_center(OutputModel(i).V, OutputModel(i).F);
		model_loaded = true;

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
	if (ImGui::Button("Load##Mesh", ImVec2((w - p) / 2.f, 0)))
	{
		//Load new model that has two copies
		modelPath = igl::file_dialog_open();
		isLoadNeeded = true;
	}
	if (isLoadNeeded) {
		load_new_model(modelPath);
		isLoadNeeded = false;
	}
	ImGui::SameLine(0, p);
	if (ImGui::Button("Save##Mesh", ImVec2((w - p) / 2.f, 0)))
	{
		viewer->open_dialog_save_mesh();
	}
			
	if (ImGui::Checkbox("Outputs settings", &Outputs_Settings))
		if(Outputs_Settings)
			show_text = !Outputs_Settings;
	if (ImGui::Checkbox("Show text", &show_text))
		if(show_text)
			Outputs_Settings = !show_text;
	ImGui::Checkbox("Highlight faces", &Highlighted_face);

	if ((view == app_utils::View::Horizontal) || (view == app_utils::View::Vertical)) {
		if(ImGui::SliderFloat("Core Size", &core_size, 0, 1.0/ Outputs.size(), std::to_string(core_size).c_str(), 1)){
			int frameBufferWidth, frameBufferHeight;
			glfwGetFramebufferSize(viewer->window, &frameBufferWidth, &frameBufferHeight);
			post_resize(frameBufferWidth, frameBufferHeight);
		}
	}
	
	if (ImGui::Combo("View", (int *)(&view), app_utils::build_view_names_list(Outputs.size()))) {
		// That's how you get the current width/height of the frame buffer (for example, after the window was resized)
		int frameBufferWidth, frameBufferHeight;
		glfwGetFramebufferSize(viewer->window, &frameBufferWidth, &frameBufferHeight);
		post_resize(frameBufferWidth, frameBufferHeight);
	}

	ImGui::Combo("Mouse Mode", (int *)(&mouse_mode), "NONE\0FACE_SELECT\0VERTEX_SELECT\0CLEAR\0\0");
	if (mouse_mode == app_utils::MouseMode::CLEAR) {
		selected_faces.clear();
		selected_vertices.clear();
		UpdateHandles();
		mouse_mode = app_utils::MouseMode::VERTEX_SELECT;
	}
	ImGui::Checkbox("Update all cores together", &UpdateAll);

	if(model_loaded)
		Draw_menu_for_Solver();
	Draw_menu_for_cores(viewer->core(inputCoreID));
	Draw_menu_for_models(viewer->data(inputModelID));
	Draw_menu_for_output_settings();
	Draw_menu_for_text_results();
	if (model_loaded && solver_settings)
		Draw_menu_for_solver_settings();

	follow_and_mark_selected_faces();
	Update_view();

	if (UpdateAll)
		update_parameters_for_all_cores();

	IsMouseHoveringAnyWindow = false;
	if (ImGui::IsAnyWindowHovered() |
		ImGui::IsRootWindowOrAnyChildHovered() |
		ImGui::IsItemHoveredRect() |
		ImGui::IsMouseHoveringAnyWindow() |
		ImGui::IsMouseHoveringWindow())
		IsMouseHoveringAnyWindow = true;
}

void deformation_plugin::update_parameters_for_all_cores() {
	for (auto& core : viewer->core_list) {
		int output_index = -1;
		for (int i = 0; i < Outputs.size(); i++) {
			if (core.id == Outputs[i].CoreID) {
				output_index = i;
			}
		}
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
	stop_solver_thread();
	
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
	stop_solver_thread();
	Outputs.push_back(OptimizationOutput(viewer, solver_type,linesearch_type));
	
	viewer->load_mesh_from_file(modelPath.c_str());
	Outputs[Outputs.size() - 1].ModelID = viewer->data_list[Outputs.size()].id;
	initializeSolver(Outputs.size() - 1);

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
		if (view == app_utils::View::Horizontal) {
			viewer->core(inputCoreID).viewport = Eigen::Vector4f(0, 0, w - w * Outputs.size() * core_size, h);
			for (int i = 0; i < Outputs.size(); i++) {
				Outputs[i].window_position = ImVec2(w - w * (Outputs.size() - i) * core_size, 0);
				Outputs[i].window_size = ImVec2(w * core_size, h);
				Outputs[i].text_position = Outputs[i].window_position;
			}
		}
		if (view == app_utils::View::Vertical) {
			viewer->core(inputCoreID).viewport = Eigen::Vector4f(0, Outputs.size() * h * core_size, w, h - Outputs.size() * h * core_size);
			for (int i = 0; i < Outputs.size(); i++) {
				Outputs[i].window_position = ImVec2(0, (Outputs.size() - i - 1) * h * core_size);
				Outputs[i].window_size = ImVec2(w, h * core_size);
				Outputs[i].text_position = ImVec2(w*0.8, h - Outputs[i].window_position[1] - Outputs[i].window_size[1]);
			}
		}
		if (view == app_utils::View::InputOnly) {
			viewer->core(inputCoreID).viewport = Eigen::Vector4f(0, 0, w, h);
			for (auto&o : Outputs) {
				o.window_position = ImVec2(w, h);
				o.window_size = ImVec2(0, 0);
				o.text_position = o.window_position;
			}
		}
// 		if (view >= app_utils::View::OutputOnly0) {
// 			viewer->core(inputCoreID).viewport = Eigen::Vector4f(0, 0, 0, 0);
// 			for (auto&o : Outputs) {
// 				o.window_position = ImVec2(w, h);
// 				o.window_size = ImVec2(0, 0);
// 				o.text_position = o.window_position;
// 			}
// 			// what does this means?
// 			Outputs[view - 3/*app_utils::View::OutputOnly0*/].window_position = ImVec2(0, 0);
// 			Outputs[view - 3/*app_utils::View::OutputOnly0*/].window_size = ImVec2(w, h);
// 			Outputs[view - 3/*app_utils::View::OutputOnly0*/].text_position = ImVec2(w*0.8, 0);
// 		}		
		for (auto& o : Outputs)
			viewer->core(o.CoreID).viewport = Eigen::Vector4f(o.window_position[0], o.window_position[1], o.window_size[0]+1, o.window_size[1]+1);
	}
}

IGL_INLINE bool deformation_plugin::mouse_move(int mouse_x, int mouse_y)
{
	if (IsMouseHoveringAnyWindow | IsMouseDraggingAnyWindow)
		return true;

	if (!IsTranslate)
		return false;
	
	if (mouse_mode == app_utils::MouseMode::FACE_SELECT)
	{
		if (!selected_faces.empty())
		{
			Eigen::RowVector3d face_avg_pt = get_face_avg();
			Eigen::RowVector3i face = viewer->data(Model_Translate_ID).F.row(Translate_Index);
			Eigen::Vector3f translation = app_utils::computeTranslation(mouse_x, down_mouse_x, mouse_y, down_mouse_y, face_avg_pt, viewer->core(Core_Translate_ID));
			if (Core_Translate_ID == inputCoreID) {
				viewer->data(Model_Translate_ID).V.row(face[0]) += translation.cast<double>();
				viewer->data(Model_Translate_ID).V.row(face[1]) += translation.cast<double>();
				viewer->data(Model_Translate_ID).V.row(face[2]) += translation.cast<double>();
				viewer->data(Model_Translate_ID).set_mesh(viewer->data(Model_Translate_ID).V, viewer->data(Model_Translate_ID).F);
			}
			else {
				for (auto& out : Outputs) {
					viewer->data(out.ModelID).V.row(face[0]) += translation.cast<double>();
					viewer->data(out.ModelID).V.row(face[1]) += translation.cast<double>();
					viewer->data(out.ModelID).V.row(face[2]) += translation.cast<double>();
					viewer->data(out.ModelID).set_mesh(viewer->data(out.ModelID).V, viewer->data(out.ModelID).F);
				}
			}
			down_mouse_x = mouse_x;
			down_mouse_y = mouse_y;
			UpdateHandles();
			return true;
		}
	}
	else if (mouse_mode == app_utils::MouseMode::VERTEX_SELECT)
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
			UpdateHandles();
			return true;
		}
	}
	UpdateHandles();
	return false;
}

IGL_INLINE bool deformation_plugin::mouse_up(int button, int modifier) {
	IsTranslate = false;
	IsMouseDraggingAnyWindow = false;
	return false;
}

IGL_INLINE bool deformation_plugin::mouse_down(int button, int modifier) {
	if (IsMouseHoveringAnyWindow)
		IsMouseDraggingAnyWindow = true;

	down_mouse_x = viewer->current_mouse_x;
	down_mouse_y = viewer->current_mouse_y;
			
	if (mouse_mode == app_utils::MouseMode::FACE_SELECT && button == GLFW_MOUSE_BUTTON_LEFT && modifier == 2)
	{
		//check if there faces which is selected on the left screen
		int f = pick_face(InputModel().V, InputModel().F, (int)app_utils::View::InputOnly);
		for(int i=0;i<Outputs.size();i++)
			if (f == -1)
				f = pick_face(OutputModel(i).V, OutputModel(i).F, 3/*app_utils::View::OutputOnly0*/+i);
		
		if (f != -1)
		{
			if (find(selected_faces.begin(), selected_faces.end(), f) != selected_faces.end())
			{
				selected_faces.erase(f);
				UpdateHandles();
			}
			else {
				selected_faces.insert(f);
				UpdateHandles();
			}
		}

	}
	else if (mouse_mode == app_utils::MouseMode::VERTEX_SELECT && button == GLFW_MOUSE_BUTTON_LEFT && modifier == 2)
	{
		//check if there faces which is selected on the left screen
		int v = pick_vertex(InputModel().V, InputModel().F, (int)app_utils::View::InputOnly);
		for(int i=0;i<Outputs.size();i++)
			if(v == -1) 
				v = pick_vertex(OutputModel(i).V, OutputModel(i).F, 3/*app_utils::OutputOnly0*/+i);
		
		if (v != -1)
		{
			if (find(selected_vertices.begin(), selected_vertices.end(), v) != selected_vertices.end())
			{
				selected_vertices.erase(v);
				UpdateHandles();
			}
			else {
				selected_vertices.insert(v);
				UpdateHandles();
			}
					
		}
	}
	else if (mouse_mode == app_utils::MouseMode::FACE_SELECT && button == GLFW_MOUSE_BUTTON_MIDDLE)
	{
		if (!selected_faces.empty())
		{
			//check if there faces which is selected on the left screen
			int f = pick_face(InputModel().V, InputModel().F, (int)app_utils::View::InputOnly);
			Model_Translate_ID = inputModelID;
			Core_Translate_ID = inputCoreID;
			for (int i = 0; i < Outputs.size(); i++) {
				if (f == -1) {
					f = pick_face(OutputModel(i).V, OutputModel(i).F, (int)app_utils::View::OutputOnly0 + i);
					Model_Translate_ID = Outputs[i].ModelID;
					Core_Translate_ID = Outputs[i].CoreID;
				}
			}

			if (find(selected_faces.begin(), selected_faces.end(), f) != selected_faces.end())
			{
				IsTranslate = true;
				Translate_Index = f;
			}
		}
	}
	else if (mouse_mode == app_utils::MouseMode::VERTEX_SELECT && button == GLFW_MOUSE_BUTTON_MIDDLE)
	{
	if (!selected_vertices.empty())
	{
		//check if there faces which is selected on the left screen
int v = pick_vertex(InputModel().V, InputModel().F, (int)app_utils::View::InputOnly);
Model_Translate_ID = inputModelID;
Core_Translate_ID = inputCoreID;
for (int i = 0; i < Outputs.size(); i++) {
	if (v == -1) {
		v = pick_vertex(OutputModel(i).V, OutputModel(i).F, (int)app_utils::View::OutputOnly0 + i);
		Model_Translate_ID = Outputs[i].ModelID;
		Core_Translate_ID = Outputs[i].CoreID;
	}
}

if (find(selected_vertices.begin(), selected_vertices.end(), v) != selected_vertices.end())
{
	IsTranslate = true;
	Translate_Index = v;
}
	}
	}

	return false;
}

IGL_INLINE bool deformation_plugin::key_pressed(unsigned int key, int modifiers) {
	if ((key == 'F' || key == 'f') && modifiers == 1)
		mouse_mode = app_utils::MouseMode::FACE_SELECT;
	if ((key == 'V' || key == 'v') && modifiers == 1)
		mouse_mode = app_utils::MouseMode::VERTEX_SELECT;
	if ((key == 'C' || key == 'c') && modifiers == 1)
		mouse_mode = app_utils::MouseMode::CLEAR;
	if ((key == ' ') && modifiers == 1)
		solver_on ? stop_solver_thread() : start_solver_thread();
	if ((key == '!') && modifiers == 1) {
		isLoadNeeded = true;
		modelPath = OptimizationUtils::RDSPath() + "\\models\\Face_1\\Triangle306090degree.obj";
	}
	if ((key == '@') && modifiers == 1) {
		isLoadNeeded = true;
		modelPath = OptimizationUtils::RDSPath() + "\\models\\Face_2\\Triangle2.obj";
	}
	if ((key == '#') && modifiers == 1) {
		isLoadNeeded = true;
		modelPath = OptimizationUtils::RDSPath() + "\\models\\Face_3\\Triangle3.obj";
	}
	if ((key == '$') && modifiers == 1) {
		isLoadNeeded = true;
		modelPath = OptimizationUtils::RDSPath() + "\\models\\Face_4\\Triangle4.obj";
	}
	if ((key == ')') && modifiers == 1) {
		isLoadNeeded = true;
		modelPath = OptimizationUtils::RDSPath() + "\\models\\cube.off";
	}

	if ((key == 'w' || key == 'W') && modifiers == 1) {

	}
	
	return ImGuiMenu::key_pressed(key, modifiers);
}

IGL_INLINE void deformation_plugin::shutdown()
{
	stop_solver_thread();
	ImGuiMenu::shutdown();
}

IGL_INLINE bool deformation_plugin::pre_draw() {
	//call parent function
	ImGuiMenu::pre_draw();

	for (auto& out : Outputs)
		if (out.solver->progressed)
			update_mesh();

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
	return false;
}

void deformation_plugin::Draw_menu_for_colors() {
	ImVec2 screen_pos = ImGui::GetCursorScreenPos();
	if (!ImGui::CollapsingHeader("colors", ImGuiTreeNodeFlags_DefaultOpen))
	{
		ImGui::ColorEdit3("Highlighted face color", Highlighted_face_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
		ImGui::ColorEdit3("Fixed face color", Fixed_face_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
		ImGui::ColorEdit3("Dragged face color", Dragged_face_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
		ImGui::ColorEdit3("Fixed vertex color", Fixed_vertex_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
		ImGui::ColorEdit3("Dragged vertex color", Dragged_vertex_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
		ImGui::ColorEdit3("Model color", model_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
		ImGui::ColorEdit3("Vertex Energy color", Vertex_Energy_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
		ImGui::ColorEdit4("text color", text_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
	}
}

void deformation_plugin::Draw_menu_for_Solver() {
	if (ImGui::CollapsingHeader("Solver", ImGuiTreeNodeFlags_DefaultOpen))
	{
		if (ImGui::Checkbox("step_by_step", &step_by_step) && solver_on) {
			start_solver_thread();
		}
		if (ImGui::Checkbox(solver_on ? "On" : "Off", &solver_on)) {
			solver_on ? start_solver_thread() : stop_solver_thread();
		}

		ImGui::Checkbox("Solver settings", &solver_settings);

		if (ImGui::Combo("step", (int *)(&solver_type), "NEWTON\0Gradient Descent\0Adam Minimizer\0\0")) {
			stop_solver_thread();
			for (int i = 0; i < Outputs.size(); i++) {
				switch (solver_type) {
				case app_utils::SolverType::NEWTON:
					Outputs[i].solver = Outputs[i].newton;
					break;
				case app_utils::SolverType::GRADIENT_DESCENT:
					Outputs[i].solver = Outputs[i].gradient_descent;
					break;
				case app_utils::SolverType::ADAM_MINIMIZER:
					Outputs[i].solver = Outputs[i].adam_minimizer;
					break;
				}
				
				Eigen::MatrixX3i F = OutputModel(i).F;
				Eigen::VectorXd initialguess = Eigen::Map<const Eigen::VectorXd>(OutputModel(i).V.data(), OutputModel(i).V.size());
				Eigen::MatrixX3d normals;
				igl::per_face_normals((Eigen::MatrixX3d)OutputModel(i).V, (Eigen::MatrixX3i)OutputModel(i).F, normals);
				Eigen::VectorXd initNormals = Eigen::Map<const Eigen::VectorXd>(normals.data(), OutputModel(i).F.size());

				Outputs[i].solver->init(
					Outputs[i].totalObjective, 
					initialguess, 
					initNormals, 
					OutputModel(i).F,
					OutputModel(i).V,
					typeAuxVar);
			}
			start_solver_thread();
		}


		ImGui::Combo("init Aux Var", (int *)(&typeAuxVar), "Sphere\0Mesh Center\0\0");


		std::shared_ptr<NewtonSolver> newtonSolver = std::dynamic_pointer_cast<NewtonSolver>(Outputs[0].solver);
		if (newtonSolver != NULL) {
			bool PD = newtonSolver->getPositiveDefiniteChecker();
			ImGui::Checkbox("Positive Definite check", &PD);
			for (auto& o : Outputs) {
				std::dynamic_pointer_cast<NewtonSolver>(o.solver)->SwitchPositiveDefiniteChecker(PD);
			}
		}

		if (ImGui::Combo("line search", (int *)(&linesearch_type), "GradientNorm\0FunctionValue\0ConstantStep\0\0")) {
			for (auto& o:Outputs)
				o.solver->lineSearch_type = linesearch_type;
		}
		if (linesearch_type == OptimizationUtils::LineSearch::ConstantStep && ImGui::DragFloat("Step value", &constant_step, 0.0001f, 0.0f, 1.0f)) {
			for (auto& o : Outputs)
				o.solver->constant_step = constant_step;
		}

		ImGui::Combo("Dist check", (int *)(&distortion_type), "NO_DISTORTION\0AREA_DISTORTION\0LENGTH_DISTORTION\0ANGLE_DISTORTION\0TOTAL_DISTORTION\0\0");
		
		
		float w = ImGui::GetContentRegionAvailWidth(), p = ImGui::GetStyle().FramePadding.x;
		if (ImGui::Button("Check gradients", ImVec2((w - p) / 2.f, 0)))
		{
			checkGradients();
		}
		ImGui::SameLine(0, p);
		if (ImGui::Button("Check Hessians", ImVec2((w - p) / 2.f, 0)))
		{
			checkHessians();
		}
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
		{
			core.align_camera_center(viewer->data_list[data_id].V, viewer->data_list[data_id].F);
		}
		if (ImGui::Button("Snap canonical view", ImVec2(-1, 0)))
		{
			viewer->snap_to_canonical_quaternion();
		}

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
	auto make_checkbox = [&](const char *label, unsigned int &option)
	{
		bool temp = option;
		bool res = ImGui::Checkbox(label, &temp);
		option = temp;
		return res;
	};

	ImGui::PushID(data.id);
	std::stringstream ss;
	if (data.id == inputModelID) {
		ss << modelName;
	}
	else {
		ss << modelName + " " + std::to_string(data.id) + " (Param.)";
	}
			
	if (!ImGui::CollapsingHeader(ss.str().c_str(), ImGuiTreeNodeFlags_DefaultOpen))
	{
		float w = ImGui::GetContentRegionAvailWidth();
		float p = ImGui::GetStyle().FramePadding.x;

		if (data.id == inputModelID) {
			ImGui::SliderFloat("texture", &texture_scaling_input, 0.01, 100, std::to_string(texture_scaling_input).c_str(), 1);
		}
		else {
			ImGui::SliderFloat("texture", &texture_scaling_output, 0.01, 100, std::to_string(texture_scaling_output).c_str(), 1);
		}
			

		if (ImGui::Checkbox("Face-based", &(data.face_based)))
		{
			data.dirty = igl::opengl::MeshGL::DIRTY_ALL;
		}

		make_checkbox("Show texture", data.show_texture);
		if (ImGui::Checkbox("Invert normals", &(data.invert_normals)))
		{
			data.dirty |= igl::opengl::MeshGL::DIRTY_NORMAL;
		}
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

void deformation_plugin::Draw_menu_for_solver_settings() {
	ImGui::SetNextWindowSize(ImVec2(800, 150), ImGuiSetCond_FirstUseEver);
	ImGui::Begin("solver settings", NULL);
	ImGui::SetWindowPos(ImVec2(800, 150), ImGuiSetCond_FirstUseEver);

	//add outputs buttons
	ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.0f, 0.6f, 0.0f, 1.0f));
	if (ImGui::Button("Add Output"))
		add_output();
	ImGui::PopStyleColor();

	int id = 0;
	if (Outputs.size() != 0) {
		// prepare the first column
		ImGui::Columns(Outputs[0].totalObjective->objectiveList.size() + 5, "Unconstrained weights table", true);
		ImGui::Separator();
		ImGui::NextColumn();
		for (auto & obj : Outputs[0].totalObjective->objectiveList) {
			ImGui::Text(obj->name.c_str());
			ImGui::NextColumn();
		}
		ImGui::Text("shift eigen values");
		ImGui::NextColumn();
		ImGui::Text("Remove output");
		ImGui::NextColumn();
		ImGui::Text("copy/paste mesh");
		ImGui::NextColumn();
		ImGui::Text("load param");
		ImGui::NextColumn();
		ImGui::Separator();

		// fill the table
		for (int i = 0; i < Outputs.size();i++) {
			ImGui::Text(("Output " + std::to_string(Outputs[i].CoreID)).c_str());
			ImGui::NextColumn();
			for (auto& obj : Outputs[i].totalObjective->objectiveList) {
				ImGui::PushID(id++);
				ImGui::DragFloat("w", &(obj->w), 0.05f, 0.0f, 100000.0f);
				
				
				std::shared_ptr<BendingEdge> BE = std::dynamic_pointer_cast<BendingEdge>(obj);
				std::shared_ptr<BendingNormal> BN = std::dynamic_pointer_cast<BendingNormal>(obj);
				std::shared_ptr<AuxBendingNormal> ABN = std::dynamic_pointer_cast<AuxBendingNormal>(obj);
				std::shared_ptr<AuxSpherePerHinge> AS = std::dynamic_pointer_cast<AuxSpherePerHinge>(obj);
				if (BE != NULL)
					ImGui::Combo("Function", (int *)(&(BE->functionType)), "Quadratic\0Exponential\0Planar\0\0");
				if (BN != NULL)
					ImGui::Combo("Function", (int *)(&(BN->functionType)), "Quadratic\0Exponential\0Planar\0\0");
				if (ABN != NULL)
					ImGui::Combo("Function", (int *)(&(ABN->functionType)), "Quadratic\0Exponential\0Planar\0\0");
				if (AS != NULL)
					ImGui::Combo("Function", (int *)(&(AS->functionType)), "Quadratic\0Exponential\0Planar\0\0");

				if (BE != NULL && BE->functionType == OptimizationUtils::PlanarL) {
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
				if (BN != NULL && BN->functionType == OptimizationUtils::PlanarL) {
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
				if (ABN != NULL && ABN->functionType == OptimizationUtils::PlanarL) {
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
					ImGui::DragFloat("w1", &(ABN->w1), 0.05f, 0.0f, 100000.0f);
					ImGui::DragFloat("w2", &(ABN->w2), 0.05f, 0.0f, 100000.0f);
					ImGui::DragFloat("w3", &(ABN->w3), 0.05f, 0.0f, 100000.0f);
				}
				if (AS != NULL && AS->functionType == OptimizationUtils::PlanarL) {
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
					ImGui::DragFloat("w1", &(AS->w1), 0.05f, 0.0f, 100000.0f);
					ImGui::DragFloat("w2", &(AS->w2), 0.05f, 0.0f, 100000.0f);
				}
				ImGui::NextColumn();
				ImGui::PopID();
			}
			ImGui::PushID(id++);
			ImGui::DragFloat("", &(Outputs[i].totalObjective->Shift_eigen_values), 0.05f, 0.0f, 100000.0f);
			ImGui::NextColumn();
			if (Outputs.size() > 1)
			{
				ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.6f, 0.0f, 0.0f, 1.0f));
				if (ImGui::Button("Remove"))
					remove_output(i);
				ImGui::PopStyleColor();
				ImGui::NextColumn();
				ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.0f, 0.6f, 0.0f, 1.0f));
				if (!copy_vertices.size() && ImGui::Button("Copy")) 
					copy_vertices = OutputModel(i).V;
				if (copy_vertices.size() && ImGui::Button("Paste")) {
					OutputModel(i).set_vertices(copy_vertices);
					copy_vertices.resize(0,0);
				}
				ImGui::PopStyleColor();
			}
			else
				ImGui::NextColumn();
			ImGui::NextColumn();
			ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.0f, 0.6f, 0.0f, 1.0f));
			if (ImGui::Button("load param")) {
				viewer->load_mesh_from_file(igl::file_dialog_open().c_str());
				Eigen::MatrixXd V = viewer->data_list[viewer->data_list.size() - 1].V;
				viewer->erase_mesh(viewer->data_list.size() - 1);
				OutputModel(i).set_vertices(V);
			}
			ImGui::PopStyleColor();

			ImGui::NextColumn();
			ImGui::PopID();
			ImGui::Separator();
		}
		ImGui::Columns(1);
	}
	
	ImGui::Spacing();

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
	

	//close the window
	ImGui::End();
}

void deformation_plugin::Draw_menu_for_output_settings() {
	for (auto& out : Outputs) {
		if (Outputs_Settings) {
			ImGui::SetNextWindowSize(ImVec2(200, 300), ImGuiSetCond_FirstUseEver);
			ImGui::Begin(("Output settings " + std::to_string(out.CoreID)).c_str(),
				NULL, 
				ImGuiWindowFlags_NoTitleBar |
				ImGuiWindowFlags_NoResize |
				ImGuiWindowFlags_NoMove);
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

void deformation_plugin::UpdateHandles() {
	std::vector<int> CurrHandlesInd;
	std::vector<Eigen::MatrixX3d> CurrHandlesPosDeformed;
	CurrHandlesInd.clear();

	//First, we push each vertices index to the handles
	for (auto vi : selected_vertices) {
		CurrHandlesInd.push_back(vi);
	}
	//Then, we push each face vertices index to the handle (3 vertices)
	for (auto fi : selected_faces) {
		//Here we get the 3 vertice's index that build each face
		int v0 = InputModel().F(fi,0);
		int v1 = InputModel().F(fi,1);
		int v2 = InputModel().F(fi,2);

		//check whether the handle already exist
		if (!(find(CurrHandlesInd.begin(), CurrHandlesInd.end(), v0) != CurrHandlesInd.end()))
			CurrHandlesInd.push_back(v0);
		if (!(find(CurrHandlesInd.begin(), CurrHandlesInd.end(), v1) != CurrHandlesInd.end())) 
			CurrHandlesInd.push_back(v1);
		if (!(find(CurrHandlesInd.begin(), CurrHandlesInd.end(), v2) != CurrHandlesInd.end())) 
			CurrHandlesInd.push_back(v2);	
	}	
	//Here we update the positions for each handle
	for (auto& out :Outputs)
		CurrHandlesPosDeformed.push_back(Eigen::MatrixX3d::Zero(CurrHandlesInd.size(),3));
	
	for (int i = 0; i < Outputs.size(); i++){
		int idx = 0;
		for (auto hi : CurrHandlesInd)
			CurrHandlesPosDeformed[i].row(idx++) << OutputModel(i).V(hi, 0), OutputModel(i).V(hi, 1), OutputModel(i).V(hi, 2);
		//Update texture
		update_texture(OutputModel(i).V, i);
	}
	//Finally, we update the handles in the constraints positional object
	for (int i = 0; i < Outputs.size();i++) {
		if (model_loaded) {
			(*Outputs[i].HandlesInd) = CurrHandlesInd;
			(*Outputs[i].HandlesPosDeformed) = CurrHandlesPosDeformed[i];
		}
	}
}

void deformation_plugin::Update_view() {
	
	for (auto& data : viewer->data_list)
		for (auto& out : Outputs) {
			data.copy_options(viewer->core(inputCoreID), viewer->core(out.CoreID));
			
		}
	
	for (auto& core : viewer->core_list)
		for (auto& data : viewer->data_list)
			viewer->data(data.id).set_visible(false, core.id);

	InputModel().set_visible(true, inputCoreID);
	for (int i = 0; i < Outputs.size(); i++)
		OutputModel(i).set_visible(true, Outputs[i].CoreID);
}

void deformation_plugin::follow_and_mark_selected_faces() {
	//check if there faces which is selected on the left screen
	int f = pick_face(InputModel().V, InputModel().F, (int) app_utils::View::InputOnly);
	for(int i=0;i<Outputs.size();i++)
		if (f == -1) 
			f = pick_face(OutputModel(i).V, OutputModel(i).F, (int) app_utils::View::OutputOnly0+i);
	
	
	if(InputModel().F.size()){
		//Mark the faces
		for (int i = 0; i < Outputs.size(); i++) {
			Outputs[i].color_per_face.resize(InputModel().F.rows(), 3);
			UpdateEnergyColors(i);

			//Mark the highlighted face
			if (f != -1 && Highlighted_face)
				Outputs[i].color_per_face.row(f) = Highlighted_face_color.cast<double>();
			//Mark the fixed faces
			for (auto fi : selected_faces)
				Outputs[i].color_per_face.row(fi) = Fixed_face_color.cast<double>(); 
			//Mark the Dragged face
			if (IsTranslate && (mouse_mode == app_utils::MouseMode::FACE_SELECT))
				Outputs[i].color_per_face.row(Translate_Index) = Dragged_face_color.cast<double>();
			//Mark the vertices
			int idx = 0;
			Vertices_Input.resize(selected_vertices.size(), 3);
			Outputs[i].Vertices_output.resize(selected_vertices.size(), 3);
			color_per_vertex.resize(selected_vertices.size(), 3);
			//Mark the dragged vertex
			if (IsTranslate && (mouse_mode == app_utils::MouseMode::VERTEX_SELECT)) {
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

Eigen::RowVector3d deformation_plugin::get_face_avg() {
	Eigen::RowVector3d avg; avg << 0, 0, 0;
	Eigen::RowVector3i face = viewer->data(Model_Translate_ID).F.row(Translate_Index);

	avg += viewer->data(Model_Translate_ID).V.row(face[0]);
	avg += viewer->data(Model_Translate_ID).V.row(face[1]);
	avg += viewer->data(Model_Translate_ID).V.row(face[2]);
	avg /= 3;

	return avg;
}

int deformation_plugin::pick_face(Eigen::MatrixXd& V, Eigen::MatrixXi& F, int CoreIndex) {
	// Cast a ray in the view direction starting from the mouse position
	int CoreID;
	if (CoreIndex == (int)app_utils::View::InputOnly)
		CoreID = inputCoreID;
	else
		CoreID = Outputs[CoreIndex - (int)app_utils::View::OutputOnly0].CoreID;
	 
	double x = viewer->current_mouse_x;
	double y = viewer->core(CoreID).viewport(3) - viewer->current_mouse_y;
	if (view == app_utils::View::Vertical) {
		y = (viewer->core(inputCoreID).viewport(3) / core_size) - viewer->current_mouse_y;
	}

	//Eigen::RowVector3d pt;
	Eigen::Matrix<double, 3, 1, 0, 3, 1> pt;

	Eigen::Matrix4f modelview = viewer->core(CoreID).view;
	int vi = -1;

	std::vector<igl::Hit> hits;

	igl::unproject_in_mesh(Eigen::Vector2f(x, y), viewer->core(CoreID).view,
		viewer->core(CoreID).proj, viewer->core(CoreID).viewport, V, F, pt, hits);

	int fi = -1;
	if (hits.size() > 0) {
		fi = hits[0].id;
	}
	return fi;
}

int deformation_plugin::pick_vertex(Eigen::MatrixXd& V, Eigen::MatrixXi& F, int CoreIndex) {
	// Cast a ray in the view direction starting from the mouse position
	int CoreID;
	if (CoreIndex == (int)app_utils::View::InputOnly)
		CoreID = inputCoreID;
	else
		CoreID = Outputs[CoreIndex - (int)app_utils::View::OutputOnly0].CoreID;
	

	double x = viewer->current_mouse_x;
	double y = viewer->core(CoreID).viewport(3) - viewer->current_mouse_y;
	if (view == app_utils::View::Vertical) {
		y = (viewer->core(inputCoreID).viewport(3) / core_size) - viewer->current_mouse_y;
	}

	//Eigen::RowVector3d pt;
	Eigen::Matrix<double, 3, 1, 0, 3, 1> pt;

	Eigen::Matrix4f modelview = viewer->core(CoreID).view;
	int vi = -1;

	std::vector<igl::Hit> hits;
			
	unproject_in_mesh(Eigen::Vector2f(x, y), viewer->core(CoreID).view,
		viewer->core(CoreID).proj, viewer->core(CoreID).viewport, V, F, pt, hits);

	if (hits.size() > 0) {
		int fi = hits[0].id;
		Eigen::RowVector3d bc;
		bc << 1.0 - hits[0].u - hits[0].v, hits[0].u, hits[0].v;
		bc.maxCoeff(&vi);
		vi = F(fi, vi);
	}
	return vi;
}

void deformation_plugin::update_texture(Eigen::MatrixXd& V_uv, const int index) {
	Eigen::MatrixXd V_uv_2D(V_uv.rows(),2);
	Eigen::MatrixXd V_uv_3D(V_uv.rows(),3);
	if (V_uv.cols() == 2) {
		V_uv_2D = V_uv;
		V_uv_3D.leftCols(2) = V_uv.leftCols(2);
		V_uv_3D.rightCols(1).setZero();
	}
	else if (V_uv.cols() == 3) {
		V_uv_3D = V_uv;
		V_uv_2D = V_uv.leftCols(2);
	}

	// Plot the mesh
	if(index == 0)
		InputModel().set_uv(V_uv_2D * texture_scaling_input);
	OutputModel(index).set_vertices(V_uv_3D);
	OutputModel(index).set_uv(V_uv_2D * texture_scaling_output);
	OutputModel(index).compute_normals();
}
	
void deformation_plugin::checkGradients()
{
	stop_solver_thread();
	for (int i = 0; i < Outputs.size(); i++) {
		if (!model_loaded) {
			solver_on = false;
			return;
		}
		Eigen::VectorXd xx = Eigen::VectorXd::Random(InputModel().V.size() + 7*InputModel().F.rows());
		
		for (auto const &objective : Outputs[i].totalObjective->objectiveList) {
			objective->checkGradient(xx);
		}
	}
}

void deformation_plugin::checkHessians()
{
	stop_solver_thread();
	for (int i = 0; i < Outputs.size(); i++) {
		if (!model_loaded) {
			solver_on = false;
			return;
		}
		Eigen::VectorXd xx = Eigen::VectorXd::Random(InputModel().V.size() + 7*InputModel().F.rows());
		
		for (auto const &objective : Outputs[i].totalObjective->objectiveList) {
			objective->checkHessian(xx);
		}
	}
}

void deformation_plugin::update_mesh()
{
	std::vector<Eigen::MatrixXd> V;
	std::vector<Eigen::VectorXd> X; X.resize(Outputs.size());
	
	for (int i = 0; i < Outputs.size(); i++){
		Outputs[i].solver->get_data(X[i]);
		
		V.push_back(Eigen::Map<Eigen::MatrixXd>(
			X[i].data(), 
			InputModel().V.rows(), 
			InputModel().V.cols())
		);

		if (IsTranslate && mouse_mode == app_utils::MouseMode::VERTEX_SELECT)
			V[i].row(Translate_Index) = OutputModel(i).V.row(Translate_Index);
		else if(IsTranslate && mouse_mode == app_utils::MouseMode::FACE_SELECT) {
			Eigen::Vector3i F = OutputModel(i).F.row(Translate_Index);
			for (int vi = 0; vi < 3; vi++)
				V[i].row(F[vi]) = OutputModel(i).V.row(F[vi]);
		}
		update_texture(V[i],i);
	}
}

void deformation_plugin::stop_solver_thread() {
	solver_on = false;
	for (auto&o : Outputs) {
		if (o.solver->is_running) {
			o.solver->stop();
		}
		while (o.solver->is_running);
	}
}

void deformation_plugin::start_solver_thread() {
	if (!model_loaded) {
		solver_on = false;
		return;
	}
	stop_solver_thread();
	for (int i = 0; i < Outputs.size();i++) {
		std::cout << ">> A new solver has been started" << std::endl;
		solver_on = true;
		//update solver
		Eigen::VectorXd init = Eigen::Map<const Eigen::VectorXd>(OutputModel(i).V.data(), OutputModel(i).V.size());
		Eigen::MatrixX3d normals;
		igl::per_face_normals((Eigen::MatrixX3d)OutputModel(i).V, (Eigen::MatrixX3i)OutputModel(i).F, normals);
		Eigen::VectorXd initNormals = Eigen::Map<const Eigen::VectorXd>(normals.data(), OutputModel(i).F.size());
		Outputs[i].newton->init(
			Outputs[i].totalObjective, 
			init,
			initNormals,
			OutputModel(i).F, 
			OutputModel(i).V,
			typeAuxVar
		);
		Outputs[i].gradient_descent->init(
			Outputs[i].totalObjective, 
			init, 
			initNormals,
			OutputModel(i).F, 
			OutputModel(i).V,
			typeAuxVar
		);
		Outputs[i].adam_minimizer->init(
			Outputs[i].totalObjective,
			init,
			initNormals,
			OutputModel(i).F,
			OutputModel(i).V,
			typeAuxVar
		);
		//start solver
		if (step_by_step) {
			static int step_counter = 0;
			solver_thread = std::thread(&solver::run_one_iteration, Outputs[i].solver.get(), step_counter++,true);
			solver_thread.join();
		}
		else {
			solver_thread = std::thread(&solver::run, Outputs[i].solver.get());
			solver_thread.detach();
		}
	}
}

void deformation_plugin::initializeSolver(const int index)
{
	Eigen::MatrixXd V = OutputModel(index).V;
	Eigen::MatrixX3i F = OutputModel(index).F;
	
	stop_solver_thread();

	if (V.rows() == 0 || F.rows() == 0)
		return;

	// initialize the energy
	std::cout << console_color::yellow << "-------Energies, begin-------" << std::endl;
	auto bendingEdge = std::make_unique<BendingEdge>(OptimizationUtils::PlanarL);
	bendingEdge->init_mesh(V, F);
	bendingEdge->init();
	auto auxBendingNormal = std::make_unique<AuxBendingNormal>(OptimizationUtils::PlanarL);
	auxBendingNormal->init_mesh(V, F);
	auxBendingNormal->init();
	auto auxSpherePerHinge = std::make_unique<AuxSpherePerHinge>(OptimizationUtils::PlanarL);
	auxSpherePerHinge->init_mesh(V, F);
	auxSpherePerHinge->init();
	auto bendingNormal = std::make_unique<BendingNormal>(OptimizationUtils::PlanarL);
	bendingNormal->init_mesh(V, F);
	bendingNormal->init();
	auto SymmDirich = std::make_unique<SymmetricDirichlet>();
	SymmDirich->init_mesh(V, F);
	SymmDirich->init();
	auto stvk = std::make_unique<STVK>();
	if (app_utils::IsMesh2D(InputModel().V)) {
		stvk->init_mesh(V, F);
		stvk->init();
	}
	auto allVertexPositions = std::make_unique<AllVertexPositions>();
	allVertexPositions->init_mesh(V, F);
	allVertexPositions->init();
	auto constraintsPositional = std::make_shared<PenaltyPositionalConstraints>();
	constraintsPositional->numV = V.rows();
	constraintsPositional->numF = F.rows();
	constraintsPositional->init();

	Outputs[index].HandlesInd = &constraintsPositional->ConstrainedVerticesInd;
	Outputs[index].HandlesPosDeformed = &constraintsPositional->ConstrainedVerticesPos;

	Outputs[index].totalObjective->objectiveList.clear();
	Outputs[index].totalObjective->init_mesh(V, F);
	Outputs[index].totalObjective->objectiveList.push_back(move(auxSpherePerHinge));
	Outputs[index].totalObjective->objectiveList.push_back(move(auxBendingNormal));
	Outputs[index].totalObjective->objectiveList.push_back(move(bendingNormal));
	Outputs[index].totalObjective->objectiveList.push_back(move(bendingEdge));
	Outputs[index].totalObjective->objectiveList.push_back(move(SymmDirich));
	if(app_utils::IsMesh2D(InputModel().V))
		Outputs[index].totalObjective->objectiveList.push_back(move(stvk));
	Outputs[index].totalObjective->objectiveList.push_back(move(allVertexPositions));
	Outputs[index].totalObjective->objectiveList.push_back(move(constraintsPositional));
	Outputs[index].totalObjective->init();
	std::cout  << "-------Energies, end-------" << console_color::white << std::endl;


	// initialize the solver
	Eigen::VectorXd init = Eigen::Map<const Eigen::VectorXd>(V.data(), V.size());
	Eigen::MatrixX3d normals;
	igl::per_face_normals((Eigen::MatrixX3d)V, (Eigen::MatrixX3i)F, normals);
	Eigen::VectorXd initNormals = Eigen::Map<const Eigen::VectorXd>(normals.data(), F.size());

	Outputs[index].newton->init(
		Outputs[index].totalObjective, 
		init, 
		initNormals, 
		OutputModel(index).F, 
		OutputModel(index).V,
		typeAuxVar);
	Outputs[index].gradient_descent->init(
		Outputs[index].totalObjective, 
		init, 
		initNormals, 
		OutputModel(index).F, 
		OutputModel(index).V,
		typeAuxVar);
	Outputs[index].adam_minimizer->init(
		Outputs[index].totalObjective, 
		init, 
		initNormals, 
		OutputModel(index).F, 
		OutputModel(index).V,
		typeAuxVar);
	
	std::cout << ">> Solver is initialized!" << std::endl;
}

void deformation_plugin::UpdateEnergyColors(const int index) {
	int numF = OutputModel(index).F.rows();
	Eigen::VectorXd DistortionPerFace(numF);
	DistortionPerFace.setZero();
	
	if (distortion_type == app_utils::Distortion::ANGLE_DISTORTION) {	//distortion according to area preserving
		Eigen::MatrixXd angle_input, angle_output, angle_ratio;
		app_utils::angle_degree(OutputModel(index).V, OutputModel(index).F, angle_output);
		app_utils::angle_degree(InputModel().V, InputModel().F, angle_input);
		// DistortionPerFace = angle_output / angle_input
		angle_ratio = angle_input.cwiseInverse().cwiseProduct(angle_output);
		// Becuase we want  DistortionPerFace to be as colse as possible to zero instead of one!
		angle_ratio = (angle_ratio - Eigen::MatrixXd::Ones(numF, 3)).cwiseAbs2();
		//sum over the vertices on each face
		DistortionPerFace = angle_ratio.rowwise().sum();
	}
	else if (distortion_type == app_utils::Distortion::LENGTH_DISTORTION) {	//distortion according to area preserving
		Eigen::MatrixXd Length_output, Length_input, Length_ratio;
		igl::edge_lengths(OutputModel(index).V, OutputModel(index).F, Length_output);
		igl::edge_lengths(InputModel().V, InputModel().F, Length_input);
		// DistortionPerFace = Length_output / Length_input
		Length_ratio = Length_input.cwiseInverse().cwiseProduct(Length_output);
		// Becuase we want  DistortionPerFace to be as colse as possible to zero instead of one!
		Length_ratio = (Length_ratio - Eigen::MatrixXd::Ones(numF,3)).cwiseAbs2();
		//sum over the vertices on each face
		DistortionPerFace = Length_ratio.rowwise().sum();
	}
	else if (distortion_type == app_utils::Distortion::AREA_DISTORTION) {
		//distortion according to area preserving
		Eigen::VectorXd Area_output, Area_input;
		igl::doublearea(OutputModel(index).V, OutputModel(index).F, Area_output);
		igl::doublearea(InputModel().V, InputModel().F, Area_input);
		// DistortionPerFace = Area_output / Area_input
		DistortionPerFace = Area_input.cwiseInverse().cwiseProduct(Area_output);
		// Because we want  DistortionPerFace to be as close as possible to zero instead of one!
		DistortionPerFace = (DistortionPerFace - Eigen::VectorXd::Ones(numF)).cwiseAbs2();
	}
	else if (distortion_type == app_utils::Distortion::TOTAL_DISTORTION) {
		// calculate the distortion over all the energies
		for (auto& obj : Outputs[index].totalObjective->objectiveList)
			if ((obj->Efi.size() != 0) && (obj->w != 0)) 
				DistortionPerFace += obj->Efi * obj->w;
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
