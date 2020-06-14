#pragma once

#ifndef RDS_PLUGINS_BASIC_MENU_H
#define RDS_PLUGINS_BASIC_MENU_H

#include "app_utils.h"

class deformation_plugin : public igl::opengl::glfw::imgui::ImGuiMenu
{
private:
	std::vector<Eigen::MatrixXd> p_edges;
	OptimizationUtils::InitAuxVariables typeAuxVar;
	bool step_by_step;
	bool isLoadNeeded;
	bool model_loaded;
	float Max_Distortion;
	bool UpdateAll;
	bool worhp_on, solver_on, solver_settings, IsMouseHoveringAnyWindow, IsMouseDraggingAnyWindow;
	app_utils::Distortion distortion_type;
	app_utils::SolverType solver_type;
	OptimizationUtils::LineSearch linesearch_type;
	float constant_step;
	Eigen::MatrixXd Vertices_Input, color_per_vertex;
	Eigen::Vector3f
		Highlighted_face_color,
		Fixed_face_color,
		Fixed_vertex_color,
		model_color,
		Dragged_face_color,
		Dragged_vertex_color,
		Vertex_Energy_color,
		text_color;
	bool show_text;
	float core_size;
	float texture_scaling_output;
	bool Highlighted_face, Outputs_Settings;
	std::set<int> selected_faces, selected_vertices;
	std::vector<OptimizationOutput> Outputs;
	Eigen::MatrixXd copy_vertices;
	//Basic (necessary) parameteres
	float prev_camera_zoom;
	Eigen::Vector3f prev_camera_translation;
	Eigen::Quaternionf prev_trackball_angle;
	std::string modelName, modelPath;
	int inputCoreID, inputModelID;
	app_utils::View view;
	app_utils::MouseMode mouse_mode;
	
	float texture_scaling_input;
	
	bool IsTranslate;
	int Translate_Index, Model_Translate_ID, Core_Translate_ID, down_mouse_x, down_mouse_y;
	ImGuiMenu menu;

	// Solver thread
	std::thread solver_thread;

public:
	//Constructor & initialization
	deformation_plugin();
	~deformation_plugin(){}

	// callbacks
	IGL_INLINE virtual void draw_viewer_menu() override;
	IGL_INLINE virtual void init(igl::opengl::glfw::Viewer *_viewer) override;
	IGL_INLINE virtual void post_resize(int w, int h) override;
	IGL_INLINE virtual bool mouse_move(int mouse_x, int mouse_y) override;
	IGL_INLINE virtual bool mouse_down(int button, int modifier) override;
	IGL_INLINE virtual bool mouse_up(int button, int modifier) override;
	IGL_INLINE virtual bool pre_draw() override;
	IGL_INLINE virtual void shutdown() override;
	IGL_INLINE virtual bool key_pressed(unsigned int key, int modifiers) override;
			
	//Draw menu methods
	void Draw_menu_for_cores(igl::opengl::ViewerCore& core);
	void Draw_menu_for_models(igl::opengl::ViewerData& data);
	void Draw_menu_for_Solver();
	void Draw_menu_for_solver_settings();
	void Draw_menu_for_output_settings();
	void Draw_menu_for_colors();
	void Draw_menu_for_text_results();

	//Pick faces & vertices and highlight them
	int pick_face(Eigen::MatrixXd& V, Eigen::MatrixXi& F, int LR);
	int pick_vertex(Eigen::MatrixXd& V, Eigen::MatrixXi& F, int LR);
	void follow_and_mark_selected_faces();
	Eigen::RowVector3d get_face_avg();
	void UpdateHandles();
	void UpdateEnergyColors(const int index);
	void update_parameters_for_all_cores();

	//Basic Methods
	igl::opengl::ViewerData& InputModel();
	igl::opengl::ViewerData& OutputModel(const int index);

	void load_new_model(const std::string modelpath);
	void Update_view();
	void update_mesh();
	void update_texture(Eigen::MatrixXd& V_uv, const int index);

	//Start/Stop the solver Thread
	void initializeSolver(const int index);
	void stop_solver_thread();
	void start_solver_thread();

	//FD check
	void checkGradients();
	void checkHessians();

	//outputs
	void add_output();
	void remove_output(const int output_index);
};

#endif