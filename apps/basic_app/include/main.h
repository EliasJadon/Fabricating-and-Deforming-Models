#include <igl/opengl/glfw/Viewer.h>
#include <GLFW/glfw3.h>
#include <string>
#include <iostream>
#include <map>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>

#define COW_PATH "D:\\OneDrive - University of Haifa\\Thesis\\Autoquads\\models\\cow.obj"

int main(int argc, char * argv[])
{
	igl::opengl::glfw::Viewer viewer;

	viewer.load_mesh_from_file(std::string(COW_PATH));
	viewer.load_mesh_from_file(std::string(COW_PATH));

	unsigned int left_view, right_view;
	int cube_id = viewer.data_list[0].id;
	int sphere_id = viewer.data_list[1].id;
	viewer.callback_init = [&](igl::opengl::glfw::Viewer &)
	{
		viewer.core().viewport = Eigen::Vector4f(0, 0, 640, 800);
		left_view = viewer.core_list[0].id;
		right_view = viewer.append_core(Eigen::Vector4f(640, 0, 640, 800));
		return false;
	};

	viewer.callback_key_down = [&](igl::opengl::glfw::Viewer &, unsigned int key, int mod)
	{
		if (key == GLFW_KEY_SPACE)
		{
			// By default, when a core is appended, all loaded meshes will be displayed in that core.
			// Displaying can be controlled by calling viewer.data().set_visible().
			viewer.data(cube_id).set_visible(false, left_view);
			viewer.data(sphere_id).set_visible(false, right_view);
		}
		return false;
	};

	viewer.callback_post_resize = [&](igl::opengl::glfw::Viewer &v, int w, int h) {
		v.core(left_view).viewport = Eigen::Vector4f(0, 0, w / 2, h);
		v.core(right_view).viewport = Eigen::Vector4f(w / 2, 0, w - (w / 2), h);
		return true;
	};







	// Attach a menu plugin
	igl::opengl::glfw::imgui::ImGuiMenu menu;
	viewer.plugins.push_back(&menu);

	// Customize the menu
	double doubleVariable = 0.1f; // Shared between two menus

	// Add content to the default menu window
	menu.callback_draw_viewer_menu = [&]()
	{
		// Draw parent menu content
		menu.draw_viewer_menu();

		// Add new group
		if (ImGui::CollapsingHeader("New Group", ImGuiTreeNodeFlags_DefaultOpen))
		{
			// Expose variable directly ...
			ImGui::InputDouble("double", &doubleVariable, 0, 0, "%.4f");

			// ... or using a custom callback
			static bool boolVariable = true;
			if (ImGui::Checkbox("bool", &boolVariable))
			{
				// do something
				std::cout << "boolVariable: " << std::boolalpha << boolVariable << std::endl;
			}

			// Expose an enumeration type
			enum Orientation { Up = 0, Down, Left, Right };
			static Orientation dir = Up;
			ImGui::Combo("Direction", (int *)(&dir), "Up\0Down\0Left\0Right\0\0");

			// We can also use a std::vector<std::string> defined dynamically
			static int num_choices = 3;
			static std::vector<std::string> choices;
			static int idx_choice = 0;
			if (ImGui::InputInt("Num letters", &num_choices))
			{
				num_choices = std::max(1, std::min(26, num_choices));
			}
			if (num_choices != (int)choices.size())
			{
				choices.resize(num_choices);
				for (int i = 0; i < num_choices; ++i)
					choices[i] = std::string(1, 'A' + i);
				if (idx_choice >= num_choices)
					idx_choice = num_choices - 1;
			}
			ImGui::Combo("Letter", &idx_choice, choices);

			// Add a button
			if (ImGui::Button("Print Hello", ImVec2(-1, 0)))
			{
				std::cout << "Hello\n";
			}
		}
	};

	// Draw additional windows
	menu.callback_draw_custom_window = [&]()
	{
		// Define next window position + size
		ImGui::SetNextWindowPos(ImVec2(180.f * menu.menu_scaling(), 10), ImGuiSetCond_FirstUseEver);
		ImGui::SetNextWindowSize(ImVec2(200, 160), ImGuiSetCond_FirstUseEver);
		ImGui::Begin(
			"New Window", nullptr,
			ImGuiWindowFlags_NoSavedSettings
		);

		// Expose the same variable directly ...
		ImGui::PushItemWidth(-80);
		ImGui::DragScalar("double", ImGuiDataType_Double, &doubleVariable, 0.1, 0, 0, "%.4f");
		ImGui::PopItemWidth();

		static std::string str = "bunny";
		ImGui::InputText("Name", str);

		ImGui::End();
	};





	viewer.launch();
	return EXIT_SUCCESS;
}
