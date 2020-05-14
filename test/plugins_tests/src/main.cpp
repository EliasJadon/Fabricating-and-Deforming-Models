#include <plugins/basic_app/include/basic_app.h>

int main(int argc, char * argv[])
{
	igl::opengl::glfw::Viewer viewer;
	basic_app menu;
	viewer.plugins.push_back(&menu);
	viewer.launch();
	return EXIT_SUCCESS;
}
