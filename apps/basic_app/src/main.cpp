#include <plugins/deformation_plugin/include/deformation_plugin.h>

int main(int argc, char * argv[])
{
	igl::opengl::glfw::Viewer viewer;
	deformation_plugin plugin;
	viewer.plugins.push_back(&plugin);
	viewer.launch();
	return EXIT_SUCCESS;
}
