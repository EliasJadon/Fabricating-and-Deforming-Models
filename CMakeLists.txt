cmake_minimum_required(VERSION 3.5)
# set the project name
project(MappingsLab)

# set variable value
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(MKL)

# libigl
option(LIBIGL_WITH_OPENGL            "Use OpenGL"                    ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"                      ON)
option(LIBIGL_WITH_OPENGL_GLFW_IMGUI "Use ImGui"                     ON)
option(LIBIGL_WITH_MATLAB 			 "Use MATLAB"                    ON)
option(BUILD_TESTS 			 "Build tests"                    OFF)
find_package(LIBIGL REQUIRED QUIET)

# libraries and plugins
add_subdirectory("libs/optimization_lib")
add_subdirectory("plugins")

# applications
add_subdirectory("apps")
if(BUILD_TESTS)
	add_subdirectory("tests")
endif()