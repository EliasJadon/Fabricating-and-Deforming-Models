cmake_minimum_required(VERSION 3.1)
# set the project name
project(plugins)

# set variable value
set(SOURCES
	deformation_plugin/src/deformation_plugin.cpp
	deformation_plugin/include/deformation_plugin.h
	deformation_plugin/include/app_utils.h)

add_library(${PROJECT_NAME} ${SOURCES})
add_library(rds::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME}
	PRIVATE
		${CMAKE_SOURCE_DIR}
		${PROJECT_SOURCE_DIR}
		${CMAKE_SOURCE_DIR}/libs/
		)

target_link_libraries(${PROJECT_NAME}
	PRIVATE
		igl::core
		igl::opengl
		igl::opengl_glfw
		igl::opengl_glfw_imgui
		rds::optimization_lib)		

if(MKL_FOUND)
	set(EIGEN_USE_MKL_ALL)
	set(MKL_LP64)
	message("MKL found. MKL Include dir: " ${MKL_INCLUDE_DIR})
	include_directories( ${MKL_INCLUDE_DIR})
	target_link_libraries(${PROJECT_NAME} PRIVATE ${MKL_LIBRARIES})
endif()

if(LIB_WITH_MATLAB)
	target_link_libraries(${PROJECT_NAME} PRIVATE igl::matlab)
endif()