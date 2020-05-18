cmake_minimum_required(VERSION 3.1)
# set the project name
project(main_app)

# Add your project files
set(SOURCES src/main.cpp ${CMAKE_SOURCE_DIR}/natvis/eigen.natvis)
add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME}
	PRIVATE
		${CMAKE_SOURCE_DIR})

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        igl::core
		rds::optimization_lib
		rds::plugins
        igl::opengl
        igl::opengl_glfw
        igl::opengl_glfw_imgui
		)

if(MKL_FOUND)
	set(EIGEN_USE_MKL_ALL)
	set(MKL_LP64)
	message("MKL found. MKL Include dir: " ${MKL_INCLUDE_DIR})
	include_directories( ${MKL_INCLUDE_DIR})
	target_link_libraries(${PROJECT_NAME} PRIVATE ${MKL_LIBRARIES})
endif()

if(LIBIGL_WITH_MATLAB)
	target_link_libraries(${PROJECT_NAME} PRIVATE igl::matlab)
endif()