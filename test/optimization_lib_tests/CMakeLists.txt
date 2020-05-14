cmake_minimum_required(VERSION 3.1)
# set the project name
project(optimization_lib_tests)

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
		igl::matlab)
		
set_target_properties(${PROJECT_NAME} 
    PROPERTIES
    	VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/ext/worhp/bin")