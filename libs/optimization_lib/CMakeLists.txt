cmake_minimum_required(VERSION 3.5)
# set the project name
project(optimization_lib)

option(OPTIMIZATION_USE_WORHP            "Use worhp"                    OFF)

add_definitions(-D_USE_MATH_DEFINES)

file(GLOB sources
	src/solvers/*.cpp
	src/objective_functions/*.cpp
	include/solvers/*.h
	include/objective_functions/*.h
	include/utils.h
	)


add_library(${PROJECT_NAME} ${sources})
add_library(rds::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME}
	PRIVATE
		${CMAKE_SOURCE_DIR}
		${PROJECT_SOURCE_DIR}/include
	)

target_link_libraries(${PROJECT_NAME} PRIVATE igl::core)
if(LIB_WITH_MATLAB)
	target_link_libraries(${PROJECT_NAME} PRIVATE igl::matlab)
endif()
if(MKL_FOUND)
	set(EIGEN_USE_MKL_ALL)
	set(MKL_LP64)
	message("MKL found. MKL Include dir: " ${MKL_INCLUDE_DIR})
	include_directories( ${MKL_INCLUDE_DIR})
	target_link_libraries(${PROJECT_NAME} PRIVATE ${MKL_LIBRARIES})
endif()

if(OPTIMIZATION_USE_WORHP)
	target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/ext/worhp/worhp)
	add_library(worhp SHARED IMPORTED)
	set_property(TARGET worhp PROPERTY IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/ext/worhp/bin/worhp.dll)
	set_property(TARGET worhp PROPERTY IMPORTED_IMPLIB ${CMAKE_SOURCE_DIR}/ext/worhp/bin/worhp.lib)
	target_link_libraries(${PROJECT_NAME} PRIVATE worhp)
endif()