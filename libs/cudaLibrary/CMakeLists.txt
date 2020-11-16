project(cudaLibrary LANGUAGES CXX CUDA)

SET(CMAKE_CXX_STANDARD 			11)
SET(CMAKE_CUDA_STANDARD 		11)
SET(CUDA_SEPARABLE_COMPILATION  ON)

file(GLOB "${PROJECT_NAME}_files"
	*.cu
	*.cuh
	*.cpp
	*.h
	)

add_library(${PROJECT_NAME} ${${PROJECT_NAME}_files})
