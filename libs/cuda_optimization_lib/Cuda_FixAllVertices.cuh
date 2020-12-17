#pragma once
#include "Cuda_Basics.cuh"

namespace Cuda {
	namespace FixAllVertices {
		extern Array<double> grad, EnergyAtomic;
		extern Array<double3> restShapeV;
		extern unsigned int num_faces, num_vertices;
		extern double value(Cuda::Array<double>& curr_x);
		extern void gradient(Cuda::Array<double>& X);
		extern void FreeAllVariables();		
	}
}