#pragma once
#include "Cuda_Basics.cuh"

namespace Cuda {
	namespace FixAllVertices {
		extern Array<double> grad, EnergyAtomic;
		extern Array<double3> restShapeV;
		extern unsigned int num_faces, num_vertices;
		extern double value();
		extern void gradient();
		extern void FreeAllVariables();		
	}
}