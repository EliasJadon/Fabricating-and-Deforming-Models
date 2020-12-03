#pragma once
#include "Cuda_Basics.cuh"

namespace Cuda {
	namespace SymmetricDirichlet {
		extern Array<double> grad, EnergyAtomic, EnergyVec, restShapeArea;
		extern Array<double3> D1d, D2d;
		extern Array<int3> restShapeF;
		extern unsigned int num_faces, num_vertices;

		extern double value();
		extern void gradient();
		extern void FreeAllVariables();		
	}
}