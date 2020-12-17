#pragma once
#include "Cuda_Basics.cuh"

namespace Cuda {
	namespace FixChosenNormals {
		extern Array<double> grad, EnergyAtomic;
		extern indices mesh_indices;
		extern Array<int> Const_Ind;
		extern Array<double3> Const_Pos;

		extern double value();
		extern void gradient();
		extern void FreeAllVariables();
	}
}
