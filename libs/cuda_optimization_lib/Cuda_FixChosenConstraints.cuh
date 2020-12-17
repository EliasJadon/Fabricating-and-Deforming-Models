#pragma once
#include "Cuda_Basics.cuh"

namespace Cuda {
	namespace FixChosenConstraints {
		extern Array<double> grad, EnergyAtomic;
		extern indices mesh_indices;
		extern Array<int> Const_Ind;
		extern Array<double3> Const_Pos;
		extern unsigned int startX, startY, startZ;

		extern double value();
		extern void gradient();
		extern void FreeAllVariables();
	}
}
