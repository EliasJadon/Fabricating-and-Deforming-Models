#include "Cuda_Basics.cuh"


namespace Cuda {
	void initIndices(
		indices& I, 
		const unsigned int F, 
		const unsigned int V,
		const unsigned int H) 
	{
		I.num_vertices = V;
		I.num_faces = F;
		I.num_hinges = H;
		I.startVx	= 0 * V + 0 * F;
		I.startVy	= 1 * V + 0 * F;
		I.startVz	= 2 * V + 0 * F;
		I.startNx	= 3 * V + 0 * F;
		I.startNy	= 3 * V + 1 * F;
		I.startNz	= 3 * V + 2 * F;
		I.startCx	= 3 * V + 3 * F;
		I.startCy	= 3 * V + 4 * F;
		I.startCz	= 3 * V + 5 * F;
		I.startR	= 3 * V + 6 * F;
	}

	hinge newHinge(int f0, int f1) {
		hinge a;
		a.f0 = f0;
		a.f1 = f1;
		return a;
	}
}

