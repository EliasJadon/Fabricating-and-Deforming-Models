#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ double3 vec_sub(const double3 a, const double3 b)
{
	return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}
	