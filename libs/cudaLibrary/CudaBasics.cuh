#pragma once

enum FunctionType {
	QUADRATIC = 0,
	EXPONENTIAL = 1,
	SIGMOID = 2
};

namespace Cuda {
	struct hinge {
		int f0, f1;
	};
	extern hinge newHinge(int f0, int f1);

	template <typename T> struct rowVector{
		T x, y, z;
	};
	template <typename T> rowVector<T> newRowVector(T x, T y, T z) {
		rowVector<T> a;
		a.x = x;
		a.y = y;
		a.z = z;
		return a;
	}
	template <typename T> struct Array {
		unsigned int size;
		T* host_arr;
		T* cuda_arr;
	};
	extern void check_devices_properties();
	extern void initCuda();
	extern void StopCudaDevice();
}
