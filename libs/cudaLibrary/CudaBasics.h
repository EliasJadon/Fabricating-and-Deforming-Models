#pragma once

namespace Cuda {
	struct hinge {
		int f0, f1;
	};
	struct rowVector {
		double x, y, z;
	};
	template <typename T>
	struct Array {
		unsigned int size;
		T* host_arr;
		T* cuda_arr;
	};

	void check_devices_properties();
	void addWithCuda(int* c, const int* a, const int* b, unsigned int size);
	void initCuda();
	void StopCudaDevice();
}

