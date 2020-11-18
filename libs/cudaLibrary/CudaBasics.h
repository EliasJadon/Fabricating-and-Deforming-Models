#pragma once

enum FunctionType {
	QUADRATIC = 0,
	EXPONENTIAL = 1,
	SIGMOID = 2
};

namespace Cuda {
	struct hinge {
		int f0, f1;
		hinge() {
			this->f0 = -1;
			this->f1 = -1;
		}
		hinge(int f0, int f1) {
			this->f0 = f0;
			this->f1 = f1;
		}
	};
	struct rowVector {
		double x, y, z;
		rowVector() {
			this->x = 0;
			this->y = 0;
			this->z = 0;
		}
		rowVector(double x, double y, double z) {
			this->x = x;
			this->y = y;
			this->z = z;
		}
	};
	template <typename T> struct Array {
		unsigned int size;
		T* host_arr;
		T* cuda_arr;
	};
	extern void check_devices_properties();
	extern void addWithCuda(int* c, const int* a, const int* b, unsigned int size);
	extern void initCuda();
	extern void StopCudaDevice();
}
