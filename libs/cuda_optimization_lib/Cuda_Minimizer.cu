#include "Cuda_Minimizer.cuh"
#define alpha_adam 1.0f
#define beta1_adam 0.90f
#define beta2_adam 0.9990f

namespace Utils_Cuda_Minimizer {
	__global__ void currXKernel(
		double* curr_x,
		const double* X,
		const double* p,
		const double step_size)
	{
		int index = blockIdx.x;
		curr_x[index] = X[index] + step_size * p[index];
	}
	__global__ void TotalGradientKernel(
		const unsigned int size, double* total_g,
		const double* g1, const double w1,
		const double* g2, const double w2,
		const double* g3, const double w3,
		const double* g4, const double w4,
		const double* g5, const double w5,
		const double* g6, const double w6,
		const double* g7, const double w7)
	{
		unsigned int Global_idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (Global_idx < size) {
			total_g[Global_idx] =
				w1 * g1[Global_idx] +
				w2 * g2[Global_idx] +
				w3 * g3[Global_idx] +
				w4 * g4[Global_idx] +
				w5 * g5[Global_idx] +
				w6 * g6[Global_idx] +
				w7 * g7[Global_idx];
		}
	}
	__global__ void AdamStepKernel(
		const double* g,
		double* v_adam,
		double* s_adam,
		double* p)
	{
		int index = blockIdx.x;
		v_adam[index] = beta1_adam * v_adam[index] + (1 - beta1_adam) * g[index];
		double tmp = pow(g[index], 2);
		s_adam[index] = beta2_adam * s_adam[index] + (1 - beta2_adam) * tmp;
		tmp = sqrt(s_adam[index]) + 1e-8;
		p[index] = -v_adam[index] / tmp;
	}
	__global__ void gradientDescentStepKernel(const double* g, double* p)
	{
		p[blockIdx.x] = -g[blockIdx.x];
	}
}

Cuda_Minimizer::Cuda_Minimizer(const unsigned int size)
{
	Cuda::AllocateMemory(X, size);
	Cuda::AllocateMemory(p, size);
	Cuda::AllocateMemory(g, size);
	Cuda::AllocateMemory(curr_x, size);
	Cuda::AllocateMemory(v_adam, size);
	Cuda::AllocateMemory(s_adam, size);
	for (int i = 0; i < size; i++) {
		v_adam.host_arr[i] = 0;
		s_adam.host_arr[i] = 0;
	}
	Cuda::MemCpyHostToDevice(v_adam);
	Cuda::MemCpyHostToDevice(s_adam);
}
Cuda_Minimizer::~Cuda_Minimizer() {
	cudaGetErrorString(cudaGetLastError());
	FreeMemory(X);
	FreeMemory(p);
	FreeMemory(g);
	FreeMemory(curr_x);
	FreeMemory(v_adam);
	FreeMemory(s_adam);
}

void Cuda_Minimizer::adam_Step()
{
	Utils_Cuda_Minimizer::AdamStepKernel << <v_adam.size, 1 >> > (
		g.cuda_arr,
		v_adam.cuda_arr,
		s_adam.cuda_arr,
		p.cuda_arr);
	Cuda::CheckErr(cudaDeviceSynchronize());
}
void Cuda_Minimizer::gradientDescent_Step()
{
	Utils_Cuda_Minimizer::gradientDescentStepKernel << <g.size, 1 >> > (
		g.cuda_arr, p.cuda_arr);
	Cuda::CheckErr(cudaDeviceSynchronize());
}
void Cuda_Minimizer::linesearch_currX(const double step_size) {
	Utils_Cuda_Minimizer::currXKernel << <curr_x.size, 1 >> > (
		curr_x.cuda_arr,
		X.cuda_arr,
		p.cuda_arr,
		step_size);
	Cuda::CheckErr(cudaDeviceSynchronize());
}
void Cuda_Minimizer::TotalGradient(
	const double* g1, const double w1,
	const double* g2, const double w2,
	const double* g3, const double w3,
	const double* g4, const double w4,
	const double* g5, const double w5,
	const double* g6, const double w6,
	const double* g7, const double w7)
{
	Utils_Cuda_Minimizer::TotalGradientKernel << <ceil(g.size / (double)1024), 1024 >> > (
		g.size, g.cuda_arr,
		g1, w1,
		g2, w2,
		g3, w3,
		g4, w4,
		g5, w5,
		g6, w6,
		g7, w7);
	Cuda::CheckErr(cudaDeviceSynchronize());
}

