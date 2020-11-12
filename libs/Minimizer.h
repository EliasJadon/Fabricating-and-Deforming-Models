#pragma once
#include "TotalObjective.h"
#include <atomic>
#include <shared_mutex>
#include <igl/flip_avoiding_line_search.h>
#include <Eigen/SparseCholesky>
#include <fstream>


class Minimizer
{
public:
	Minimizer(const int solverID);
	int run();
	
	void run_one_iteration(const int steps,int* lambda_counter , const bool showGraph);
	void stop();
	void get_data(Eigen::MatrixXd& X, Eigen::MatrixXd& center, Eigen::VectorXd& radius, Eigen::MatrixXd& norm);
	void init(
		std::shared_ptr<ObjectiveFunction> objective,
		const Eigen::VectorXd& X0,
		const Eigen::VectorXd& norm0,
		const Eigen::VectorXd& center0,
		const Eigen::VectorXd& Radius0,
		const Eigen::MatrixXi& F,
		const Eigen::MatrixXd& V);
	
	// Pointer to the energy class
	std::shared_ptr<ObjectiveFunction> objective;

	// Activity flags
	std::atomic_bool is_running = {false};
	std::atomic_bool progressed = {false};

	// Synchronization functions used by the wrapper
	void wait_for_parameter_update_slot();
	void release_parameter_update_slot();

	// External (interface) and internal working mesh
	Eigen::VectorXd ext_x;
	Eigen::VectorXd ext_center, ext_radius, ext_norm;
	Eigen::VectorXd X;
	Eigen::MatrixX3i F;
	Eigen::MatrixXd V;
	
	double timer_curr=0, timer_sum = 0, timer_avg = 0;

	OptimizationUtils::LineSearch lineSearch_type;
	double constantStep_LineSearch;
	inline int getNumiter() {
		return this->numIteration;
	}
	void update_lambda(int*);
	bool isAutoLambdaRunning = false;
	int autoLambda_from = 100, autoLambda_count = 70, autoLambda_jump = 50;
protected:
	// Give the wrapper a chance to intersect gracefully
	void give_parameter_update_slot();
	// Updating the data after a step has been done
	void update_external_data();
	// Descent direction evaluated in step
	Eigen::VectorXd p;
	// Current energy, gradient and hessian
	Eigen::VectorXd g;
	double currentEnergy;
	int numIteration = 0;
private:
	int solverID;
	virtual void step() = 0;

	void linesearch();
	void value_linesearch();
	void gradNorm_linesearch();
	void constant_linesearch();
	virtual void internal_init() = 0;
	double step_size;
	int cur_iter;

	// Mutex stuff
	std::unique_ptr<std::shared_timed_mutex> data_mutex;
	std::unique_ptr<std::mutex> parameters_mutex;
	std::unique_ptr<std::condition_variable> param_cv;
	// Synchronization structures
	std::atomic_bool params_ready_to_update = { false };
	std::atomic_bool wait_for_param_update = { false };
	std::atomic_bool halt = { false };
};