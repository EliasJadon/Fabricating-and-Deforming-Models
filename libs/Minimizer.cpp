#include "Minimizer.h"
#include "AuxBendingNormal.h"
#include "AuxSpherePerHinge.h"

#define BETA1_ADAM 0.90f
#define BETA2_ADAM 0.9990f
#define EPSILON_ADAM 1e-8

Minimizer::Minimizer(const int solverID)
	:
	solverID(solverID),
	parameters_mutex(std::make_unique<std::mutex>()),
	data_mutex(std::make_unique<std::shared_timed_mutex>()),
	param_cv(std::make_unique<std::condition_variable>())
{ }

void Minimizer::init(
	std::shared_ptr<TotalObjective> Tobjective,
	const Eigen::VectorXd& X0,
	const Eigen::VectorXd& norm0,
	const Eigen::VectorXd& center0,
	const Eigen::VectorXd& Radius0,
	const Eigen::MatrixXi& F, 
	const Eigen::MatrixXd& V) 
{
	assert(X0.rows()			== (3 * V.rows()) && "X0 size is illegal!");
	assert(norm0.rows()			== (3 * F.rows()) && "norm0 size is illegal!");
	assert(center0.rows()		== (3 * F.rows()) && "center0 size is illegal!");
	assert(Radius0.rows()		== (1 * F.rows()) && "Radius0 size is illegal!");
	
	this->F = F;
	this->V = V;
	this->ext_x = X0;
	this->ext_center = center0;
	this->ext_norm = norm0;
	this->ext_radius = Radius0;
	this->constantStep_LineSearch = 0.01;
	this->totalObjective = Tobjective;
	
	unsigned int size = 3 * V.rows() + 7 * F.rows();

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
	
	for (int i = 0; i < 3 * V.rows(); i++)
		X.host_arr[0 * V.rows() + 0 * F.rows() + i] = X0[i];
	for (int i = 0; i < 3 * F.rows(); i++)
		X.host_arr[3 * V.rows() + 0 * F.rows() + i] = norm0[i];
	for (int i = 0; i < 3 * F.rows(); i++)
		X.host_arr[3 * V.rows() + 3 * F.rows() + i] = center0[i];
	for (int i = 0; i < F.rows(); i++)
		X.host_arr[3 * V.rows() + 6 * F.rows() + i] = Radius0[i];
	for (int i = 0; i < g.size; i++) {
		curr_x.host_arr[i] = X.host_arr[i];
	}
}

void Minimizer::run()
{
	is_running = true;
	halt = false;
	while (!halt)
		run_one_iteration();
	is_running = false;
	std::cout << ">> solver " + std::to_string(solverID) + " stopped" << std::endl;
}

void Minimizer::update_lambda() 
{
	std::shared_ptr<AuxSpherePerHinge> ASH = std::dynamic_pointer_cast<AuxSpherePerHinge>(totalObjective->objectiveList[0]);
	std::shared_ptr<AuxBendingNormal> ABN = std::dynamic_pointer_cast<AuxBendingNormal>(totalObjective->objectiveList[1]);
	
	if (isAutoLambdaRunning && numIteration >= autoLambda_from && !(numIteration % autoLambda_jump))
	{
		const double target = pow(2, -autoLambda_count);
		if (ASH->w)
			ASH->Dec_SigmoidParameter(target);
		if (ABN->w)
			ABN->Dec_SigmoidParameter(target);
	}
}

void Minimizer::run_one_iteration() 
{
	OptimizationUtils::Timer t(&timer_sum, &timer_curr);
	timer_avg = timer_sum / ++numIteration;
	update_lambda();

	totalObjective->gradient(X, true);
	if (Optimizer_type == Cuda::OptimizerType::Adam) {
		for (int i = 0; i < g.size; i++) {
			v_adam.host_arr[i] = BETA1_ADAM * v_adam.host_arr[i] + (1 - BETA1_ADAM) * g.host_arr[i];
			s_adam.host_arr[i] = BETA2_ADAM * s_adam.host_arr[i] + (1 - BETA2_ADAM) * pow(g.host_arr[i], 2);
			p.host_arr[i] = -v_adam.host_arr[i] / (sqrt(s_adam.host_arr[i]) + EPSILON_ADAM);
		}
	}
	else if (Optimizer_type == Cuda::OptimizerType::Gradient_Descent) {
		for (int i = 0; i < g.size; i++) {
			p.host_arr[i] = -g.host_arr[i];
		}
	}
	currentEnergy = totalObjective->value(X, true);
	linesearch();
	update_external_data();
}

void Minimizer::linesearch()
{
	if (lineSearch_type == OptimizationUtils::LineSearch::GRADIENT_NORM)
		gradNorm_linesearch();
	else if (lineSearch_type == OptimizationUtils::LineSearch::FUNCTION_VALUE)
		value_linesearch();
	else if (lineSearch_type == OptimizationUtils::LineSearch::CONSTANT_STEP)
		constant_linesearch();
}

void Minimizer::value_linesearch()
{	
	step_size = init_step_size;
	cur_iter = 0; 
	int MAX_STEP_SIZE_ITER = 50;
	while (cur_iter++ < MAX_STEP_SIZE_ITER) 
	{
		for (int i = 0; i < g.size; i++) {
			curr_x.host_arr[i] = X.host_arr[i] + step_size * p.host_arr[i];
		}

		double new_energy = totalObjective->value(curr_x,false);
		if (new_energy >= currentEnergy)
			step_size /= 2;
		else 
		{
			for (int i = 0; i < g.size; i++) {
				X.host_arr[i] = curr_x.host_arr[i];
			}
			break;
		}
	}
	if (cur_iter == 1)
		init_step_size *= 2;
	if (cur_iter > 2)
		init_step_size /= 2;
	linesearch_numiterations = cur_iter;
}

void Minimizer::constant_linesearch()
{
	for (int i = 0; i < g.size; i++) {
		curr_x.host_arr[i] = X.host_arr[i] + constantStep_LineSearch * p.host_arr[i];
		X.host_arr[i] = curr_x.host_arr[i];
	}
}

void Minimizer::gradNorm_linesearch()
{
}

void Minimizer::stop()
{
	wait_for_parameter_update_slot();
	halt = true;
	release_parameter_update_slot();
}

void Minimizer::update_external_data()
{
	give_parameter_update_slot();
	std::unique_lock<std::shared_timed_mutex> lock(*data_mutex);
	for (int i = 0; i < 3 * V.rows(); i++)
		ext_x[i] = X.host_arr[i];
	for (int i = 0; i < 3 * F.rows(); i++)
		ext_norm[i] = X.host_arr[3 * V.rows() + i];
	for (int i = 0; i < 3 * F.rows(); i++)
		ext_center[i] = X.host_arr[3 * V.rows() + 3 * F.rows() + i];
	for (int i = 0; i < F.rows(); i++)
		ext_radius[i] = X.host_arr[3 * V.rows() + 6 * F.rows() + i];
	progressed = true;
}

void Minimizer::get_data(
	Eigen::MatrixXd& X, 
	Eigen::MatrixXd& center, 
	Eigen::VectorXd& radius, 
	Eigen::MatrixXd& norm)
{
	std::unique_lock<std::shared_timed_mutex> lock(*data_mutex);
	X				= Eigen::Map<Eigen::MatrixXd>(ext_x.data(), ext_x.rows() / 3, 3);
	center			= Eigen::Map<Eigen::MatrixXd>(ext_center.data(), ext_center.rows() / 3, 3);
	radius			= ext_radius;
	norm			= Eigen::Map<Eigen::MatrixXd>(ext_norm.data(), ext_norm.rows() / 3, 3);
	progressed = false;
}

void Minimizer::give_parameter_update_slot()
{
	std::unique_lock<std::mutex> lock(*parameters_mutex);
	params_ready_to_update = true;
	param_cv->notify_one();
	while (wait_for_param_update)
	{
		param_cv->wait(lock);
	}
	params_ready_to_update = false;
}

void Minimizer::wait_for_parameter_update_slot()
{
	std::unique_lock<std::mutex> lock(*parameters_mutex);
	wait_for_param_update = true;
	while (!params_ready_to_update && is_running)
		param_cv->wait_for(lock, std::chrono::milliseconds(50));
}

void Minimizer::release_parameter_update_slot()
{
	wait_for_param_update = false;
	param_cv->notify_one();
}