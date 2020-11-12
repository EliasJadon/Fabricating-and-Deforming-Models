#include "Minimizer.h"
#include "NewtonMinimizer.h"
#include "AuxBendingNormal.h"
#include "AuxSpherePerHinge.h"
#include "BendingEdge.h"
#include "BendingNormal.h"

Minimizer::Minimizer(const int solverID)
	:
	solverID(solverID),
	parameters_mutex(std::make_unique<std::mutex>()),
	data_mutex(std::make_unique<std::shared_timed_mutex>()),
	param_cv(std::make_unique<std::condition_variable>())
{ }

void Minimizer::init(
	std::shared_ptr<ObjectiveFunction> objective, 
	const Eigen::VectorXd& X0,
	const Eigen::VectorXd& norm0,
	const Eigen::VectorXd& center0,
	const Eigen::VectorXd& Radius0,
	const Eigen::MatrixXi& F, 
	const Eigen::MatrixXd& V
) 
{
	this->F = F;
	this->V = V;
	this->constantStep_LineSearch = 0.01;
	this->objective = objective;
	std::cout << "F.rows() = " << F.rows() << std::endl;
	std::cout << "V.rows() = " << V.rows() << std::endl;
	assert(X0.rows() == (3*V.rows()) && "X0 should contain the (x,y,z) coordinates for each vertice");
	assert(norm0.rows() == (3*F.rows()) && "norm0 should contain the (x,y,z) coordinates for each face");
	X.resize(3 * V.rows() + 7 * F.rows());
	X.middleRows(0 * V.rows() + 0 * F.rows(), 3 * V.rows()) = X0;
	X.middleRows(3 * V.rows() + 0 * F.rows(), 3 * F.rows()) = norm0;
	X.middleRows(3 * V.rows() + 3 * F.rows(), 3 * F.rows()) = center0;
	X.middleRows(3 * V.rows() + 6 * F.rows(), 1 * F.rows()) = Radius0;
	ext_x = X0;
	ext_center = center0;
	ext_norm = norm0;
	ext_radius = Radius0;
	internal_init();
}

int Minimizer::run()
{
	is_running = true;
	halt = false;
	numIteration = 0;
	int lambda_counter = 0;
	do 
	{
		run_one_iteration(numIteration, &lambda_counter, false);
		numIteration++;
	} while (!halt);
	is_running = false;
	std::cout << ">> solver " + std::to_string(solverID) + " stopped" << std::endl;
	return 0;
}

void Minimizer::update_lambda(int* lambda_counter) 
{
	if (isAutoLambdaRunning &&
		numIteration >= autoLambda_from &&
		(*lambda_counter) < autoLambda_count &&
		numIteration % autoLambda_jump == 0)
	{
		std::shared_ptr<TotalObjective> totalObjective = std::dynamic_pointer_cast<TotalObjective>(objective);
		for (auto& obj : totalObjective->objectiveList) 
		{
			std::shared_ptr<AuxSpherePerHinge> ASH = std::dynamic_pointer_cast<AuxSpherePerHinge>(obj);
			std::shared_ptr<AuxBendingNormal> ABN = std::dynamic_pointer_cast<AuxBendingNormal>(obj);
			std::shared_ptr<BendingEdge> BE = std::dynamic_pointer_cast<BendingEdge>(obj);
			std::shared_ptr<BendingNormal> BN = std::dynamic_pointer_cast<BendingNormal>(obj);
			if (ASH)
				ASH->planarParameter /= 2;
			if (ABN)
				ABN->planarParameter /= 2;
			if (BE)
				BE->planarParameter /= 2;
			if (BN)
				BN->planarParameter /= 2;
		}
		(*lambda_counter)++;
	}
}

void Minimizer::run_one_iteration(
	const int steps,
	int* lambda_counter, 
	const bool showGraph) 
{
	OptimizationUtils::Timer t(&timer_sum, &timer_curr);
	numIteration = steps;
	timer_avg = timer_sum / numIteration;
	update_lambda(lambda_counter);
	step();
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
	step_size = 1;
	double new_energy = currentEnergy;
	cur_iter = 0; int MAX_STEP_SIZE_ITER = 50;
	while (cur_iter++ < MAX_STEP_SIZE_ITER) 
	{
		Eigen::VectorXd curr_x = X + step_size * p;
		objective->updateX(curr_x);
		new_energy = objective->value(false);
		if (new_energy >= currentEnergy)
			step_size /= 2;
		else 
		{
			X = curr_x;
			break;
		}
	}
}

void Minimizer::constant_linesearch()
{
	step_size = constantStep_LineSearch;
	cur_iter = 0;
	X = X + step_size * p;
}

void Minimizer::gradNorm_linesearch()
{
	step_size = 1;
	Eigen::VectorXd grad;
	objective->updateX(X);
	objective->gradient(grad,false);
	double current_GradNrom = grad.norm();
	double new_GradNrom = current_GradNrom;
	cur_iter = 0; int MAX_STEP_SIZE_ITER = 50;
	while (cur_iter++ < MAX_STEP_SIZE_ITER) 
	{
		Eigen::VectorXd curr_x = X + step_size * p;
		objective->updateX(curr_x);
		objective->gradient(grad,false);
		new_GradNrom = grad.norm();
		if (new_GradNrom >= current_GradNrom)
			step_size /= 2;
		else 
		{
			X = curr_x;
			break;
		}
	}
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
	ext_x =			X.middleRows(0 * V.rows() + 0 * F.rows(), 3 * V.rows());
	ext_norm =		X.middleRows(3 * V.rows() + 0 * F.rows(), 3 * F.rows());
	ext_center =	X.middleRows(3 * V.rows() + 3 * F.rows(), 3 * F.rows());
	ext_radius =	X.middleRows(3 * V.rows() + 6 * F.rows(), 1 * F.rows());
	progressed = true;
}

void Minimizer::get_data(
	Eigen::MatrixXd& X, 
	Eigen::MatrixXd& center, 
	Eigen::VectorXd& radius, 
	Eigen::MatrixXd& norm)
{
	std::unique_lock<std::shared_timed_mutex> lock(*data_mutex);
	X = Eigen::Map<Eigen::MatrixXd>(ext_x.data(), ext_x.rows() / 3, 3);
	center = Eigen::Map<Eigen::MatrixXd>(ext_center.data(), ext_center.rows() / 3, 3);
	radius = ext_radius;
	norm = Eigen::Map<Eigen::MatrixXd>(ext_norm.data(), ext_norm.rows() / 3, 3);
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