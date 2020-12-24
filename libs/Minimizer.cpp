#include "Minimizer.h"
#include "AuxBendingNormal.h"
#include "AuxSpherePerHinge.h"

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
	assert(X0.rows() == (3 * V.rows()) && "X0 should contain the (x,y,z) coordinates for each vertice");
	assert(norm0.rows() == (3 * F.rows()) && "norm0 should contain the (x,y,z) coordinates for each face");
	
	this->F = F;
	this->V = V;
	this->ext_x = X0;
	this->ext_center = center0;
	this->ext_norm = norm0;
	this->ext_radius = Radius0;
	this->constantStep_LineSearch = 0.01;
	this->totalObjective = Tobjective;
	
	std::cout << "F.rows() = " << F.rows() << std::endl;
	std::cout << "V.rows() = " << V.rows() << std::endl;
	Cuda::initCuda();

	unsigned int size = 3 * V.rows() + 7 * F.rows();
	this->cuda_Minimizer = std::make_shared<Cuda_Minimizer>(size);
	for (int i = 0; i < 3 * V.rows(); i++)
		cuda_Minimizer->X.host_arr[i] = X0[i];
	for (int i = 0; i < 3 * F.rows(); i++)
		cuda_Minimizer->X.host_arr[3 * V.rows() + i] = norm0[i];
	for (int i = 0; i < 3 * F.rows(); i++)
		cuda_Minimizer->X.host_arr[3 * V.rows() + 3 * F.rows() + i] = center0[i];
	for (int i = 0; i < F.rows(); i++)
		cuda_Minimizer->X.host_arr[3 * V.rows() + 6 * F.rows() + i] = Radius0[i];
	Cuda::MemCpyHostToDevice(cuda_Minimizer->X);
	Cuda::copyArrays(cuda_Minimizer->curr_x, cuda_Minimizer->X);
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
		std::shared_ptr<AuxSpherePerHinge> ASH = std::dynamic_pointer_cast<AuxSpherePerHinge>(totalObjective->objectiveList[0]);
		std::shared_ptr<AuxBendingNormal> ABN = std::dynamic_pointer_cast<AuxBendingNormal>(totalObjective->objectiveList[1]);
		ASH->cuda_ASH->planarParameter /= 2;
		ABN->cuda_ABN->planarParameter /= 2;
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

	totalObjective->gradient(cuda_Minimizer, cuda_Minimizer->X, true);
	if (step_type == MinimizerType::ADAM_MINIMIZER)
		cuda_Minimizer->adam_Step();
	currentEnergy = totalObjective->value(cuda_Minimizer->X, true);
	linesearch();
	update_external_data(steps);
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
	step_size = 0.00390625;//1;
	cur_iter = 0; 
	int MAX_STEP_SIZE_ITER = 50;
	while (cur_iter++ < MAX_STEP_SIZE_ITER) 
	{
		//Eigen::VectorXd curr_x = X + step_size * p;
		cuda_Minimizer->linesearch_currX(step_size);

		double new_energy = totalObjective->value(cuda_Minimizer->curr_x,false);
		if (new_energy >= currentEnergy)
			step_size /= 2;
		else 
		{
			//X = curr_x;
			Cuda::copyArrays(cuda_Minimizer->X, cuda_Minimizer->curr_x);
			break;
		}
	}
	std::cout << "cur_iter = " << cur_iter << std::endl;
}

void Minimizer::constant_linesearch()
{
	/*step_size = constantStep_LineSearch;
	cur_iter = 0;
	X = X + step_size * p;*/
}

void Minimizer::gradNorm_linesearch()
{
	/*step_size = 1;
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
	}*/
}

void Minimizer::stop()
{
	wait_for_parameter_update_slot();
	halt = true;
	release_parameter_update_slot();
}

void Minimizer::update_external_data(int steps)
{
	give_parameter_update_slot();
	std::unique_lock<std::shared_timed_mutex> lock(*data_mutex);
	//if (steps % 10 == 0) {
		Cuda::MemCpyDeviceToHost(cuda_Minimizer->X);
		for (int i = 0; i < 3 * V.rows(); i++)
			ext_x[i] = cuda_Minimizer->X.host_arr[i];
		for (int i = 0; i < 3 * F.rows(); i++)
			ext_norm[i] = cuda_Minimizer->X.host_arr[3 * V.rows() + i];
		for (int i = 0; i < 3 * F.rows(); i++)
			ext_center[i] = cuda_Minimizer->X.host_arr[3 * V.rows() + 3 * F.rows() + i];
		for (int i = 0; i < F.rows(); i++)
			ext_radius[i] = cuda_Minimizer->X.host_arr[3 * V.rows() + 6 * F.rows() + i];
	//}
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