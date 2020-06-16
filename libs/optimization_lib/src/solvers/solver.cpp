#include "solvers/solver.h"
#include "solvers/NewtonSolver.h"

#define HIGH 3
#define LOW -3
#define JUMP 0.01f
#define ARRAY_OUTPUT_SIZE (int)((HIGH - LOW) / JUMP) + 1

solver::solver(const int solverID)
	:
	solverID(solverID),
	parameters_mutex(std::make_unique<std::mutex>()),
	data_mutex(std::make_unique<std::shared_timed_mutex>()),
	param_cv(std::make_unique<std::condition_variable>()),
	num_steps(2147483647)
{
	lineSearch_alfa.resize(ARRAY_OUTPUT_SIZE, 1);
	lineSearch_value.resize(ARRAY_OUTPUT_SIZE, 1);
	lineSearch_gradientNorm.resize(ARRAY_OUTPUT_SIZE, 1);
#ifdef SAVE_DATA_IN_MATLAB
	// Launch MATLAB
	igl::matlab::mlinit(&engine);
	igl::matlab::mleval(&engine, "desktop");
#endif
#ifdef SAVE_DATA_IN_CSV
	//save data in csv files
	std::string path = OptimizationUtils::
		() + "CSV_Output\\" + "Solver" + std::to_string(solverID) + "\\";
	mkdir((OptimizationUtils::RDSPath() + "CSV_Output\\").c_str());
	mkdir(path.c_str());
	SearchDirInfo.open(path + "SearchDirInfo.csv");
	solverInfo.open(path + "solverInfo.csv");
	hessianInfo.open(path + "hessianInfo.csv");
#endif
}

solver::~solver() {
#ifdef SAVE_DATA_IN_CSV
	//close csv files
	SearchDirInfo.close();
	solverInfo.close();
	hessianInfo.close();
	std::cout << ">> csv " + std::to_string(solverID) + " files has been closed!" << std::endl;
#endif
}

void solver::init(
	std::shared_ptr<ObjectiveFunction> objective, 
	const Eigen::VectorXd& X0,
	const Eigen::VectorXd& norm0,
	const Eigen::MatrixXi& F, 
	const Eigen::MatrixXd& V,
	const OptimizationUtils::InitAuxVariables initAuxType
) {
	this->F = F;
	this->V = V;
	this->constantStep_LineSearch = 0.01;
	this->objective = objective;
	std::cout << "F.rows() = " << F.rows() << std::endl;
	std::cout << "V.rows() = " << V.rows() << std::endl;
	assert(X0.rows() == (3*V.rows()) && "X0 should contain the (x,y,z) coordinates for each vertice");
	assert(norm0.rows() == (3*F.rows()) && "norm0 should contain the (x,y,z) coordinates for each face");
	
	Eigen::MatrixXd center0;
	Eigen::VectorXd Radius0;
	if(initAuxType == OptimizationUtils::Sphere)
		OptimizationUtils::Least_Squares_Sphere_Fit(V, F, center0, Radius0);
	else if (initAuxType == OptimizationUtils::MeshCenter)
		OptimizationUtils::center_of_mesh(V, F, center0, Radius0);
	
	X.resize(3 * V.rows() + 7 * F.rows());
	X.middleRows(0 * V.rows() + 0 * F.rows(), 3 * V.rows()) = X0;
	X.middleRows(3 * V.rows() + 0 * F.rows(), 3 * F.rows()) = norm0;
	X.middleRows(3 * V.rows() + 3 * F.rows(), 3 * F.rows()) = Eigen::Map<Eigen::VectorXd>(center0.data(), F.size());
	X.middleRows(3 * V.rows() + 6 * F.rows(), 1 * F.rows()) = Radius0;
	ext_x = X0;
	ext_center = Eigen::Map<Eigen::VectorXd>(center0.data(), F.size());
	std::cout << "hello :)\n";
	internal_init();
}

int solver::run()
{
	is_running = true;
	halt = false;
	int steps = 0;
	do {
		std::cout << "step = " << steps << std::endl;
		run_one_iteration(steps,false);
	} while ((a_parameter_was_updated || test_progress()) && !halt && ++steps < num_steps);
	is_running = false;
	
	std::cout << ">> solver " + std::to_string(solverID) + " stopped" << std::endl;
	return 0;
}

void solver::run_one_iteration(const int steps,const bool showGraph) {
	step();
#if defined SAVE_DATA_IN_CSV || defined SAVE_DATA_IN_MATLAB
	prepareData();
#endif
	if (lineSearch_type == OptimizationUtils::LineSearch::GradientNorm)
		gradNorm_linesearch();
	else if (lineSearch_type == OptimizationUtils::LineSearch::FunctionValue)
		value_linesearch();
	else if (lineSearch_type == OptimizationUtils::LineSearch::ConstantStep)
		constant_linesearch();
	update_external_data();

#ifdef SAVE_DATA_IN_MATLAB
	sendDataToMatlab(showGraph);
#endif 
#ifdef SAVE_DATA_IN_CSV
	saveSolverInfo(steps, solverInfo);
	saveHessianInfo(steps, hessianInfo);
	saveSearchDirInfo(steps, SearchDirInfo);
#endif 
}

void solver::saveSolverInfo(int numIteration, std::ofstream& solverInfo) {
	//show only once the objective's function data
	std::shared_ptr<TotalObjective> totalObj = std::dynamic_pointer_cast<TotalObjective>(objective);
	if (!numIteration) {
		solverInfo << "Obj name,weight," << std::endl;
		for (auto& obj : totalObj->objectiveList) 
			solverInfo << obj->name << "," << obj->w << ","<< std::endl;
		solverInfo << std::endl << std::endl << ",," << totalObj->name << ",,,";
		for (auto& obj : totalObj->objectiveList) 
			solverInfo << obj->name << ",,,";
		solverInfo << std::endl << "Round,,value,grad,";
		for (auto& obj : totalObj->objectiveList)
			solverInfo << ",value,grad,";
		solverInfo << std::endl;
	}
	solverInfo << numIteration << ",," << totalObj->energy_value << "," << totalObj->gradient_norm << ",,";
	for (auto& obj : totalObj->objectiveList)
		solverInfo << obj->energy_value << "," << obj->gradient_norm << ",,";
	solverInfo << std::endl;
}

void solver::prepareData() {
	NewtonSolver* newtonSolver = dynamic_cast<NewtonSolver*>(this);
	assert(newtonSolver != NULL && "could not calculate matrix with gradient descent mode");
	CurrHessian = newtonSolver->get_Hessian();
	
	X_before = X;
	//calculate values in the search direction vector
	int counter;
	double alpha = 0;
	for (alpha = LOW, counter = 0; alpha <= HIGH; alpha += JUMP, counter++) {
		Eigen::VectorXd curr_x = X + alpha * p;
		Eigen::VectorXd grad;
		objective->updateX(curr_x);
		objective->gradient(grad, false);
		lineSearch_alfa(counter, 0) = alpha;
		lineSearch_value(counter, 0) = objective->value(false);
		lineSearch_gradientNorm(counter, 0) = grad.norm();
	}
	objective->updateX(X);
}

#ifdef SAVE_DATA_IN_MATLAB
void solver::sendDataToMatlab(const bool show_graph) {
	auto N = [&](std::string name) { return name + std::to_string(solverID); };
	NewtonSolver* newtonSolver = dynamic_cast<NewtonSolver*>(this);

	// Launch MATLAB
	igl::matlab::mlinit(&engine);
	igl::matlab::mleval(&engine, "desktop");
	
	// Send matrix to matlab
	igl::matlab::mlsetmatrix(&engine, N("H"), CurrHessian);
	igl::matlab::mleval(&engine, N("H") + N(" = full(H") + ")");
	igl::matlab::mlsetmatrix(&engine, N("g"), Eigen::MatrixXd(g));
	igl::matlab::mlsetmatrix(&engine, N("p"), Eigen::MatrixXd(p));
	
	igl::matlab::mlsetscalar(&engine, N("MSE"), newtonSolver->get_MSE());
	igl::matlab::mleval(&engine, N("Matlab_MSE")+ N(" = sum((H")+ N(" * (H") + N("\\(-g") + N(")) + g") + ").^2)");
	
	
	igl::matlab::mlsetmatrix(&engine, N("X_before"), Eigen::MatrixXd(X_before));
	igl::matlab::mlsetmatrix(&engine, N("X_After"), Eigen::MatrixXd(X));
	igl::matlab::mlsetmatrix(&engine, N("lineSearch_alfa"), Eigen::MatrixXd(lineSearch_alfa));
	igl::matlab::mlsetmatrix(&engine, N("lineSearch_value"), Eigen::MatrixXd(lineSearch_value));
	igl::matlab::mlsetmatrix(&engine, N("lineSearch_gradientNorm"), Eigen::MatrixXd(lineSearch_gradientNorm));
	igl::matlab::mlsetscalar(&engine, N("chosen_alfa"), step_size);
	igl::matlab::mlsetscalar(&engine, N("line_search_iter"), cur_iter);

	if (show_graph) {
		//open figure & set position
		igl::matlab::mleval(&engine, N("f") + " = figure");
		igl::matlab::mleval(&engine, N("f") + ".Position = get(groot, 'Screensize')");

		std::string linesearch_Name = (lineSearch_type == OptimizationUtils::LineSearch::GradientNorm) ?
			"Gradient Norm: " :
			"Function value: ";

		//plot first graph
		igl::matlab::mleval(&engine, "subplot(2, 2, 1)");
		igl::matlab::mleval(&engine, N("ln") + " = plot(" + N("lineSearch_alfa") + "," + N("lineSearch_value") + ")");
		igl::matlab::mleval(&engine, N("ax") + " = gca");
		igl::matlab::mleval(&engine, N("ax") + ".XAxisLocation = 'origin'");
		igl::matlab::mleval(&engine, N("ax") + ".YAxisLocation = 'origin';");
		igl::matlab::mleval(&engine, N("ln") + ".LineWidth = 2");
		igl::matlab::mleval(&engine, N("ln") + ".Color = [0.0 0.0 1.0]");
		igl::matlab::mleval(&engine, N("ln") + ".MarkerEdgeColor = 'b'");
		igl::matlab::mleval(&engine, "xlabel('alfa')");
		igl::matlab::mleval(&engine, "ylabel('function values')");
		igl::matlab::mleval(&engine,
			"title(['" +
			linesearch_Name +
			"\\alpha = ',num2str(" + N("chosen_alfa") + ") , '  ,  iteration = ',num2str(" + N("line_search_iter") + ")])");

		//plot second graph
		igl::matlab::mleval(&engine, "subplot(2, 2, 2)");
		igl::matlab::mleval(&engine, N("ln") + " = plot(" + N("lineSearch_alfa") + "," + N("lineSearch_augmentedValue") + ")");
		igl::matlab::mleval(&engine, N("ax") + " = gca");
		igl::matlab::mleval(&engine, N("ax") + ".XAxisLocation = 'origin'");
		igl::matlab::mleval(&engine, N("ax") + ".YAxisLocation = 'origin';");
		igl::matlab::mleval(&engine, N("ln") + ".LineWidth = 2");
		igl::matlab::mleval(&engine, N("ln") + ".Color = [0.0 1.0 0.0]");
		igl::matlab::mleval(&engine, N("ln") + ".MarkerEdgeColor = 'g'");
		igl::matlab::mleval(&engine, "xlabel('alfa')");
		igl::matlab::mleval(&engine, "ylabel('augmented function values')");
		igl::matlab::mleval(&engine,
			"title(['" +
			linesearch_Name +
			"\\alpha = ',num2str(" + N("chosen_alfa") + ") , '  ,  iteration = ',num2str(" + N("line_search_iter") + ")])");


		//plot third graph
		igl::matlab::mleval(&engine, "subplot(2, 2, 3)");
		igl::matlab::mleval(&engine, N("ln") + " = plot(" + N("lineSearch_alfa") + "," + N("lineSearch_gradientNorm") + ")");
		igl::matlab::mleval(&engine, N("ax") + " = gca");
		igl::matlab::mleval(&engine, N("ax") + ".XAxisLocation = 'origin'");
		igl::matlab::mleval(&engine, N("ax") + ".YAxisLocation = 'origin';");
		igl::matlab::mleval(&engine, N("ln") + ".LineWidth = 2");
		igl::matlab::mleval(&engine, N("ln") + ".Color = [1.0 0.0 0.0]");
		igl::matlab::mleval(&engine, N("ln") + ".MarkerEdgeColor = 'r'");
		igl::matlab::mleval(&engine, "xlabel('alfa')");
		igl::matlab::mleval(&engine, "ylabel('Gradient norm values')");
		igl::matlab::mleval(&engine,
			"title(['" +
			linesearch_Name +
			"\\alpha = ',num2str(" + N("chosen_alfa") + ") , '  ,  iteration = ',num2str(" + N("line_search_iter") + ")])");


		//plot fourth graph
		igl::matlab::mleval(&engine, "subplot(2, 2, 4)");
		igl::matlab::mleval(&engine, N("ln") + " = plot(" + N("lineSearch_alfa") + "," + N("lineSearch_value") + ")");
		igl::matlab::mleval(&engine, N("ax") + " = gca");
		igl::matlab::mleval(&engine, N("ax") + ".XAxisLocation = 'origin'");
		igl::matlab::mleval(&engine, N("ax") + ".YAxisLocation = 'origin';");
		igl::matlab::mleval(&engine, N("ln") + ".LineWidth = 2");
		igl::matlab::mleval(&engine, N("ln") + ".Color = [0.0 0.0 1.0]");
		igl::matlab::mleval(&engine, N("ln") + ".MarkerEdgeColor = 'b'");
		igl::matlab::mleval(&engine, "hold on");
		igl::matlab::mleval(&engine, N("ln") + " = plot(" + N("lineSearch_alfa") + "," + N("lineSearch_augmentedValue") + ")");
		igl::matlab::mleval(&engine, N("ln") + ".LineWidth = 2");
		igl::matlab::mleval(&engine, N("ln") + ".Color = [0.0 1.0 0.0]");
		igl::matlab::mleval(&engine, N("ln") + ".MarkerEdgeColor = 'g'");
		igl::matlab::mleval(&engine, N("ln") + " = plot(" + N("lineSearch_alfa") + "," + N("lineSearch_gradientNorm") + ")");
		igl::matlab::mleval(&engine, N("ln") + ".LineWidth = 2");
		igl::matlab::mleval(&engine, N("ln") + ".Color = [1.0 0.0 0.0]");
		igl::matlab::mleval(&engine, N("ln") + ".MarkerEdgeColor = 'r'");
		igl::matlab::mleval(&engine, "hold off");
		igl::matlab::mleval(&engine, "xlabel('alfa')");
		igl::matlab::mleval(&engine, "ylabel('values')");
		igl::matlab::mleval(&engine,
			"title(['" +
			linesearch_Name +
			"\\alpha = ',num2str(" + N("chosen_alfa") + ") , '  ,  iteration = ',num2str(" + N("line_search_iter") + ")])");


		//clear unused variables
		igl::matlab::mleval(&engine, "clear " + N("f") + " " + N("ln") + " " + N("ax"));
	}
	
	//clear unused variables
	igl::matlab::mleval(&engine, "clear " + N("lineSearch_alfa") + " " + N("lineSearch_value"));
	igl::matlab::mleval(&engine, "clear " + N("lineSearch_augmentedValue") + " " + N("lineSearch_gradientNorm"));
	igl::matlab::mleval(&engine, "clear " + N("chosen_alfa")+" " + N("line_search_iter"));
}
#endif

void solver::saveHessianInfo(int numIteration, std::ofstream& hessianInfo) {
	//show only once the objective's function data
	if (!numIteration) {
		std::shared_ptr<TotalObjective> t = std::dynamic_pointer_cast<TotalObjective>(objective);
		hessianInfo << "Obj name,weight," << std::endl; 
		for (auto& obj : t->objectiveList)
			hessianInfo << obj->name << "," << obj->w << "," << std::endl; 
		hessianInfo << std::endl;
	}
		
	//output the hessian
	hessianInfo << ("Round " + std::to_string(numIteration)).c_str() << std::endl;
	for (int i = 0; i < CurrHessian.rows(); i++) {
		for (int j = 0; j < CurrHessian.cols(); j++)
			hessianInfo << CurrHessian.coeff(i, j) << ",";
		hessianInfo << "," << g(i) << std::endl;
	}
	hessianInfo << std::endl;
}

void solver::saveSearchDirInfo(int numIteration, std::ofstream& SearchDirInfo) {
	//show only once the objective's function data
	if (!numIteration) {
		std::shared_ptr<TotalObjective> t = std::dynamic_pointer_cast<TotalObjective>(objective);
		SearchDirInfo << "Obj name,weight,";
		SearchDirInfo << std::endl;
		for (auto& obj : t->objectiveList)
			SearchDirInfo << obj->name << "," << obj->w << "," << std::endl;
		SearchDirInfo << std::endl << "Round" << std::endl;
	}
		
	//add the alfa values as one row
	SearchDirInfo << numIteration << ",alfa,";
	for (int i = 0; i < ARRAY_OUTPUT_SIZE; i++)
		SearchDirInfo << lineSearch_alfa(i, 0) << ",";
	SearchDirInfo << std::endl;

	//add the total objective's values as one row
	SearchDirInfo << ",value,";
	for (int i = 0; i < ARRAY_OUTPUT_SIZE; i++)
		SearchDirInfo << lineSearch_value(i, 0) << ",";
	SearchDirInfo << std::endl;

	//add the solver's choice of alfa
	if (lineSearch_type == OptimizationUtils::LineSearch::GradientNorm)
		SearchDirInfo << ",line search type,Gradient norm," << std::endl;
	else
		SearchDirInfo << ",line search type,Function value," << std::endl;
	SearchDirInfo << ",Chosen alfa," << step_size << "," << std::endl;
	SearchDirInfo << ",LineSearch iter," << cur_iter << "," << std::endl;
}

void solver::value_linesearch()
{
	step_size = 1;
	double new_energy = currentEnergy;
	cur_iter = 0; int MAX_STEP_SIZE_ITER = 50;
	while (cur_iter++ < MAX_STEP_SIZE_ITER) {
		Eigen::VectorXd curr_x = X + step_size * p;
		objective->updateX(curr_x);
		new_energy = objective->value(false);
		if (new_energy >= currentEnergy)
			step_size /= 2;
		else {
			X = curr_x;
			break;
		}
	}
}

void solver::constant_linesearch()
{
	step_size = constantStep_LineSearch;
	cur_iter = 0;
	X = X + step_size * p;
}

void solver::gradNorm_linesearch()
{
	step_size = 1;
	Eigen::VectorXd grad;

	objective->updateX(X);
	objective->gradient(grad,false);
	double current_GradNrom = grad.norm();
	double new_GradNrom = current_GradNrom;

	cur_iter = 0; int MAX_STEP_SIZE_ITER = 50;

	while (cur_iter++ < MAX_STEP_SIZE_ITER) {
		Eigen::VectorXd curr_x = X + step_size * p;

		objective->updateX(curr_x);
		objective->gradient(grad,false);
		new_GradNrom = grad.norm();
		
		if (new_GradNrom >= current_GradNrom)
			step_size /= 2;
		else {
			X = curr_x;
			break;
		}
	}
}

void solver::stop()
{
	wait_for_parameter_update_slot();
	halt = true;
	release_parameter_update_slot();
}

void solver::update_external_data()
{
	give_parameter_update_slot();
	std::unique_lock<std::shared_timed_mutex> lock(*data_mutex);
	ext_x = X.middleRows(0, 3 * V.rows());
	ext_center = X.middleRows(3 * V.rows() + 3 * F.rows(), 3 * F.rows());
	progressed = true;
}

void solver::get_data(Eigen::MatrixXd& X, Eigen::MatrixXd& center)
{
	std::unique_lock<std::shared_timed_mutex> lock(*data_mutex);
	X = Eigen::Map<Eigen::MatrixXd>(ext_x.data(), ext_x.rows() / 3, 3);
	center = Eigen::Map<Eigen::MatrixXd>(ext_center.data(), ext_center.rows() / 3, 3);
	progressed = false;
}

void solver::give_parameter_update_slot()
{
	a_parameter_was_updated = false;
	std::unique_lock<std::mutex> lock(*parameters_mutex);
	params_ready_to_update = true;
	param_cv->notify_one();
	while (wait_for_param_update)
	{
		param_cv->wait(lock);
		a_parameter_was_updated = true;
	}
	params_ready_to_update = false;
}

void solver::wait_for_parameter_update_slot()
{
	std::unique_lock<std::mutex> lock(*parameters_mutex);
	wait_for_param_update = true;
	while (!params_ready_to_update && is_running)
		param_cv->wait_for(lock, std::chrono::milliseconds(50));
}

void solver::release_parameter_update_slot()
{
	wait_for_param_update = false;
	param_cv->notify_one();
}