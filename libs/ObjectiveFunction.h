#pragma once
#include "OptimizationUtils.h"
#include "..//..//plugins/console_color.h"

class ObjectiveFunction
{
public:
	// mesh vertices and faces
	Eigen::MatrixX3i restShapeF;
	Eigen::MatrixX3d restShapeV;

public:
	ObjectiveFunction() {}
	virtual ~ObjectiveFunction(){}
	virtual void init() = 0;
	virtual void init_hessian() = 0;
	virtual void updateX(const Eigen::VectorXd& X) = 0;
	virtual double value(const bool update) = 0;
	virtual void gradient(Eigen::VectorXd& g, const bool update) = 0;
	virtual void hessian() = 0;
	
	void init_mesh(const Eigen::MatrixXd& V, const Eigen::MatrixX3i& F);

	//Finite Differences check point
	Eigen::VectorXd FDGradient(const Eigen::VectorXd& X);
    void FDHessian(const Eigen::VectorXd& X, std::vector<int>& I, std::vector<int>& J, std::vector<double>& S);
    void checkGradient(const Eigen::VectorXd& X);
    void checkHessian(const Eigen::VectorXd& X);
	
	// Hessian sparse reprensentation
	std::vector<int> II, JJ;
	std::vector<double> SS;

	//weight for each objective function
	float w = 0;
	Eigen::VectorXd Efi;
	double energy_value = 0;
	double gradient_norm = 0;
	std::string name = "Objective function";
};

