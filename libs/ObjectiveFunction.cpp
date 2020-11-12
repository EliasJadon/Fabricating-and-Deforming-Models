#include "ObjectiveFunction.h"

void ObjectiveFunction::init_mesh(const Eigen::MatrixXd& V, const Eigen::MatrixX3i& F) {
	Eigen::MatrixX3d V3d(V.rows(), 3);
	if (V.cols() == 2) {
		V3d.leftCols(2) = V;
		V3d.col(2).setZero();
	}
	else if (V.cols() == 3) {
		V3d = V;
	}
	this->restShapeV = V3d;
	this->restShapeF = F;
}

Eigen::VectorXd ObjectiveFunction::FDGradient(const Eigen::VectorXd& X)
{
	Eigen::VectorXd g,Xd = X;
	g.resize(X.rows());
    updateX(Xd);
	double dX = 10e-6; //10e-6;
    double f_P, f_M;

    //this is a very slow method that evaluates the gradient of the objective function through FD...
    for (int i = 0; i < g.size(); i++) {
        double tmpVal = X(i);
        Xd(i) = tmpVal + dX;
        updateX(Xd);
        f_P = value(false);

        Xd(i) = tmpVal - dX;
        updateX(Xd);
        f_M = value(false);
		
        //now reset the ith param value
        Xd(i) = tmpVal;
        g(i) = (f_P - f_M) / (2 * dX);
    }
    //now restore the parameter set and make sure to tidy up a bit...
    updateX(X);
	return g;
}

void ObjectiveFunction::FDHessian(const Eigen::VectorXd& X, std::vector<int>& I, std::vector<int>& J, std::vector<double>& S)
{
	I.clear(); J.clear(); S.clear();
	int rows,columns;
	Eigen::VectorXd gp, gm, Hi;
	
	rows = columns = X.rows();
	gp.resize(X.rows());
	gm.resize(X.rows());
	I.push_back(X.rows()-1); J.push_back(X.rows()-1); S.push_back(0.);
	
	Eigen::VectorXd Xd = X;
	updateX(Xd);
	double dX = 10e-6;//10e-9;
    
    for (int i = 0; i < rows; i++) {
        double tmpVal = X(i);
        Xd(i) = tmpVal + dX;
        updateX(Xd);
		gradient(gp, false);
  
        Xd(i) = tmpVal - dX;
        updateX(Xd);
		gradient(gm, false);
        //now reset the ith param value
        Xd(i) = tmpVal;
        //and compute the row of the hessian
        Hi = (gp - gm) / (2 * dX);
        //each vector is a column vector of the hessian, so copy it in place...
		for (int j = i; j < columns; j++)
        {
            if (abs(Hi[j]) > 1e-8)
            {
                I.push_back(i);
                J.push_back(j);
                S.push_back(Hi[j]);
            }
        }
    }
}

void ObjectiveFunction::checkGradient(const Eigen::VectorXd& X)
{
	Eigen::VectorXd FD_gradient, Analytic_gradient;
	updateX(X);
	Analytic_gradient.resize(X.size());
	gradient(Analytic_gradient, false);
	FD_gradient = FDGradient(X);
	assert(FD_gradient.rows() == Analytic_gradient.rows() && "The size of analytic gradient & FD gradient must be equal!");
	double tol = 1e-4;
	double eps = 1e-10;
	
	//std::cout << "g= " << Analytic_gradient  << std::endl;
	//std::cout << "FD= " << FD_gradient << std::endl;
	
	std::cout << name << ": g.norm() = " << Analytic_gradient.norm() << "(analytic) , " << FD_gradient.norm() << "(FD)" << std::endl;
	for (int i = 0; i < Analytic_gradient.size(); i++) {
        double absErr = abs(FD_gradient[i] - Analytic_gradient[i]);
        double relError = 2 * absErr / (eps + Analytic_gradient[i] + FD_gradient[i]);
        if (relError > tol && absErr > 1e-6) {
            printf("Mismatch element %d: Analytic val: %lf, FD val: %lf. Error: %lf(%lf%%)\n", i, Analytic_gradient(i), FD_gradient(i), absErr, relError * 100);
        }
    }
}

void ObjectiveFunction::checkHessian(const Eigen::VectorXd& X)
{
	double tol = 1e-4;
	double eps = 1e-10;
	std::vector<int> I, J;
	std::vector<double> S;
	Eigen::SparseMatrix<double>  FDH, H;
		
	init();
	updateX(X);
	
	hessian();
	H = OptimizationUtils::BuildMatrix(II, JJ, SS);
	
	FDHessian(X, I, J, S);
	FDH = OptimizationUtils::BuildMatrix(I, J, S);
	
	assert(H.size() == FDH.size() && "The size of analytic hessian & FD hessian must be equal!");
	
	/*std::cout << std::endl << "H = " << std::endl <<
		H.toDense() << std::endl << "-------------";
	std::cout << std::endl << "FDH = " << std::endl << 
		FDH.toDense() << std::endl;*/

	std::cout << name << ": testing hessians...\n";
	for (int i = 0; i < H.rows(); i++) {
		for (int j = 0; j < H.cols(); j++) {
			double absErr = std::abs(FDH.coeff(i, j) - H.coeff(i, j));
			double relError = 2 * absErr / (eps + FDH.coeff(i, j) + H.coeff(i, j));
			if (relError > tol && absErr > 1e-6) {
				printf("Mismatch element %d,%d: Analytic val: %lf, FD val: %lf. Error: %lf(%lf%%)\n", i, j, H.coeff(i, j), FDH.coeff(i, j), absErr, relError * 100);
			}
		}
	}
}