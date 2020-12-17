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
	return Eigen::VectorXd::Zero(1);
	//Eigen::VectorXd g,Xd = X;
	//g.resize(X.rows());
 //   updateX(Xd);
	//double dX = 10e-6; //10e-6;
 //   double f_P, f_M;

 //   //this is a very slow method that evaluates the gradient of the objective function through FD...
 //   for (int i = 0; i < g.size(); i++) {
 //       double tmpVal = X(i);
 //       Xd(i) = tmpVal + dX;
 //       updateX(Xd);
 //       f_P = value(false);

 //       Xd(i) = tmpVal - dX;
 //       updateX(Xd);
 //       f_M = value(false);
	//	
 //       //now reset the ith param value
 //       Xd(i) = tmpVal;
 //       g(i) = (f_P - f_M) / (2 * dX);
 //   }
 //   //now restore the parameter set and make sure to tidy up a bit...
 //   updateX(X);
	//return g;
}

void ObjectiveFunction::checkGradient(const Eigen::VectorXd& X)
{
	//Eigen::VectorXd FD_gradient, Analytic_gradient;
	//updateX(X);
	//Analytic_gradient.resize(X.size());
	//gradient(Analytic_gradient, false);
	//FD_gradient = FDGradient(X);
	//assert(FD_gradient.rows() == Analytic_gradient.rows() && "The size of analytic gradient & FD gradient must be equal!");
	//double tol = 1e-4;
	//double eps = 1e-10;
	//
	////std::cout << "g= " << Analytic_gradient  << std::endl;
	////std::cout << "FD= " << FD_gradient << std::endl;
	//
	//std::cout << name << ": g.norm() = " << Analytic_gradient.norm() << "(analytic) , " << FD_gradient.norm() << "(FD)" << std::endl;
	//for (int i = 0; i < Analytic_gradient.size(); i++) {
 //       double absErr = abs(FD_gradient[i] - Analytic_gradient[i]);
 //       double relError = 2 * absErr / (eps + Analytic_gradient[i] + FD_gradient[i]);
 //       if (relError > tol && absErr > 1e-6) {
 //           printf("Mismatch element %d: Analytic val: %lf, FD val: %lf. Error: %lf(%lf%%)\n", i, Analytic_gradient(i), FD_gradient(i), absErr, relError * 100);
 //       }
 //   }
}
