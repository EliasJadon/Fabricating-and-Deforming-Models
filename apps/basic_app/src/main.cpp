#include <plugins/deformation_plugin/include/deformation_plugin.h>
//#include "libs/optimization_lib/include/linear_equation_solvers/tryPardiso.h"

int main()
{
	igl::opengl::glfw::Viewer viewer;
	deformation_plugin plugin;
	viewer.plugins.push_back(&plugin);
	viewer.launch();
	return EXIT_SUCCESS;
}



//
//#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
//#include <imgui/imgui.h>
//
//
//#include <iostream>
//#include <stdio.h>
//#include <stdlib.h>
//#include <math.h>
//
//using namespace std;
//
///* PARDISO prototype. */
//extern "C" void pardisoinit(void   *, int    *, int *, int *, double *, int *);
//extern "C" void pardiso(void   *, int    *, int *, int *, int *, int *,
//	double *, int    *, int *, int *, int *, int *,
//	int *, double *, double *, int *, double *);
//extern "C" void pardiso_chkmatrix(int *, int *, double *, int *, int *, int *);
//extern "C" void pardiso_chkvec(int *, int *, double *, int *);
//extern "C" void pardiso_printstats(int *, int *, double *, int *, int *, int *,
//	double *, int *);
//
//
//inline int Fortran_1_based(int value) {
//	return value + 1;
//}
//
//class pardisoSolver
//{
//private:
//	/* Matrix data. */
//	int    n;
//	int*    ia;
//	int*    ja;
//	double*  a;
//	int      nnz;
//	int      mtype = -2;        /* Real symmetric matrix */
//
//	/* RHS and solution vectors. */
//	double   b[8], x[8];
//	int      nrhs = 1;          /* Number of right hand sides. */
//
//	/* Internal solver memory pointer pt,                  */
//	/* 32-bit: int pt[64]; 64-bit: long int pt[64]         */
//	/* or void *pt[64] should be OK on both architectures  */
//	void    *pt[64];
//
//	/* Pardiso control parameters. */
//	int      iparm[64];
//	double   dparm[64];
//	int      maxfct, mnum, phase, error, msglvl, solver;
//
//	/* Auxiliary variables. */
//	char    *var;
//	double   ddum;              /* Double dummy */
//	int      idum;              /* Integer dummy. */
//public:
//	void init() {
//		/* -------------------------------------------------------------------- */
//		/* ..  Setup Pardiso control parameters.                                */
//		/* -------------------------------------------------------------------- */
//		error = 0;
//		solver = 0; /* use sparse direct solver */
//		mtype = -2;
//		nrhs = 1;
//		pardisoinit(pt, &mtype, &solver, iparm, dparm, &error);
//
//		if (error != 0)
//		{
//			if (error == -10)
//				printf("No license file found \n");
//			if (error == -11)
//				printf("License is expired \n");
//			if (error == -12)
//				printf("Wrong username or hostname \n");
//			exit(1);
//		}
//		else
//			printf("[PARDISO]: License check was successful ... \n");
//
//		iparm[2] = 1;	// num_procs
//		maxfct = 1;		/* Maximum number of numerical factorizations.  */
//		mnum = 1;         /* Which factorization to use. */
//		msglvl = 1;         /* Print statistical information  */
//		error = 0;         /* Initialize error flag */
//	}
//	void debugVectorAndMatrix() {
//		///* -------------------------------------------------------------------- */
//		///*  .. pardiso_chk_matrix(...)                                          */
//		///*     Checks the consistency of the given matrix.                      */
//		///*     Use this functionality only for debugging purposes               */
//		///* -------------------------------------------------------------------- */
//
//		//pardiso_chkmatrix(&mtype, &n, a, ia, ja, &error);
//		//if (error != 0) {
//		//	printf("\nERROR in consistency of matrix: %d", error);
//		//	exit(1);
//		//}
//
//		///* -------------------------------------------------------------------- */
//		///* ..  pardiso_chkvec(...)                                              */
//		///*     Checks the given vectors for infinite and NaN values             */
//		///*     Input parameters (see PARDISO user manual for a description):    */
//		///*     Use this functionality only for debugging purposes               */
//		///* -------------------------------------------------------------------- */
//
//		//pardiso_chkvec(&n, &nrhs, b, &error);
//		//if (error != 0) {
//		//	printf("\nERROR  in right hand side: %d", error);
//		//	exit(1);
//		//}
//
//		///* -------------------------------------------------------------------- */
//		///* .. pardiso_printstats(...)                                           */
//		///*    prints information on the matrix to STDOUT.                       */
//		///*    Use this functionality only for debugging purposes                */
//		///* -------------------------------------------------------------------- */
//
//		//pardiso_printstats(&mtype, &n, a, ia, ja, &nrhs, b, &error);
//		//if (error != 0) {
//		//	printf("\nERROR right hand side: %d", error);
//		//	exit(1);
//		//}
//	}
//	void releaseMemory() {
//		/* -------------------------------------------------------------------- */
//		/* ..  Convert matrix back to 0-based C-notation.                       */
//		/* -------------------------------------------------------------------- */
//		/*for (i = 0; i < n + 1; i++) {
//			ia[i] -= 1;
//		}
//		for (i = 0; i < nnz; i++) {
//			ja[i] -= 1;
//		}*/
//
//		/* -------------------------------------------------------------------- */
//		/* ..  Termination and release of memory.                               */
//		/* -------------------------------------------------------------------- */
//		phase = -1;                 /* Release internal memory. */
//
//		pardiso(pt, &maxfct, &mnum, &mtype, &phase,
//			&n, &ddum, ia, ja, &idum, &nrhs,
//			iparm, &msglvl, &ddum, &ddum, &error, dparm);
//	}
//	pardisoSolver() {
//		//variables
//		init();
//
//		//updateMatrixAndRHS();
//
//		//debugVectorAndMatrix(); 
//
//
//		/* -------------------------------------------------------------------- */
//		/* ..  Reordering and Symbolic Factorization.  This step also allocates */
//		/*     all memory that is necessary for the factorization.              */
//		/* -------------------------------------------------------------------- */
//		phase = 11;
//
//		pardiso(pt, &maxfct, &mnum, &mtype, &phase,
//			&n, a, ia, ja, &idum, &nrhs,
//			iparm, &msglvl, &ddum, &ddum, &error, dparm);
//
//		if (error != 0) {
//			printf("\nERROR during symbolic factorization: %d", error);
//			exit(1);
//		}
//		printf("\nReordering completed ... ");
//		printf("\nNumber of nonzeros in factors  = %d", iparm[17]);
//		printf("\nNumber of factorization MFLOPS = %d", iparm[18]);
//
//		/* -------------------------------------------------------------------- */
//		/* ..  Numerical factorization.                                         */
//		/* -------------------------------------------------------------------- */
//		phase = 22;
//		//iparm[32] = 1; /* compute determinant */
//		iparm[32] = 0;
//
//		pardiso(pt, &maxfct, &mnum, &mtype, &phase,
//			&n, a, ia, ja, &idum, &nrhs,
//			iparm, &msglvl, &ddum, &ddum, &error, dparm);
//
//		if (error != 0) {
//			printf("\nERROR during numerical factorization: %d", error);
//			exit(2);
//		}
//		printf("\nFactorization completed ...\n ");
//
//		/* -------------------------------------------------------------------- */
//		/* ..  Back substitution and iterative refinement.                      */
//		/* -------------------------------------------------------------------- */
//		phase = 33;
//
//		iparm[7] = 1;       /* Max numbers of iterative refinement steps. */
//
//		pardiso(pt, &maxfct, &mnum, &mtype, &phase,
//			&n, a, ia, ja, &idum, &nrhs,
//			iparm, &msglvl, b, x, &error, dparm);
//
//		if (error != 0) {
//			printf("\nERROR during solution: %d", error);
//			exit(3);
//		}
//
//		printf("\nSolve completed ... ");
//		printf("\nThe solution of the system is: ");
//		for (int i = 0; i < n; i++) {
//			printf("\n x [%d] = % f", i, x[i]);
//		}
//		printf("\n");
//
//		releaseMemory();
//	}
//	~pardisoSolver() {
//
//	}
//
//	void set_pattern(const std::vector<int> &II, const std::vector<int> &JJ, const std::vector<double> &SS);
//	void analyze_pattern();
//	void factorize(const std::vector<int> &II, const std::vector<int> &JJ, const std::vector<double> &SS);
//	void perpareMatrix(const std::vector<int> &II, const std::vector<int> &JJ, const std::vector<double> &SS);
//	Eigen::VectorXd solve(Eigen::VectorXd &rhs);
//};
//
//
//int main(void)
//{
//	Eigen::SparseMatrix<double> mat(8,8);
//	mat.insert(0, 0) = 7;
//	mat.insert(0, 2) = 1;
//	mat.insert(0, 5) = 2;
//	mat.insert(0, 6) = 7;
//	mat.insert(1, 1) = -4;
//	mat.insert(1, 2) = 8;
//	mat.insert(1, 4) = 2;
//	mat.insert(2, 2) = 1;
//	mat.insert(2, 7) = 5;
//	mat.insert(3, 3) = 7;
//	mat.insert(3, 6) = 9;
//	mat.insert(4, 4) = 5;
//	mat.insert(4, 5) = -1;
//	mat.insert(4, 6) = 5;
//	mat.insert(5, 5) = 0;
//	mat.insert(5, 7) = 5;
//	mat.insert(6, 6) = 11;
//	mat.insert(7, 7) = 5;
//	//mat.makeCompressed();
//	//////////////////////////////////
//	
//	
//	std::cout << mat;
//
//	int    n = mat.cols();
//	int    nnz = mat.nonZeros();
//	int* ia = new int[n + 1];
//	int* ja = new int[nnz];
//	double* a = new double[nnz];
//
//	mat = mat.transpose();
//	int currNNZ = 0;
//	ia[0] = Fortran_1_based(currNNZ);
//	for (int k = 0; k < mat.outerSize(); ++k)
//	{
//		for (Eigen::SparseMatrix<double>::InnerIterator it(mat, k); it; ++it)
//		{
//			ja[currNNZ] = Fortran_1_based(it.row());
//			a[currNNZ] = it.value();
//			currNNZ++;
//		}
//		ia[k+1] = Fortran_1_based(currNNZ);
//	}
//
//	/////////////////////////////////////////
//	cout << "ia = \n";
//	for (int i = 0; i < n + 1; i++) {
//		cout << ia[i] << " ";
//	}
//	cout << endl << "ja = \n";
//	
//	for (int i = 0; i < nnz; i++) {
//		cout << ja[i] << " ";
//	}
//	cout << endl << "a = \n";
//	
//	for (int i = 0; i < nnz; i++) {
//		cout << a[i] << " ";
//	}
//	cout << endl;
//	
//	
//	//pardisoSolver p;
//
//	delete[] ia;
//	delete[] ja;
//	delete[] a;
//	return 0;
//}


