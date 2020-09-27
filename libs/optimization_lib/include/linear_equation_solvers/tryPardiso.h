#pragma once

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//#include <vector>
//#include <Eigen/Core>
//#include <Eigen/Sparse>

using namespace std;

/* PARDISO prototype. */
extern "C" void pardisoinit(void   *, int    *, int *, int *, double *, int *);
extern "C" void pardiso(void   *, int    *, int *, int *, int *, int *,
	double *, int    *, int *, int *, int *, int *,
	int *, double *, double *, int *, double *);
extern "C" void pardiso_chkmatrix(int *, int *, double *, int *, int *, int *);
extern "C" void pardiso_chkvec(int *, int *, double *, int *);
extern "C" void pardiso_printstats(int *, int *, double *, int *, int *, int *,
	double *, int *);

class pardisoSolver
{
private:
	        
public:
	pardisoSolver();
	~pardisoSolver();
};

