#include "linear_equation_solvers/tryPardiso.h"

pardisoSolver::pardisoSolver() {
	/* Matrix data. */
	int    n = 8;
	int      nnz = 18;
	int      mtype = -2;        /* Real symmetric matrix */

	/* RHS and solution vectors. */
	double   b[8], x[8];
	int      nrhs = 1;          /* Number of right hand sides. */

	/* Internal solver memory pointer pt,                  */
	/* 32-bit: int pt[64]; 64-bit: long int pt[64]         */
	/* or void *pt[64] should be OK on both architectures  */
	void    *pt[64];

	/* Pardiso control parameters. */
	int      iparm[64];
	double   dparm[64];
	int      maxfct, mnum, phase, error, msglvl, solver;

	/* Number of processors. */
	int      num_procs;

	/* Auxiliary variables. */
	char    *var;
	int      i;

	double   ddum;              /* Double dummy */
	int      idum;              /* Integer dummy. */

	
/* -------------------------------------------------------------------- */
/* ..  Setup Pardiso control parameters.                                */
/* -------------------------------------------------------------------- */

	error = 0;
	solver = 0; /* use sparse direct solver */
	pardisoinit(pt, &mtype, &solver, iparm, dparm, &error);
	int    ia[9] = { 0, 4, 7, 9, 11, 14, 16, 17, 18 };
	int    ja[18] = { 0,    2,       5, 6,
						 1, 2,    4,
							2,             7,
							   3,       6,
								  4, 5, 6,
									 5,    7,
										6,
										   7 };
	double  a[18] = { 7.0,      1.0,           2.0, 7.0,
						  -4.0, 8.0,           2.0,
								1.0,                     5.0,
									 7.0,           9.0,
										  5.0, 1.0, 5.0,
											   0.0,      5.0,
												   11.0,
														 5.0 };
	if (error != 0)
	{
		if (error == -10)
			printf("No license file found \n");
		if (error == -11)
			printf("License is expired \n");
		if (error == -12)
			printf("Wrong username or hostname \n");
		exit(1);
	}
	else
		printf("[PARDISO]: License check was successful ... \n");

	iparm[2] = 1;// num_procs;

	maxfct = 1;		/* Maximum number of numerical factorizations.  */
	mnum = 1;         /* Which factorization to use. */

	msglvl = 1;         /* Print statistical information  */
	error = 0;         /* Initialize error flag */

/* -------------------------------------------------------------------- */
/* ..  Convert matrix from 0-based C-notation to Fortran 1-based        */
/*     notation.                                                        */
/* -------------------------------------------------------------------- */
	
	
	for (i = 0; i < n + 1; i++) {
		ia[i] += 1;
	}
	for (i = 0; i < nnz; i++) {
		ja[i] += 1;
	}

	/* Set right hand side to one. */
	for (i = 0; i < n; i++) {
		b[i] = i;
	}

	/* -------------------------------------------------------------------- */
	/* ..  Reordering and Symbolic Factorization.  This step also allocates */
	/*     all memory that is necessary for the factorization.              */
	/* -------------------------------------------------------------------- */
	phase = 11;
	pardiso(pt, &maxfct, &mnum, &mtype, &phase,
		&n, a, ia, ja, &idum, &nrhs,
		iparm, &msglvl, &ddum, &ddum, &error, dparm);

	if (error != 0) {
		printf("\nERROR during symbolic factorization: %d", error);
		exit(1);
	}
	printf("\nReordering completed ... ");
	printf("\nNumber of nonzeros in factors  = %d", iparm[17]);
	printf("\nNumber of factorization MFLOPS = %d", iparm[18]);

	/* -------------------------------------------------------------------- */
	/* ..  Numerical factorization.                                         */
	/* -------------------------------------------------------------------- */
	phase = 22;
	iparm[32] = 1; /* compute determinant */

	pardiso(pt, &maxfct, &mnum, &mtype, &phase,
		&n, a, ia, ja, &idum, &nrhs,
		iparm, &msglvl, &ddum, &ddum, &error, dparm);

	if (error != 0) {
		printf("\nERROR during numerical factorization: %d", error);
		exit(2);
	}
	printf("\nFactorization completed ...\n ");

	/* -------------------------------------------------------------------- */
	/* ..  Back substitution and iterative refinement.                      */
	/* -------------------------------------------------------------------- */
	phase = 33;

	iparm[7] = 1;       /* Max numbers of iterative refinement steps. */

	pardiso(pt, &maxfct, &mnum, &mtype, &phase,
		&n, a, ia, ja, &idum, &nrhs,
		iparm, &msglvl, b, x, &error, dparm);

	if (error != 0) {
		printf("\nERROR during solution: %d", error);
		exit(3);
	}

	printf("\nSolve completed ... ");
	printf("\nThe solution of the system is: ");
	for (i = 0; i < n; i++) {
		printf("\n x [%d] = % f", i, x[i]);
	}
	printf("\n");

	/* -------------------------------------------------------------------- */
	/* ..  Convert matrix back to 0-based C-notation.                       */
	/* -------------------------------------------------------------------- */
	for (i = 0; i < n + 1; i++) {
		ia[i] -= 1;
	}
	for (i = 0; i < nnz; i++) {
		ja[i] -= 1;
	}

	/* -------------------------------------------------------------------- */
	/* ..  Termination and release of memory.                               */
	/* -------------------------------------------------------------------- */
	phase = -1;                 /* Release internal memory. */

	pardiso(pt, &maxfct, &mnum, &mtype, &phase,
		&n, &ddum, ia, ja, &idum, &nrhs,
		iparm, &msglvl, &ddum, &ddum, &error, dparm);

}


pardisoSolver::~pardisoSolver() {

}
