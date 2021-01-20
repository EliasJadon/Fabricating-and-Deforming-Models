#pragma once

#include <direct.h>
#include <iostream>
#include <igl/doublearea.h>
#include <igl/local_basis.h>
#include <igl/boundary_loop.h>
#include <igl/per_face_normals.h>
#include <windows.h>
#include <Eigen/sparse>
#include <igl/vertex_triangle_adjacency.h>
#include <chrono>
#include <igl/triangle_triangle_adjacency.h>
#include <set>
#include <igl/PI.h>

namespace OptimizationUtils
{
	enum InitAuxVariables {
		SPHERE_FIT = 0,
		MODEL_CENTER_POINT,
		MINUS_NORMALS,
		CYLINDER_FIT
	};

	enum LineSearch {
		GRADIENT_NORM,
		FUNCTION_VALUE,
		CONSTANT_STEP
	};

	static Eigen::SparseMatrix<double> BuildMatrix(const std::vector<int>& I, const std::vector<int>& J, const std::vector<double>& S) {
		assert(I.size() == J.size() && I.size() == S.size() && "II,JJ,SS must have the same size!");
		std::vector<Eigen::Triplet<double>> tripletList;
		tripletList.reserve(I.size());
		int rows = *std::max_element(I.begin(), I.end()) + 1;
		int cols = *std::max_element(J.begin(), J.end()) + 1;

		for (int i = 0; i < I.size(); i++)
			tripletList.push_back(Eigen::Triplet<double>(I[i], J[i], S[i]));

		Eigen::SparseMatrix<double> A;
		A.resize(rows, cols);
		A.setFromTriplets(tripletList.begin(), tripletList.end());
		A.makeCompressed();
		return A;
	}

	static void LocalBasis(const Eigen::MatrixX3d &V, const Eigen::MatrixX3i &F, Eigen::MatrixX3d &B1, Eigen::MatrixX3d &B2) {
		Eigen::MatrixX3d B3;
		igl::local_basis(V, F, B1, B2, B3);
	}

	
	static void computeSurfaceGradientPerFace(const Eigen::MatrixX3d &V, const Eigen::MatrixX3i &F, Eigen::MatrixX3d &D1, Eigen::MatrixX3d &D2)
	{
		Eigen::MatrixX3d F1, F2, F3;
		igl::local_basis(V, F, F1, F2, F3);
		const int Fn = F.rows();  const int vn = V.rows();

		Eigen::MatrixX3d Dx(Fn,3), Dy(Fn, 3), Dz(Fn, 3);
		Eigen::MatrixX3d fN; igl::per_face_normals(V, F, fN);
		Eigen::VectorXd Ar; igl::doublearea(V, F, Ar);
		Eigen::PermutationMatrix<3> perm;

		Eigen::Vector3i Pi;
		Pi << 1, 2, 0;
		Eigen::PermutationMatrix<3> P = Eigen::PermutationMatrix<3>(Pi);

		for (int i = 0; i < Fn; i++) {
			// renaming indices of vertices of triangles for convenience
			int i1 = F(i, 0);
			int i2 = F(i, 1);
			int i3 = F(i, 2);

			// #F x 3 matrices of triangle edge vectors, named after opposite vertices
			Eigen::Matrix3d e;
			e.col(0) = V.row(i2) - V.row(i1);
			e.col(1) = V.row(i3) - V.row(i2);
			e.col(2) = V.row(i1) - V.row(i3);;

			Eigen::Vector3d Fni = fN.row(i);
			double Ari = Ar(i);

			//grad3_3f(:,[3*i,3*i-2,3*i-1])=[0,-Fni(3), Fni(2);Fni(3),0,-Fni(1);-Fni(2),Fni(1),0]*e/(2*Ari);
			Eigen::Matrix3d n_M;
			n_M << 0, -Fni(2), Fni(1), Fni(2), 0, -Fni(0), -Fni(1), Fni(0), 0;
			Eigen::VectorXi R(3); R << 0, 1, 2;
			Eigen::VectorXi C(3); C << 3 * i + 2, 3 * i, 3 * i + 1;
			Eigen::Matrix3d res = ((1. / Ari)*(n_M*e))*P;

			Dx.row(i) = res.row(0);
			Dy.row(i) = res.row(1);
			Dz.row(i) = res.row(2);
		}
		D1 = F1.col(0).asDiagonal()*Dx + F1.col(1).asDiagonal()*Dy + F1.col(2).asDiagonal()*Dz;
		D2 = F2.col(0).asDiagonal()*Dx + F2.col(1).asDiagonal()*Dy + F2.col(2).asDiagonal()*Dz;
	}
	
	static inline void SSVD2x2(const Eigen::Matrix2d& A, Eigen::Matrix2d& U, Eigen::Matrix2d& S, Eigen::Matrix2d& V)
	{
		double e = (A(0) + A(3))*0.5;
		double f = (A(0) - A(3))*0.5;
		double g = (A(1) + A(2))*0.5;
		double h = (A(1) - A(2))*0.5;
		double q = sqrt((e*e) + (h*h));
		double r = sqrt((f*f) + (g*g));
		double a1 = atan2(g, f);
		double a2 = atan2(h, e);
		double rho = (a2 - a1)*0.5;
		double phi = (a2 + a1)*0.5;

		S(0) = q + r;
		S(1) = 0;
		S(2) = 0;
		S(3) = q - r;

		double c = cos(phi);
		double s = sin(phi);
		U(0) = c;
		U(1) = s;
		U(2) = -s;
		U(3) = c;

		c = cos(rho);
		s = sin(rho);
		V(0) = c;
		V(1) = -s;
		V(2) = s;
		V(3) = c;
	}

	// The directory path returned by native GetCurrentDirectory() no end backslash
	static std::string getCurrentDirectoryOnWindows()
	{
		const unsigned long maxDir = 260;
		char currentDir[maxDir];
		GetCurrentDirectory(maxDir, currentDir);
		return std::string(currentDir);
	}

	static std::string workingdir() {
		char buf[256];
		GetCurrentDirectoryA(256, buf);
		return std::string(buf) + '\\';
	}

	static std::string ExePath() {
		char buffer[MAX_PATH];
		GetModuleFileName(NULL, buffer, MAX_PATH);
		std::string::size_type pos = std::string(buffer).find_last_of("\\/");
		return std::string(buffer).substr(0, pos);
	}

	static std::string ProjectPath() {
		char buffer[MAX_PATH];
		GetModuleFileName(NULL, buffer, MAX_PATH);
		std::string::size_type pos = std::string(buffer).find("\\MappingsLab\\");
		return std::string(buffer).substr(0, pos + 11 + 2);
	}

	static std::vector<int> temp_get_one_ring_vertices_per_vertex(const Eigen::MatrixXi& F, const std::vector<int>& OneRingFaces) {
		std::vector<int> vertices;
		vertices.clear();
		for (int i = 0; i < OneRingFaces.size(); i++) {
			int fi = OneRingFaces[i];
			int P0 = F(fi, 0);
			int P1 = F(fi, 1);
			int P2 = F(fi, 2);

			//check if the vertex already exist
			if (!(find(vertices.begin(), vertices.end(), P0) != vertices.end())) {
				vertices.push_back(P0);
			}
			if (!(find(vertices.begin(), vertices.end(), P1) != vertices.end())) {
				vertices.push_back(P1);
			}
			if (!(find(vertices.begin(), vertices.end(), P2) != vertices.end())) {
				vertices.push_back(P2);
			}
		}
		return vertices;
	}

	static std::vector<std::vector<int>> get_one_ring_vertices(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F) {
		// adjacency matrix (vertex to face)
		std::vector<std::vector<int> > VF, VFi;
		std::vector<std::vector<int>> OneRingVertices;
		OneRingVertices.resize(V.rows());

		igl::vertex_triangle_adjacency(V, F, VF, VFi);


		for (int vi = 0; vi < V.rows(); vi++) {
			std::vector<int> OneRingFaces = VF[vi];
			OneRingVertices[vi] = temp_get_one_ring_vertices_per_vertex(F, OneRingFaces);
		}
		return OneRingVertices;
	}

	static std::vector<std::vector<int>> get_adjacency_vertices_per_face(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F) {
		std::vector<std::vector<int>> OneRingVertices = get_one_ring_vertices(V, F);
		std::vector<std::vector<int>> adjacency;
		adjacency.resize(F.rows());
		for (int fi = 0; fi < F.rows(); fi++) {
			adjacency[fi].clear();
			for (int vi = 0; vi < 3; vi++) {
				int Vi = F(fi, vi);
				std::vector<int> vi_oneRing = OneRingVertices[Vi];
				for (int v_index : vi_oneRing)
					if (!(find(adjacency[fi].begin(), adjacency[fi].end(), v_index) != adjacency[fi].end()))
						adjacency[fi].push_back(v_index);
			}
		}

		////for debugging
		//for (int fi = 0; fi < adjacency.size(); fi++) {
		//	std::cout << console_color::blue << "----------face " << fi << ":\n";
		//	for (int v : adjacency[fi]) {
		//		std::cout << v << " ";
		//	}
		//	std::cout << std::endl;
		//}
		//std::cout << console_color::white;

		return adjacency;
	}

	static void center_of_mesh(
		const Eigen::MatrixXd& V,
		const Eigen::MatrixXi& F,
		Eigen::MatrixXd& center0,
		Eigen::VectorXd& radius0)
	{
		center0.resize(F.rows(), 3);
		radius0.resize(F.rows(), 1);
		Eigen::Vector3d avg;
		avg.setZero();
		for (int vi = 0; vi < V.rows(); vi++)
			avg += V.row(vi);
		avg /= V.rows();

		//update center0
		center0.col(0) = Eigen::VectorXd::Constant(F.rows(), avg(0));
		center0.col(1) = Eigen::VectorXd::Constant(F.rows(), avg(1));
		center0.col(2) = Eigen::VectorXd::Constant(F.rows(), avg(2));

		//update radius0
		for (int fi = 0; fi < F.rows(); fi++) {
			int x0 = F(fi, 0);
			int x1 = F(fi, 1);
			int x2 = F(fi, 2);
			Eigen::VectorXd x = (V.row(x0) + V.row(x1) + V.row(x2)) / 3;
			radius0(fi) = (x - avg).norm();
		}
	}

	static std::vector<std::set<int>> Triangle_triangle_adjacency(const Eigen::MatrixX3i& F) {
		std::vector<std::vector<std::vector<int>>> TT;
		igl::triangle_triangle_adjacency(F, TT);
		assert(TT.size() == F.rows());
		std::vector<std::set<int>> neigh; neigh.clear();

		for (int fi = 0; fi < TT.size(); fi++) {
			assert(TT[fi].size() == 3 && "Each face should be a triangle (not square for example)!");
			std::set<int> neigh_faces; neigh_faces.clear();
			neigh_faces.insert(fi);
			for (std::vector<int> hinge : TT[fi])
				for (int Face_neighbor : hinge)
					neigh_faces.insert(Face_neighbor);
			neigh.push_back(neigh_faces);
		}
		return neigh;
	}

	static Eigen::MatrixX3d Vertices_Neighbors(
		const int fi,
		const int distance,
		const Eigen::MatrixXd& V,
		const std::vector<std::set<int>>& TT,
		const std::vector<std::vector<int>>& TV)
	{
		std::set<int> faces;
		if (distance < 1) {
			std::cout << "Error! Distance should be 1 or Greater! (OptimizationUtils::Vertices_Neighbors)";
			exit(1);
		}
		else {
			faces = { fi };
			for (int i = 1; i < distance; i++) {
				std::set<int> currfaces = faces;
				for (int neighF : currfaces)
					faces.insert(TT[neighF].begin(), TT[neighF].end());
			}
		}

		std::set<int> neigh; neigh.clear();
		for (int currF : faces)
			for (int n : TV[currF])
				neigh.insert(n);

		Eigen::MatrixX3d neigh_vertices(neigh.size(), 3);
		int i = 0;
		for (int vi : neigh) {
			neigh_vertices.row(i++) = V.row(vi);
		}
		return neigh_vertices;
	}

	static double Least_Squares_Sphere_Fit_perFace(
		const int fi,
		const Eigen::MatrixXd& V,
		const Eigen::MatrixXi& F,
		const Eigen::MatrixX3d& vertices_indices,
		Eigen::MatrixXd& center0,
		Eigen::VectorXd& radius0)
	{
		//for more info:
		//https://jekel.me/2015/Least-Squares-Sphere-Fit/
		const int n = vertices_indices.rows();
		Eigen::MatrixXd A(n, 4);
		Eigen::VectorXd c(4), f(n);
		for (int ni = 0; ni < n; ni++) {
			const double xi = vertices_indices(ni, 0);
			const double yi = vertices_indices(ni, 1);
			const double zi = vertices_indices(ni, 2);
			A.row(ni) << 2 * xi, 2 * yi, 2 * zi, 1;
			f(ni) = pow(xi, 2) + pow(yi, 2) + pow(zi, 2);
		}
		//solve Ac = f and get c!
		c = (A.transpose() * A).colPivHouseholderQr().solve(A.transpose() * f);
		//after we got the solution c we pick from c: radius & center=(X,Y,Z)
		center0.row(fi) << c(0), c(1), c(2);
		radius0(fi) = sqrt(c(3) + pow(c(0), 2) + pow(c(1), 2) + pow(c(2), 2));
		//calculate MSE
		double toatal_MSE = 0;
		for (int ni = 0; ni < n; ni++)
			toatal_MSE += pow((vertices_indices.row(ni) - center0.row(fi)).squaredNorm() - pow(radius0(fi), 2), 2);
		toatal_MSE /= n;
		return toatal_MSE;
	}

	static void Least_Squares_Sphere_Fit(
		const int Distance,
		const Eigen::MatrixXd& V,
		const Eigen::MatrixXi& F,
		Eigen::MatrixXd& center0,
		Eigen::VectorXd& radius0)
	{
		//for more info:
		//https://jekel.me/2015/Least-Squares-Sphere-Fit/
		center0.resize(F.rows(), 3);
		radius0.resize(F.rows(), 1);
		std::vector<std::vector<int>> TV = get_adjacency_vertices_per_face(V, F);
		std::vector<std::set<int>> TT = Triangle_triangle_adjacency(F);
		
		for (int fi = 0; fi < F.rows(); fi++) {
			double minMSE = std::numeric_limits<double>::infinity();
			int argmin = -1;
			for (int d = 1; d <= Distance; d++) {
				double currMSE = Least_Squares_Sphere_Fit_perFace(fi, V, F,
					Vertices_Neighbors(fi, d,V, TT, TV),
					center0, radius0);
				if (currMSE < minMSE) {
					minMSE = currMSE;
					argmin = d;
				}
			}
			std::cout << "fi =\t" << fi << "\t, argmin = " << argmin<< "\t, minMSE = " << minMSE << std::endl;
			Least_Squares_Sphere_Fit_perFace(fi, V, F,
				Vertices_Neighbors(fi, argmin,V, TT, TV),
				center0, radius0);
		}
	}

	static void Preprocess(
		const int n, 
		const Eigen::MatrixX3d& points,
		Eigen::MatrixX3d& X,
		Eigen::Vector3d& average, 
		Eigen::VectorXd& mu,
		Eigen::Matrix<double, 3, 3>& F0,
		Eigen::Matrix<double, 3, 6>& F1,
		Eigen::Matrix<double, 6, 6>& F2)
	{
		average << 0, 0, 0;
		for(int i = 0; i < n; ++i)
		{
			average += points.row(i).transpose();
		}
		average /= n;
		for(int i = 0; i < n; ++i)
		{
			X.row(i) = points.row(i) - average.transpose();
		}
		Eigen::MatrixXd products(n, 6);
		products.setZero();
		mu.resize(6);
		mu.setZero();
		for(int i = 0; i < n; ++i)
		{
			products(i, 0) = X(i, 0) * X(i, 0);
			products(i, 1) = X(i, 0) * X(i, 1);
			products(i, 2) = X(i, 0) * X(i, 2);
			products(i, 3) = X(i, 1) * X(i, 1);
			products(i, 4) = X(i, 1) * X(i, 2);
			products(i, 5) = X(i, 2) * X(i, 2);
			mu[0] += products(i, 0);
			mu[1] += 2 * products(i, 1);
			mu[2] += 2 * products(i, 2);
			mu[3] += products(i, 3);
			mu[4] += 2 * products(i, 4);
			mu[5] += products(i, 5);
		}
		mu /= n;
		F0.setZero();
		F1.setZero();
		F2.setZero();
		for(int i = 0; i < n; ++i)
		{
			Eigen::RowVectorXd delta(6);
			delta[0] = products(i, 0)		- mu[0];
			delta[1] = 2 * products(i, 1)	- mu[1];
			delta[2] = 2 * products(i, 2)	- mu[2];
			delta[3] = products(i, 3)		- mu[3];
			delta[4] = 2 * products(i, 4)	- mu[4];
			delta[5] = products(i, 5)		- mu[5];
			F0(0, 0) += products(i, 0);
			F0(0, 1) += products(i, 1);
			F0(0, 2) += products(i, 2);
			F0(1, 1) += products(i, 3);
			F0(1, 2) += products(i, 4);
			F0(2, 2) += products(i, 5);
			F1 += X.row(i).transpose() * delta;
			F2 += delta.transpose() * delta;
		}
		F0 /= n;
		F0(1, 0) = F0(0, 1);
		F0(2, 0) = F0(0, 2);
		F0(2, 1) = F0(1, 2);
		F1 /= n;
		F2 /= n;
	}


	static double G(
		const int n,
		const Eigen::MatrixX3d& X,
		const Eigen::VectorXd& mu,
		const Eigen::Matrix<double, 3, 3>& F0,
		const Eigen::Matrix<double, 3, 6>& F1,
		const Eigen::Matrix<double, 6, 6>& F2,
		const Eigen::Vector3d& W,
		Eigen::Vector3d& PC,
		double& rSqr)
	{
		Eigen::Matrix<double, 3, 3> P, S, A, hatA, hatAA, Q;
		// P = I - W * WˆT
		P = Eigen::Matrix<double, 3, 3>::Identity() - W * W.transpose();
		S <<
			0		, -W[2]	, W[1]	,
			W[2]	, 0		, -W[0]	,
			-W[1]	, W[0]	, 0;
		A = P * F0 * P;
		hatA = - (S * A * S);
		hatAA = hatA * A;
		Q = hatA / hatAA.trace();
		Eigen::VectorXd p(6);
		p << P(0, 0), P(0, 1), P(0, 2), P(1, 1), P(1, 2), P(2, 2);
		Eigen::Vector3d alpha = F1 * p;
		Eigen::Vector3d beta = Q * alpha;
		double error = (p.dot(F2 * p) - 4 * alpha.dot(beta) + 4 * beta.dot(F0 * beta)) / n;
		PC = beta;
		rSqr = p.dot(mu) + beta.dot(beta);
		return error;
	}

	// The X[] are the points to be fit. The outputs rSqr , C, and W are the
	// cylinder parameters. The function return value is the error function
	// evaluated at the cylinder parameters.
	static double FitCylinder(
		const int n, 
		const Eigen::MatrixX3d& points, 
		double& rSqr, 
		Eigen::Vector3d& C,
		Eigen::Vector3d& W,
		const int imax,
		const int jmax)
	{
		//For more info:
		// https://www.geometrictools.com/Documentation/LeastSquaresFitting.pdf
		Eigen::MatrixX3d X(n,3);
		Eigen::VectorXd mu(6);
		Eigen::Vector3d average;
		Eigen::Matrix<double, 3, 3> F0;
		Eigen::Matrix<double, 3, 6> F1;
		Eigen::Matrix<double, 6, 6> F2;

		Preprocess(n, points, X, average, mu, F0, F1, F2);
		// Choose imax and jmax as desired for the level of granularity you
		// want for sampling W vectors on the hemisphere.
		double minError = std::numeric_limits<double>::infinity();
		W = Eigen::Vector3d::Zero();
		C = Eigen::Vector3d::Zero();
		rSqr = 0;
		for(int j = 0; j <= jmax; ++j)
		{
			double PI = 3.14159265358979323846;
			double phi = (0.5 * PI * j) / jmax; // in [0, pi/2]
			double csphi = cos(phi), snphi = sin(phi);
			for(int i = 0; i < imax; ++i)
			{
				double theta = (2.0f * PI * i) / imax; // in [0, 2*pi)
				double cstheta = cos(theta);
				double sntheta = sin(theta);
				Eigen::Vector3d currentW(cstheta * snphi, sntheta * snphi, csphi);
				Eigen::Vector3d currentC;
				double currentRSqr;
				double error = G(n, X, mu, F0, F1, F2, currentW, currentC, currentRSqr);
				if(error < minError)
				{
					minError = error;
					W = currentW;
					C = currentC;
					rSqr = currentRSqr;
				}
			}
		}
		// Translate the center to the original coordinate system.
		C += average;
		return minError;
	}
	static void Least_Squares_Cylinder_Fit(
		const int imax,
		const int jmax,
		const int Distance,
		const Eigen::MatrixXd& V,
		const Eigen::MatrixXi& F,
		Eigen::MatrixXd& center0,
		Eigen::MatrixXd& dir0,
		Eigen::VectorXd& radius0)
	{
		center0.resize(F.rows(), 3);
		dir0.resize(F.rows(), 3);
		radius0.resize(F.rows(), 1);
		std::vector<std::vector<int>> TV = get_adjacency_vertices_per_face(V, F);
		std::vector<std::set<int>> TT = Triangle_triangle_adjacency(F);
		
		for (int fi = 0; fi < F.rows(); fi++) {
			double minMSE = std::numeric_limits<double>::infinity();
			int argmin = -1;
			for (int d = 1; d <= Distance; d++) {
				const Eigen::MatrixX3d& points = Vertices_Neighbors(fi, d, V, TT, TV);
				double rSqr;
				Eigen::Vector3d C;
				Eigen::Vector3d W;
				double currMSE = FitCylinder(points.rows(), points, rSqr, C, W, imax, jmax);
				if (currMSE < minMSE) {
					minMSE = currMSE;
					argmin = d;
					dir0.row(fi) = (W.normalized()).transpose();
					center0.row(fi) = C.transpose();
					radius0(fi) = sqrt(rSqr);
				}
			}
			std::cout << "fi =\t" << fi << "\t, argmin = " << argmin << "\t, minMSE = " << minMSE << std::endl;
			//std::cout << "center0 = " << center0.row(fi) << std::endl;
			//std::cout << "radius0 = " << radius0(fi) << std::endl;
			//std::cout << "dir0 = " << dir0.row(fi) << std::endl;
		}
	}


	static std::vector<std::vector<int>> Get_adjacency_vertices_per_face(
		const Eigen::MatrixXd& V,
		const Eigen::MatrixXi& F)
	{
		return get_adjacency_vertices_per_face(V, F);
	}

	
	static Eigen::MatrixXd center_per_triangle(const Eigen::MatrixXd& V,const Eigen::MatrixXi& F)
	{
		Eigen::MatrixXd centers(F.rows(), 3);
		for (int fi = 0; fi < F.rows(); fi++) {
			int x0 = F(fi, 0);
			int x1 = F(fi, 1);
			int x2 = F(fi, 2);
			centers.row(fi) = (V.row(x0) + V.row(x1) + V.row(x2)) / 3;
		}
		return centers;
	}

	class Timer {
	private:
		std::chrono::time_point<std::chrono::steady_clock> start, end;
		std::chrono::duration<double> duration;
		double* sum;
		double* curr;
	public:
		Timer(double* sum, double* current) {
			start = std::chrono::high_resolution_clock::now();
			this->sum = sum;
			this->curr = current;
		}
		~Timer() {
			end = std::chrono::high_resolution_clock::now();
			duration = end - start;
			double ms = duration.count() * 1000.0f;
			*sum += ms;
			*curr = ms;
			std::cout << "Timer took " << ms << "ms\n";
		}
	};
}
