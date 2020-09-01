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


namespace OptimizationUtils
{
	enum InitAuxVariables {
		SPHERE = 0,
		MESH_CENTER,
		MINUS_NORMALS
	};

	enum FunctionType {
		QUADRATIC = 0,
		EXPONENTIAL = 1,
		SIGMOID = 2
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

	static void Least_Squares_Sphere_Fit(
		const Eigen::MatrixXd& V,
		const Eigen::MatrixXi& F,
		Eigen::MatrixXd& center0,
		Eigen::VectorXd& radius0)
	{
		//for more info:
		//https://jekel.me/2015/Least-Squares-Sphere-Fit/
		center0.resize(F.rows(), 3);
		radius0.resize(F.rows(), 1);
		std::vector<std::vector<int>> adjacency = get_adjacency_vertices_per_face(V, F);
		for (int fi = 0; fi < F.rows(); fi++) {
			std::vector<int> f_adj = adjacency[fi];
			int n = f_adj.size();
			Eigen::MatrixXd A(n, 4);
			Eigen::VectorXd c(4), f(n);
			for (int ni = 0; ni < n; ni++) {
				double xi = V(f_adj[ni], 0);
				double yi = V(f_adj[ni], 1);
				double zi = V(f_adj[ni], 2);
				A.row(ni) << 2 * xi, 2 * yi, 2 * zi, 1;
				f(ni) = pow(xi, 2) + pow(yi, 2) + pow(zi, 2);
			}
			//solve Ac = f and get c!
			c = (A.transpose()*A).colPivHouseholderQr().solve(A.transpose()*f);

			//for debugging
			/*std::cout << "A:\n" << A << std::endl;
			std::cout << "f:\n" << f << std::endl;
			std::cout << "c:\n" << c << std::endl;
			std::cout << "MSE:\n" << (A*c-f).squaredNorm() << std::endl;*/

			//after we got the solution c we pick from c: radius & center=(X,Y,Z)
			center0.row(fi) << c(0), c(1), c(2);
			radius0(fi) = sqrt(c(3) + pow(c(0), 2) + pow(c(1), 2) + pow(c(2), 2));
		}
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
	public:
		Timer() {
			start = std::chrono::high_resolution_clock::now();
		}
		~Timer() {
			end = std::chrono::high_resolution_clock::now();
			duration = end - start;
			double ms = duration.count() * 1000.0f;
			std::cout << "Timer took " << ms << "ms\n";
		}
	};
}
