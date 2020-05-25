#pragma once

#include <direct.h>
#include <iostream>
#include <igl/doublearea.h>
#include <igl/local_basis.h>
#include <igl/boundary_loop.h>
#include <igl/per_face_normals.h>
#include <windows.h>
#include <Eigen/sparse>
class OptimizationUtils
{
public:
	enum FunctionType {
		Quadratic = 0,
		Exponential = 1,
		PlanarL = 2
	};

	static enum class LineSearch {
		GradientNorm,
		FunctionValue,
		ConstantStep
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

	static std::string RDSPath() {
		char buffer[MAX_PATH];
		GetModuleFileName(NULL, buffer, MAX_PATH);
		std::string::size_type pos = std::string(buffer).find("\\RDS\\");
		return std::string(buffer).substr(0, pos + 5);
	}
};
