#include "ROY_soup2D.h"

// C includes
#include <corecrt_math_defines.h>
#include <tuple>
#include <time.h>

// STL includes
//#include <ranges>
#include <queue>

// LIBIGL includes
#include <igl/slice.h>
#include <igl/local_basis.h>
#include <igl/doublearea.h>
#include <igl/per_face_normals.h>
#include <igl/adjacency_matrix.h>
#include <igl/readOFF.h>
#include <igl/readOBJ.h>

MeshWrapper::MeshWrapper(const Eigen::MatrixX3d& v, const Eigen::MatrixX3i& f) :
	v_dom_(v),
	f_dom_(f)
{
	Initialize();
}

MeshWrapper::~MeshWrapper()
{

}

void MeshWrapper::GenerateSoupFaces(const Eigen::MatrixX3i& f_in, Eigen::MatrixX3i& f_out)
{
	auto f_in_count = f_in.rows();
	auto f_out_count = 3 * f_in_count;
	Eigen::VectorXi lin = Eigen::VectorXi::LinSpaced(f_out_count, 0, f_out_count - 1);
	f_out = Eigen::Map<Eigen::Matrix3Xi>(lin.data(), 3, f_in_count).transpose();
}

void MeshWrapper::GenerateIsometric2DSoup(const Eigen::MatrixX3i& f_in, const Eigen::MatrixX3d& v_in, const ED2EIMap& ed_2_ei, const EI2FIsMap& ei_2_fi, Eigen::MatrixX3i& f_out, Eigen::MatrixX2d& v_out)
{
	std::vector<bool> face_visit_status;
	std::queue<std::tuple<RDS::FaceIndex, RDS::ProjectionDescriptor, RDS::ProjectionDescriptor>> queue;
	GenerateSoupFaces(f_in, f_out);
	v_out = Eigen::MatrixX2d::Zero(3 * f_out.rows(), 2);
	face_visit_status.resize(f_out.rows());

	for (int64_t i = 0; i < f_out.rows(); i++)
	{
		face_visit_status[i] = false;
	}
	
	auto first_face = f_in.row(0);
	RDS::EdgeIndex initial_edge_index = ed_2_ei.at(std::make_pair(first_face(0), first_face(1)));
	RDS::VertexIndex initial_v0_index = e_dom_(initial_edge_index, 0);
	RDS::VertexIndex initial_v1_index = e_dom_(initial_edge_index, 1);
	double edge_length = (v_dom_.row(e_dom_(initial_edge_index, 0)) - v_dom_.row(e_dom_(initial_edge_index, 1))).norm();
	Eigen::Vector2d initial_v0 = Eigen::Vector2d(0, 0);
	Eigen::Vector2d initial_v1 = Eigen::Vector2d(edge_length, 0);
	queue.push(std::make_tuple(0, std::make_pair(initial_v0_index, initial_v0), std::make_pair(initial_v1_index, initial_v1)));
	
	while (!queue.empty())
	{
		RDS::FaceIndex current_face_index = std::get<0>(queue.front());
		RDS::ProjectionDescriptor pair0_in = std::get<1>(queue.front());
		RDS::ProjectionDescriptor pair1_in = std::get<2>(queue.front());

		std::vector<RDS::ProjectionDescriptor> input_pairs;
		std::vector<RDS::ProjectionDescriptor> output_pairs;
		input_pairs.push_back(pair0_in);
		input_pairs.push_back(pair1_in);

		GetOrderedProjectedVertices(input_pairs, current_face_index, output_pairs);
		RDS::VertexIndex base_index = current_face_index * 3;
		RDS::VertexIndex base_vertex_index = f_dom_.coeff(current_face_index, 0);

		int bla;
		for (int i = 0; i < 3; i++)
		{
			if (output_pairs[i].first == base_vertex_index)
			{
				bla = i;
				break;
			}
		}


		for (int i = 0; i < 3; i++)
		{
			v_out.row(base_index + i) = output_pairs[((bla + i) % 3)].second;
		}

		face_visit_status[current_face_index] = true;
		queue.pop();

		for (int i = 0; i < 3; i++)
		{
			RDS::ProjectionDescriptor first_pair = output_pairs[i];
			RDS::ProjectionDescriptor second_pair = output_pairs[(i + 1) % 3];

			RDS::VertexIndex current_vertex_index = first_pair.first;
			RDS::VertexIndex next_vertex_index = second_pair.first;
			RDS::EdgeIndex edge_index = ed_2_ei.at(std::make_pair(current_vertex_index, next_vertex_index));
			auto adjacent_faces = ei_2_fi.at(edge_index);
			for (RDS::FaceIndex adjacent_face_index : adjacent_faces)
			{
				if (adjacent_face_index != current_face_index)
				{
					if (!face_visit_status[adjacent_face_index])
					{
						queue.push(std::make_tuple(adjacent_face_index, first_pair, second_pair));
					}
				}
			}
		}
	}
}

void MeshWrapper::GetOrderedProjectedVertices(const std::vector<RDS::ProjectionDescriptor>& input_pairs, RDS::FaceIndex face_index, std::vector<RDS::ProjectionDescriptor>& output_pairs)
{
	output_pairs.resize(3);

	const RDS::ProjectionDescriptor pair0 = input_pairs[0];
	const RDS::ProjectionDescriptor pair1 = input_pairs[1];
	RDS::ProjectionDescriptor pair2;

	for (int i = 0; i < 3; i++)
	{
		const RDS::VertexIndex current_vertex_index = f_dom_(face_index, i);
		if ((current_vertex_index != pair0.first) && (current_vertex_index != pair1.first))
		{
			pair2.first = current_vertex_index;
		}
	}

	RDS::ProjectionDescriptor unordered_pairs[3];
	unordered_pairs[0] = pair0;
	unordered_pairs[1] = pair1;
	unordered_pairs[2] = pair2;
	
	RDS::ProjectionDescriptor ordered_pairs[3];
	for (int i = 0; i < 3; i++)
	{
		const int64_t vertex1_index = f_dom_(face_index, i);
		const int64_t vertex2_index = f_dom_(face_index, (i + 1) % 3);
		const int64_t vertex3_index = f_dom_(face_index, (i + 2) % 3);

		if ((vertex1_index == pair0.first) && (vertex2_index == pair1.first))
		{
			ordered_pairs[0] = pair0;
			ordered_pairs[1] = pair1;
			ordered_pairs[2] = pair2;
			break;
		}

		if ((vertex1_index == pair1.first) && (vertex2_index == pair0.first))
		{
			ordered_pairs[0] = pair1;
			ordered_pairs[1] = pair0;
			ordered_pairs[2] = pair2;
			break;
		}
	}
	


	Eigen::Vector3d v0_in = v_dom_.row(ordered_pairs[0].first);
	Eigen::Vector3d v1_in = v_dom_.row(ordered_pairs[1].first);
	Eigen::Vector3d v2_in = v_dom_.row(ordered_pairs[2].first);

	Eigen::Vector2d v0_out = ordered_pairs[0].second;
	Eigen::Vector2d v1_out = ordered_pairs[1].second;
	Eigen::Vector2d v2_out = Eigen::Vector2d::Zero();
	ProjectVertexToPlane(v0_in, v1_in, v2_in, v0_out, v1_out, v2_out);
	ordered_pairs[2].second = v2_out;

	const Eigen::Vector2d vec0_out = v0_out - v1_out;
	const Eigen::Vector2d vec1_out = v2_out - v1_out;

	const Eigen::Vector3d vec0_out_3d = Eigen::Vector3d(vec0_out.x(), vec0_out.y(), 0);
	const Eigen::Vector3d vec1_out_3d = Eigen::Vector3d(vec1_out.x(), vec1_out.y(), 0);

	Eigen::Vector3d cross = vec0_out_3d.cross(vec1_out_3d);

	if (cross.z() < 0)
	{
		output_pairs[0] = ordered_pairs[0];
		output_pairs[1] = ordered_pairs[1];
		output_pairs[2] = ordered_pairs[2];
	}
	else
	{
		output_pairs[0] = ordered_pairs[1];
		output_pairs[1] = ordered_pairs[0];
		output_pairs[2] = ordered_pairs[2];
	}
}

void MeshWrapper::ProjectVertexToPlane(const Eigen::Vector3d& v0_in, const Eigen::Vector3d& v1_in, const Eigen::Vector3d& v2_in, const Eigen::Vector2d& v0_out, const Eigen::Vector2d& v1_out, Eigen::Vector2d& v2_out)
{
	const Eigen::Vector3d vec0_in = v1_in - v0_in;
	const Eigen::Vector3d vec1_in = v2_in - v0_in;

	Eigen::Vector3d rotation_axis_in = vec0_in.cross(vec1_in);
	rotation_axis_in.normalize();

	Eigen::Matrix3d rotation_in;
	rotation_in = Eigen::AngleAxisd(-M_PI / 2, rotation_axis_in);

	const Eigen::Vector3d axis0_in = vec0_in.normalized();
	const Eigen::Vector3d axis1_in = (rotation_in * axis0_in).normalized();

	double axis0_projection = axis0_in.dot(vec1_in);
	double axis1_projection = axis1_in.dot(vec1_in);

	const Eigen::Vector3d roation_axis_out = Eigen::Vector3d(0, 0, 1);
	Eigen::Matrix3d rotation_out;
	rotation_out = Eigen::AngleAxisd(-M_PI / 2, roation_axis_out);
	const Eigen::Vector2d vec0_out = v1_out - v0_out;
	const Eigen::Vector2d axis0_out = vec0_out.normalized();
	const Eigen::Vector3d axis0_3d_out = Eigen::Vector3d(axis0_out.x(), axis0_out.y(), 0);

	const Eigen::Vector3d axis1_3d_out = (rotation_out * axis0_3d_out).normalized();
	const Eigen::Vector2d axis1_out = Eigen::Vector2d(axis1_3d_out.x(), axis1_3d_out.y());

	v2_out = v0_out + axis0_projection * axis0_out + axis1_projection * axis1_out;
}

//void MeshWrapper::CalculateAxes(const Eigen::Vector3d& v0, const Eigen::Vector3d& v1, const Eigen::Vector3d& v2, Eigen::Vector3d& axis0, Eigen::Vector3d& axis1)
//{
//	Eigen::Vector3d vec0 = v1 - v0;
//	Eigen::Vector3d vec1 = v2 - v0;
//
//	Eigen::Vector3d normal_in = vec0.cross(vec1);
//	normal_in.normalize();
//
//	Eigen::Matrix3d axis_rotation;
//	axis_rotation = Eigen::AngleAxisd(-M_PI / 2, normal_in);
//
//	axis0 = vec0.normalized();
//	axis1 = (axis_rotation * axis0).normalized();
//}

void MeshWrapper::ComputeEdges(const Eigen::MatrixX3i& f, Eigen::MatrixX2i& e)
{
	Eigen::SparseMatrix<int> adjacency_matrix;
	igl::adjacency_matrix(f, adjacency_matrix);
	e.resize(adjacency_matrix.nonZeros() >> 1, 2);
	Eigen::MatrixX2i::Index index = 0;
	for (Eigen::SparseMatrix<int>::Index outer_index = 0; outer_index < adjacency_matrix.outerSize(); ++outer_index)
	{
		for (Eigen::SparseMatrix<int>::InnerIterator it(adjacency_matrix, outer_index); it; ++it)
		{
			int row = it.row();
			int column = it.col();
			if (column < row)
			{
				e(index, 0) = row;
				e(index, 1) = column;
				index++;
			}
		}
	}
}

//void MeshWrapper::ComputeSurfaceGradientPerFace(const Eigen::MatrixX3d& v, const Eigen::MatrixX3i& f, Eigen::MatrixX3d& d1, Eigen::MatrixX3d& d2)
//{
//	Eigen::MatrixX3d F1, F2, F3;
//	igl::local_basis(v, f, F1, F2, F3);
//
//	auto f_count = f.rows();
//	auto v_count = v.rows();
//
//	Eigen::MatrixXd Dx(f_count, 3);
//	Eigen::MatrixXd Dy(f_count, 3);
//	Eigen::MatrixXd Dz(f_count, 3);
//	Eigen::MatrixXd fN;
//	Eigen::VectorXd Ar;
//
//	igl::per_face_normals(v, f, fN);
//	igl::doublearea(v, f, Ar);
//
//	Eigen::Vector3i Pi;
//	Pi << 1, 2, 0;
//	Eigen::PermutationMatrix<3> P = Eigen::PermutationMatrix<3>(Pi);
//
//	for (int i = 0; i < f_count; i++)
//	{
//		// renaming indices of vertices of triangles for convenience
//		int i1 = f(i, 0);
//		int i2 = f(i, 1);
//		int i3 = f(i, 2);
//
//		// #F x 3 matrices of triangle edge vectors, named after opposite vertices
//		Eigen::Matrix3d e;
//		e.col(0) = v.row(i2) - v.row(i1);
//		e.col(1) = v.row(i3) - v.row(i2);
//		e.col(2) = v.row(i1) - v.row(i3);;
//
//		Eigen::Vector3d Fni = fN.row(i);
//		double Ari = Ar(i);
//
//		Eigen::Matrix3d n_M;
//		n_M << 0, -Fni(2), Fni(1), Fni(2), 0, -Fni(0), -Fni(1), Fni(0), 0;
//		Eigen::VectorXi R(3); R << 0, 1, 2;
//		Eigen::VectorXi C(3); C << 3 * i + 2, 3 * i, 3 * i + 1;
//		Eigen::Matrix3d res = ((1. / Ari) * (n_M * e)) * P;
//
//		Dx.row(i) = res.row(0);
//		Dy.row(i) = res.row(1);
//		Dz.row(i) = res.row(2);
//	}
//
//	d1 = F1.col(0).asDiagonal() * Dx + F1.col(1).asDiagonal() * Dy + F1.col(2).asDiagonal() * Dz;
//	d2 = F2.col(0).asDiagonal() * Dx + F2.col(1).asDiagonal() * Dy + F2.col(2).asDiagonal() * Dz;
//}
//
void MeshWrapper::NormalizeVertices(Eigen::MatrixX3d& v)
{
	Eigen::RowVector3d barycenter = (v.colwise().minCoeff() + v.colwise().maxCoeff()) / 2.0;
	v = v.rowwise() - barycenter;
	double max_coeff = v.cwiseAbs().maxCoeff();
	v = v / max_coeff;
}

void MeshWrapper::ComputeEdgeDescriptorMap(const Eigen::MatrixX2i& e, ED2EIMap& ed_2_ei)
{
	for (int64_t edge_index = 0; edge_index < e.rows(); ++edge_index)
	{
		ed_2_ei[std::make_pair(e(edge_index, 0), e(edge_index, 1))] = edge_index;
	}
}

//void MeshWrapper::ComputeEdgeIndexMaps()
//{
//	for (int64_t edge_index_im = 0; edge_index_im < e_im_.rows(); ++edge_index_im)
//	{
//		auto v1_index_im = e_im_(edge_index_im, 0);
//		auto v2_index_im = e_im_(edge_index_im, 1);
//		auto v1_index_dom = v_im_2_v_dom_.at(v1_index_im);
//		auto v2_index_dom = v_im_2_v_dom_.at(v2_index_im);
//		auto edge_index_dom = ed_dom_2_ei_dom_.at(std::make_pair(v1_index_dom, v2_index_dom));
//
//		e_im_2_e_dom_[edge_index_im] = edge_index_dom;
//		e_dom_2_e_im_[edge_index_dom].push_back(edge_index_im);
//	}
//}

void MeshWrapper::ComputeVertexIndexMaps()
{
	for (int64_t face_index = 0; face_index < f_dom_.rows(); ++face_index)
	{
		auto current_face_dom = f_dom_.row(face_index);
		auto current_face_im = f_im_.row(face_index);

		v_im_2_v_dom_[current_face_im(0)] = current_face_dom(0);
		v_im_2_v_dom_[current_face_im(1)] = current_face_dom(1);
		v_im_2_v_dom_[current_face_im(2)] = current_face_dom(2);

		v_dom_2_v_im_[current_face_dom(0)].push_back(current_face_im(0));
		v_dom_2_v_im_[current_face_dom(1)].push_back(current_face_im(1));
		v_dom_2_v_im_[current_face_dom(2)].push_back(current_face_im(2));
	}
}

//void MeshWrapper::ComputeVertexToEdgeIndexMaps()
//{
//	for (int64_t edge_index = 0; edge_index < e_im_.rows(); ++edge_index)
//	{
//		auto current_edge_im = e_im_.row(edge_index);
//
//		v_im_2_e_im_[current_edge_im(0)].push_back(edge_index);
//		v_im_2_e_im_[current_edge_im(1)].push_back(edge_index);
//	}
//}

void MeshWrapper::ComputeAdjacencyMaps(
	const Eigen::MatrixX3i& f,
	const ED2EIMap& ed_2_ei,
	VI2FIsMap& vi_2_fi,
	VI2FIsMap& ei_2_fi,
	VI2FIsMap& fi_2_vi,
	VI2FIsMap& fi_2_ei,
	FI2FIsMap& fi_2_fi)
{
	for (int64_t face_index = 0; face_index < f.rows(); ++face_index)
	{
		auto current_face = f.row(face_index);
		for (int i = 0; i < 3; i++)
		{
			int64_t current_vertex_index = current_face(i);
			int64_t next_vertex_index = current_face((i + 1) % 3);

			/**
			 * Vertex to face adjacency
			 */
			if (vi_2_fi.find(current_vertex_index) == vi_2_fi.end())
			{
				vi_2_fi[current_vertex_index] = {};
			}

			vi_2_fi[current_vertex_index].push_back(face_index);

			/**
			 * Edge to face adjacency
			 */
			int64_t edge_index = ed_2_ei.at(std::make_pair(current_vertex_index, next_vertex_index));
			if (ei_2_fi.find(edge_index) == ei_2_fi.end())
			{
				ei_2_fi[edge_index] = {};
			}

			ei_2_fi[edge_index].push_back(face_index);

			/**
			 * Face to vertex adjacency
			 */
			if (fi_2_vi.find(face_index) == fi_2_vi.end())
			{
				fi_2_vi[face_index] = {};
			}

			fi_2_vi[face_index].push_back(current_vertex_index);

			/**
			 * Face to edge adjacency
			 */
			if (fi_2_ei.find(face_index) == fi_2_ei.end())
			{
				fi_2_ei[face_index] = {};
			}

			fi_2_ei[face_index].push_back(edge_index);
		}
	}

	for (int64_t face_index = 0; face_index < f.rows(); ++face_index)
	{
		auto current_face = f.row(face_index);
		fi_2_fi[face_index] = {};
		for (int i = 0; i < 3; i++)
		{
			int64_t current_vertex_index = current_face(i);
			int64_t next_vertex_index = current_face((i + 1) % 3);

			/**
			 * Face to face adjacency
			 */
			int64_t edge_index = ed_2_ei.at(std::make_pair(current_vertex_index, next_vertex_index));
			if (ei_2_fi.find(edge_index) != ei_2_fi.end())
			{
				for (RDS::FaceIndex adjacent_face_index : ei_2_fi[edge_index])
				{
					if (adjacent_face_index != face_index)
					{
						fi_2_fi[face_index].push_back(adjacent_face_index);
					}
				}
			}
		}
	}
}

//void MeshWrapper::ComputeCorrespondingPairs()
//{
//	int64_t current_triplet_index = 0;
//	std::vector<Eigen::Triplet<double>> triplets;
//
//	// Iterate over each edge of in the domain
//	for (int64_t edge_index_dom = 0; edge_index_dom < e_dom_.rows(); ++edge_index_dom)
//	{
//		// Get the corresponding copies in the image
//		std::vector<int64_t> edge_indices_im = e_dom_2_e_im_.at(edge_index_dom);
//
//		// If the domain edge has two image copies
//		if (edge_indices_im.size() == 2)
//		{
//			// Get the indices of the two image edges
//			int64_t edge1_index_im = edge_indices_im[0];
//			int64_t edge2_index_im = edge_indices_im[1];
//
//			// Get the indices of the four image vertices that constitute the two image edges
//			int64_t e1_v1_index_im = e_im_(edge1_index_im, 0);
//			int64_t e1_v2_index_im = e_im_(edge1_index_im, 1);
//			int64_t e2_v1_index_im = e_im_(edge2_index_im, 0);
//			int64_t e2_v2_index_im = e_im_(edge2_index_im, 1);
//
//			// Get the indices of the four corresponding domain vertices
//			int64_t e1_v1_index_dom = v_im_2_v_dom_.at(e1_v1_index_im);
//			int64_t e1_v2_index_dom = v_im_2_v_dom_.at(e1_v2_index_im);
//			int64_t e2_v1_index_dom = v_im_2_v_dom_.at(e2_v1_index_im);
//			int64_t e2_v2_index_dom = v_im_2_v_dom_.at(e2_v2_index_im);
//
//			// Sort the four image vertex-indices according to their corresponding domain vertex-index
//			std::unordered_map<int64_t, std::vector<std::pair<int64_t, int64_t>>> dom_v_index_2_im_v_e_indices;
//			dom_v_index_2_im_v_e_indices[e1_v1_index_dom].push_back(std::make_pair(e1_v1_index_im, edge1_index_im));
//			dom_v_index_2_im_v_e_indices[e1_v2_index_dom].push_back(std::make_pair(e1_v2_index_im, edge1_index_im));
//			dom_v_index_2_im_v_e_indices[e2_v1_index_dom].push_back(std::make_pair(e2_v1_index_im, edge2_index_im));
//			dom_v_index_2_im_v_e_indices[e2_v2_index_dom].push_back(std::make_pair(e2_v2_index_im, edge2_index_im));
//
//			// Get a liat of unique dom vertices indices values
//			std::vector<int64_t> dom_v_indices = { e1_v1_index_dom, e1_v2_index_dom, e2_v1_index_dom, e2_v2_index_dom };
//			const auto it = std::unique(dom_v_indices.begin(), dom_v_indices.end());
//			dom_v_indices.resize(std::distance(dom_v_indices.begin(), it));
//
//			// Sort each bucket of 'dom_v_index_2_im_v_e_indices' according to the image edge index
//			for (auto i = 0; i < dom_v_indices.size(); i++)
//			{
//				std::sort(dom_v_index_2_im_v_e_indices[dom_v_indices[i]].begin(), dom_v_index_2_im_v_e_indices[dom_v_indices[i]].end(), [](std::pair<int64_t, int64_t>& pair1, std::pair<int64_t, int64_t>& pair2) {
//					return pair1.second < pair2.second;
//					});
//			}
//
//			// Record the two corresponding image vertices pairs
//			for (auto& it : dom_v_index_2_im_v_e_indices)
//			{
//				std::pair<int64_t, int64_t> cv_pair = std::minmax(it.second[0].first, it.second[1].first);
//				cv_pairs_.push_back(cv_pair);
//			}
//
//			// Get the vertices indices of the image edges in order. that means that 'e1_v1_index_im_ordered' and 'e2_v1_index_im_ordered' both correspond to the same dom vertex, same for the 'v2' indices.
//			auto e1_v1_index_im_ordered = dom_v_index_2_im_v_e_indices[dom_v_indices[0]][0].first;
//			auto e2_v1_index_im_ordered = dom_v_index_2_im_v_e_indices[dom_v_indices[0]][1].first;
//			auto e1_v2_index_im_ordered = dom_v_index_2_im_v_e_indices[dom_v_indices[1]][0].first;
//			auto e2_v2_index_im_ordered = dom_v_index_2_im_v_e_indices[dom_v_indices[1]][1].first;
//
//			// Record corresponding image edges, expressed using their vertices
//			auto cev_pair = std::make_pair(std::make_pair(e1_v1_index_im_ordered, e1_v2_index_im_ordered), std::make_pair(e2_v1_index_im_ordered, e2_v2_index_im_ordered));
//			edge_pair_descriptors_.push_back(cev_pair);
//
//			// Record corresponding image edges pair 
//			std::pair<int64_t, int64_t> ce_pair = std::minmax(edge1_index_im, edge2_index_im);
//			ce_pairs_.push_back(ce_pair);
//		}
//	}
//}
//
//void MeshWrapper::ComputeCorrespondingVertexPairsCoefficients()
//{
//	std::vector<Eigen::Triplet<double>> triplets;
//
//	for (size_t cv_pair_index = 0; cv_pair_index < cv_pairs_.size(); cv_pair_index++)
//	{
//		auto cv_pair = cv_pairs_[cv_pair_index];
//		triplets.push_back(Eigen::Triplet<double>(cv_pair_index, cv_pair.first, 1));
//		triplets.push_back(Eigen::Triplet<double>(cv_pair_index, cv_pair.second, -1));
//	}
//
//	cv_pairs_coefficients_.resize(cv_pairs_.size(), v_im_.rows());
//	cv_pairs_coefficients_.setFromTriplets(triplets.begin(), triplets.end());
//}
//
//void MeshWrapper::ComputeCorrespondingVertexPairsEdgeLength()
//{
//	auto cv_pairs_size = cv_pairs_.size();
//	cv_pairs_edge_length_.resize(cv_pairs_size, 1);
//	for (size_t cv_pair_index = 0; cv_pair_index < cv_pairs_size; cv_pair_index++)
//	{
//		auto cv_pair = cv_pairs_[cv_pair_index];
//		auto edge_index_im = ed_im_2_ei_im_[std::make_pair(cv_pair.first, cv_pair.second)];
//		auto edge_index_dom = e_im_2_e_dom_[edge_index_im];
//		auto v1_index_dom = e_dom_(edge_index_dom, 0);
//		auto v2_index_dom = e_dom_(edge_index_dom, 1);
//		Eigen::Vector3d v1_dom = v_dom_.row(v1_index_dom);
//		Eigen::Vector3d v2_dom = v_dom_.row(v2_index_dom);
//		auto edge_length = 0.5 * (v1_dom - v2_dom).squaredNorm();
//		cv_pairs_edge_length_(cv_pair_index) = edge_length;
//	}
//}
//
//void MeshWrapper::ComputeVertexNeighbours()
//{
//	for (int64_t i = 0; i < v_im_.rows(); i++)
//	{
//		for (auto& edge_index : v_im_2_e_im_[i])
//		{
//			auto edge = e_im_.row(edge_index);
//			for (int64_t col = 0; col < edge.cols(); col++)
//			{
//				auto vertex_index = edge.coeffRef(col);
//				if (vertex_index != i)
//				{
//					v_im_2_neighbours[i].push_back(vertex_index);
//				}
//			}
//		}
//	}
//}
//
//void MeshWrapper::ComputeFaceFans()
//{
//	for (int64_t i = 0; i < v_dom_.rows(); i++)
//	{
//		RDS::FaceFan face_fan;
//		auto& image_indices = v_dom_2_v_im_.at(i);
//		for (int64_t corner_index = 0; corner_index < image_indices.size(); corner_index++)
//		{
//			auto image_index = image_indices[corner_index];
//			auto& neighbours = v_im_2_neighbours.at(image_index);
//			RDS::FaceFanSlice face_fan_slice = std::make_pair(image_index, std::make_pair(neighbours[0], neighbours[1]));
//			face_fan.push_back(face_fan_slice);
//		}
//
//		face_fans_.push_back(face_fan);
//	}
//}
//
void MeshWrapper::Initialize()
{
	NormalizeVertices(v_dom_);
	ComputeEdges(f_dom_, e_dom_);
	ComputeEdgeDescriptorMap(e_dom_, ed_dom_2_ei_dom_);
	ComputeAdjacencyMaps(f_dom_, ed_dom_2_ei_dom_, vi_dom_2_fi_dom_, ei_dom_2_fi_dom_, fi_dom_2_vi_dom_, fi_dom_2_ei_dom_, fi_dom_2_fi_dom_);

	//////GenerateRandom2DSoup(f_dom_, f_im_, v_im_);
	GenerateIsometric2DSoup(f_dom_, v_dom_, ed_dom_2_ei_dom_, ei_dom_2_fi_dom_, f_im_, v_im_);
	ComputeEdges(f_im_, e_im_);
	ComputeEdgeDescriptorMap(e_im_, ed_im_2_ei_im_);
	ComputeAdjacencyMaps(f_im_, ed_im_2_ei_im_, vi_im_2_fi_im_, ei_im_2_fi_im_, fi_im_2_vi_im_, fi_im_2_ei_im_, fi_im_2_fi_im_);

	ComputeVertexIndexMaps();
	/*ComputeEdgeIndexMaps();
	ComputeVertexToEdgeIndexMaps();
	ComputeCorrespondingPairs();
	ComputeCorrespondingVertexPairsCoefficients();
	ComputeCorrespondingVertexPairsEdgeLength();
	ComputeVertexNeighbours();
	ComputeFaceFans();
	ComputeSurfaceGradientPerFace(v_dom_, f_dom_, d1_, d2_);*/
}