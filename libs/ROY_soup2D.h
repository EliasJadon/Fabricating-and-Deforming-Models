#pragma once
#ifndef OPTIMIZATION_LIB_MESH_WRAPPER_H
#define OPTIMIZATION_LIB_MESH_WRAPPER_H

// STL Includes
#include <vector>
#include <map>
#include <unordered_map>
#include <tuple>
#include <utility>
#include <algorithm>
#include <functional>
#include <string>

// Boost includes
//#include <boost/container_hash/hash.hpp>
//#include <boost/signals2/signal.hpp>
//#include <boost/functional/hash.hpp>

// Eigen Includes
#include <Eigen/Core>
#include <Eigen/Sparse>

// Optimization lib includes
//#include "../core/core.h"
//#include "./mesh_data_provider.h" - No need


namespace RDS
{
	/**
	 * Index types
	 */
	using VertexIndex = std::int64_t;
	using EdgeIndex = std::int64_t;
	using FaceIndex = std::int64_t;
	using EdgeIndices = std::vector<EdgeIndex>;
	using EdgeDescriptor = std::pair<VertexIndex, VertexIndex>;
	using EdgePairDescriptor = std::pair<EdgeDescriptor, EdgeDescriptor>;
	using EdgePairDescriptors = std::vector<EdgePairDescriptor>;
	using SparseVariableIndex = std::int64_t;
	using DenseVariableIndex = std::int64_t;
	using PartialDerivativeIndex = std::int64_t;
	using Face = std::vector<VertexIndex>;
	using Faces = std::vector<Face>;
	using FaceFanSlice = std::pair<VertexIndex, std::pair<VertexIndex, VertexIndex>>;
	using FaceFan = std::vector<FaceFanSlice>;
	using FaceFans = std::vector<FaceFan>;
	using ProjectionDescriptor = std::pair<RDS::VertexIndex, Eigen::Vector2d>;

	/**
	 * Enums
	 */
	enum class CoordinateType : int32_t
	{
		X,
		Y
	};

	/**
	 * Custom hash and equals function objects for unordered_map
	 * https://stackoverflow.com/questions/32685540/why-cant-i-compile-an-unordered-map-with-a-pair-as-key
	 * https://stackoverflow.com/questions/35985960/c-why-is-boosthash-combine-the-best-way-to-combine-hash-values/35991300#35991300
	 */
	struct OrderedPairEquals {
		template <class T1, class T2>
		bool operator () (const std::pair<T1, T2>& pair1, const std::pair<T1, T2>& pair2) const
		{
			const auto pair1_first = static_cast<int64_t>(pair1.first);
			const auto pair1_second = static_cast<int64_t>(pair1.second);
			const auto pair2_first = static_cast<int64_t>(pair2.first);
			const auto pair2_second = static_cast<int64_t>(pair2.second);

			return (pair1_first == pair2_first) && (pair1_second == pair2_second);
		}
	};

	template <class T>
	inline void hash_combine(std::size_t& seed, const T& v)
	{
		std::hash<T> hasher;
		seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
	}

	struct UnorderedPairHash {
		template <class T1, class T2>
		std::size_t operator () (const std::pair<T1, T2>& pair) const
		{
			const auto first = static_cast<int64_t>(pair.first);
			const auto second = static_cast<int64_t>(pair.second);

			const auto minmax_pair = std::minmax(first, second);
			std::size_t seed = 0;

			hash_combine(seed, minmax_pair.first);
			hash_combine(seed, minmax_pair.second);
			return seed;
		}
	};

	struct UnorderedPairEquals {
		template <class T1, class T2>
		bool operator () (const std::pair<T1, T2>& pair1, const std::pair<T1, T2>& pair2) const
		{
			const auto pair1_first = static_cast<int64_t>(pair1.first);
			const auto pair1_second = static_cast<int64_t>(pair1.second);
			const auto pair2_first = static_cast<int64_t>(pair2.first);
			const auto pair2_second = static_cast<int64_t>(pair2.second);

			const auto minmax_pair1 = std::minmax(pair1_first, pair1_second);
			const auto minmax_pair2 = std::minmax(pair2_first, pair2_second);
			return (minmax_pair1.first == minmax_pair2.first) && (minmax_pair1.second == minmax_pair2.second);
		}
	};

	struct VectorEquals {
		template <class T>
		bool operator () (const std::vector<T>& vector1, const std::vector<T>& vector2) const
		{
			if (vector1.size() != vector2.size())
			{
				return false;
			}

			std::vector<T> sorted_vector1 = vector1;
			std::vector<T> sorted_vector2 = vector2;
			std::sort(sorted_vector1.begin(), sorted_vector1.end());
			std::sort(sorted_vector2.begin(), sorted_vector2.end());

			return sorted_vector1 == sorted_vector2;
		}

		bool operator () (const Eigen::VectorXi& vector1, const Eigen::VectorXi& vector2) const
		{
			const auto vector1_internal = std::vector<int64_t>(vector1.data(), vector1.data() + vector1.rows());
			const auto vector2_internal = std::vector<int64_t>(vector2.data(), vector2.data() + vector2.rows());
			return this->operator()(vector1_internal, vector2_internal);
		}
	};

	// https://wjngkoh.wordpress.com/2015/03/04/c-hash-function-for-eigen-matrix-and-vector/
	struct MatrixBaseHash
	{
		template <class Derived>
		std::size_t operator () (const Eigen::MatrixBase<Derived>& matrix_base) const
		{
			return Utils::GenerateHash(matrix_base);
		}
	};

	struct EdgePairDescriptorEquals
	{
		bool operator () (const RDS::EdgePairDescriptor& edge_pair_descriptor1, const RDS::EdgePairDescriptor& edge_pair_descriptor2) const
		{
			const auto edge_descriptor1_1 = static_cast<RDS::EdgeDescriptor>(edge_pair_descriptor1.first);
			const auto edge_descriptor2_1 = static_cast<RDS::EdgeDescriptor>(edge_pair_descriptor1.second);

			const auto edge_descriptor1_2 = static_cast<RDS::EdgeDescriptor>(edge_pair_descriptor2.first);
			const auto edge_descriptor2_2 = static_cast<RDS::EdgeDescriptor>(edge_pair_descriptor2.second);

			RDS::EdgeDescriptor minmax_edge_descriptor1_1 = std::minmax(edge_descriptor1_1.first, edge_descriptor1_1.second);
			RDS::EdgeDescriptor minmax_edge_descriptor2_1 = std::minmax(edge_descriptor2_1.first, edge_descriptor2_1.second);

			RDS::EdgeDescriptor minmax_edge_descriptor1_2 = std::minmax(edge_descriptor1_2.first, edge_descriptor1_2.second);
			RDS::EdgeDescriptor minmax_edge_descriptor2_2 = std::minmax(edge_descriptor2_2.first, edge_descriptor2_2.second);

			if ((minmax_edge_descriptor1_1.first > minmax_edge_descriptor2_1.first) ||
				(minmax_edge_descriptor1_1.first == minmax_edge_descriptor2_1.first) && (minmax_edge_descriptor1_1.second > minmax_edge_descriptor2_1.second))
			{
				std::swap(minmax_edge_descriptor1_1, minmax_edge_descriptor2_1);
			}

			if ((minmax_edge_descriptor1_2.first > minmax_edge_descriptor2_2.first) ||
				(minmax_edge_descriptor1_2.first == minmax_edge_descriptor2_2.first) && (minmax_edge_descriptor1_2.second > minmax_edge_descriptor2_2.second))
			{
				std::swap(minmax_edge_descriptor1_2, minmax_edge_descriptor2_2);
			}

			return (minmax_edge_descriptor1_1 == minmax_edge_descriptor1_2) && (minmax_edge_descriptor2_1 == minmax_edge_descriptor2_2);
		}
	};

	/**
	 * Map types
	 */
	using SparseVariableIndexToDenseVariableIndexMap = std::unordered_map<RDS::SparseVariableIndex, RDS::DenseVariableIndex>;
	using SparseVariableIndexToVertexIndexMap = std::unordered_map<RDS::SparseVariableIndex, RDS::VertexIndex>;
	using DenseVariableIndexToSparseVariableIndexMap = std::unordered_map<RDS::DenseVariableIndex, RDS::SparseVariableIndex>;
}



class MeshWrapper
{
public:
	enum class SoupInitType {
		RANDOM,
		ISOMETRIC
	};

	using ModelLoadedCallback = void();

	using EV2EVMap = std::vector<RDS::EdgePairDescriptor>;
	using VI2VIsMap = std::unordered_map<int64_t, std::vector<int64_t>>;
	using VI2FIsMap = std::unordered_map<int64_t, std::vector<int64_t>>;
	using EI2FIsMap = std::unordered_map<int64_t, std::vector<int64_t>>;
	using FI2VIsMap = std::unordered_map<int64_t, std::vector<int64_t>>;
	using FI2EIsMap = std::unordered_map<int64_t, std::vector<int64_t>>;
	using FI2FIsMap = std::unordered_map<int64_t, std::vector<int64_t>>;

	MeshWrapper(const Eigen::MatrixX3d& v, const Eigen::MatrixX3i& f);
	~MeshWrapper();

	/**
	 * Private type definitions
	 */
	using EdgeDescriptor = std::pair<int64_t, int64_t>;
	using ED2EIMap = std::unordered_map<EdgeDescriptor, int64_t, RDS::UnorderedPairHash, RDS::UnorderedPairEquals>;
	using VI2VIMap = std::unordered_map<int64_t, int64_t>;
	using EI2EIsMap = std::unordered_map<int64_t, std::vector<int64_t>>;
	using EI2EIMap = std::unordered_map<int64_t, int64_t>;
	using VI2EIsMap = std::unordered_map<int64_t, std::vector<int64_t>>;

	/**
	 * Private functions
	 */
	void Initialize();

	/**
	* Private enums
	*/
	enum class ModelFileType
	{
		OBJ,
		OFF,
		UNKNOWN
	};

	/**
	 * General use mesh methods
	 */
	void ComputeEdges(const Eigen::MatrixX3i& f, Eigen::MatrixX2i& e);
	void NormalizeVertices(Eigen::MatrixX3d& v);

	/**
	 * Discrete operators
	 */
	void ComputeSurfaceGradientPerFace(const Eigen::MatrixX3d& v, const Eigen::MatrixX3i& f, Eigen::MatrixX3d& d1, Eigen::MatrixX3d& d2);

	/**
	 * Triangle soup methods
	 */

	 // Soup generation
	void GenerateSoupFaces(const Eigen::MatrixX3i& f_in, Eigen::MatrixX3i& f_out);
	void FixFlippedFaces(const Eigen::MatrixX3i& f_im, Eigen::MatrixX2d& v_im);
	void GenerateRandom2DSoup(const Eigen::MatrixX3i& f_in, Eigen::MatrixX3i& f_out, Eigen::MatrixX2d& v_out);
	void GenerateIsometric2DSoup(const Eigen::MatrixX3i& f_in, const Eigen::MatrixX3d& v_in, const ED2EIMap& ed_2_ei, const EI2FIsMap& ei_dom_2_fi, Eigen::MatrixX3i& f_out, Eigen::MatrixX2d& v_out);
	void CalculateAxes(const Eigen::Vector3d& v0, const Eigen::Vector3d& v1, const Eigen::Vector3d& v2, Eigen::Vector3d& axis0, Eigen::Vector3d& axis1);
	void ProjectVertexToPlane(const Eigen::Vector3d& v0_in, const Eigen::Vector3d& v1_in, const Eigen::Vector3d& v2_in, const Eigen::Vector2d& v0_out, const Eigen::Vector2d& v1_out, Eigen::Vector2d& v2_out);
	void GetOrderedProjectedVertices(const std::vector<RDS::ProjectionDescriptor>& input_pairs, RDS::FaceIndex face_index, std::vector<RDS::ProjectionDescriptor>& output_pairs);

	// Edge descriptor -> edge index map
	void ComputeEdgeDescriptorMap(const Eigen::MatrixX2i& e, ED2EIMap& ed_2_ei);

	// Domain edge index <-> image edge index maps
	void ComputeEdgeIndexMaps();

	// Domain vertex index <-> image vertex index maps
	void ComputeVertexIndexMaps();

	// Vertex index -> edge indices maps
	void ComputeVertexToEdgeIndexMaps();

	// Face/edge/vertex adjacency maps
	void ComputeAdjacencyMaps(
		const Eigen::MatrixX3i& f,
		const ED2EIMap& ed_2_ei,
		VI2FIsMap& vi_2_fi,
		VI2FIsMap& ei_2_fi,
		VI2FIsMap& fi_2_vi,
		VI2FIsMap& fi_2_ei,
		FI2FIsMap& fi_2_fi);

	// Image vertices corresponding pairs and image edges corresponding pairs
	void ComputeCorrespondingPairs();
	void ComputeCorrespondingVertexPairsCoefficients();
	void ComputeCorrespondingVertexPairsEdgeLength();

	// Compute vertex neighbours
	void ComputeVertexNeighbours();

	// Compute adjacent faces vertices
	void ComputeFaceFans();

	/**
	 * Fields
	 */

	 // Domain matrices
	Eigen::MatrixX3d v_dom_;
	Eigen::MatrixX3i f_dom_;
	Eigen::MatrixX2i e_dom_;

	// Image matrices
	Eigen::MatrixX2d v_im_;
	Eigen::MatrixX3i f_im_;
	Eigen::MatrixX2i e_im_;

	// Discrete partial-derivatives matrices
	Eigen::MatrixX3d d1_;
	Eigen::MatrixX3d d2_;

	// Image corresponding pairs
	std::vector<std::pair<int64_t, int64_t>> cv_pairs_;
	std::vector<std::pair<int64_t, int64_t>> ce_pairs_;
	RDS::EdgePairDescriptors edge_pair_descriptors_;
	Eigen::SparseMatrix<double> cv_pairs_coefficients_;
	Eigen::VectorXd cv_pairs_edge_length_;

	// Image neighbour vertices
	VI2VIsMap v_im_2_neighbours;

	// Faces
	RDS::FaceFans face_fans_;

	// Face fans
	RDS::Faces faces_;

	// Maps
	ED2EIMap ed_im_2_ei_im_;
	ED2EIMap ed_dom_2_ei_dom_;
	VI2VIsMap v_dom_2_v_im_;
	VI2VIMap v_im_2_v_dom_;
	EI2EIsMap e_dom_2_e_im_;
	EI2EIMap e_im_2_e_dom_;
	VI2EIsMap v_im_2_e_im_;

	VI2FIsMap vi_im_2_fi_im_;
	EI2FIsMap ei_im_2_fi_im_;
	FI2VIsMap fi_im_2_vi_im_;
	FI2EIsMap fi_im_2_ei_im_;
	FI2FIsMap fi_im_2_fi_im_;

	VI2FIsMap vi_dom_2_fi_dom_;
	EI2FIsMap ei_dom_2_fi_dom_;
	FI2VIsMap fi_dom_2_vi_dom_;
	FI2EIsMap fi_dom_2_ei_dom_;
	FI2FIsMap fi_dom_2_fi_dom_;
};

#endif