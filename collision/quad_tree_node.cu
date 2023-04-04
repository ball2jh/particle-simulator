#include "quad_tree_node.cuh"

// Constructor.
__host__ __device__ QuadTreeNode::QuadTreeNode() : id(0), begin(0), end(0) {}

// The ID of a node at its level.
__host__ __device__ int QuadTreeNode::get_id() const {
  return id;
}

// Set the ID of a node at its level.
__host__ __device__ void QuadTreeNode::set_id(int new_id) {
  id = new_id;
}

// Get the bounding box.
__host__ __device__ __forceinline__ const BoundingBox &QuadTreeNode::get_bounding_box() const {
  return bounding_box;
}

// Set the bounding box.
__host__ __device__ __forceinline__ void QuadTreeNode::set_bounding_box(float min_x,
                                                                         float min_y,
                                                                         float max_x,
                                                                         float max_y) {
  bounding_box.set(min_x, min_y, max_x, max_y);
}

// The number of circles in the tree.
__host__ __device__ __forceinline__ int QuadTreeNode::num_points() const {
  return end - begin;
}

// The range of circles in the tree.
__host__ __device__ __forceinline__ int QuadTreeNode::points_begin() const {
  return begin;
}

__host__ __device__ __forceinline__ int QuadTreeNode::points_end() const {
  return end;
}

// Define the range for that node.
__host__ __device__ __forceinline__ void QuadTreeNode::set_range(int begin, int end) {
  this->begin = begin;
  this->end = end;
}
