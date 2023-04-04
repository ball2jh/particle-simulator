#ifndef QUADTREE_NODE_H
#define QUADTREE_NODE_H

#include "bounding_box.cuh"

class QuadTreeNode {
 public:
  // Constructor.
  __host__ __device__ QuadTreeNode();

  // The ID of a node at its level.
  __host__ __device__ int get_id() const;

  // Set the ID of a node at its level.
  __host__ __device__ void set_id(int new_id);

  // The bounding box.
  __host__ __device__ __forceinline__ const BoundingBox &get_bounding_box() const;

  // Set the bounding box.
  __host__ __device__ __forceinline__ void set_bounding_box(float min_x, float min_y,
                                                            float max_x, float max_y);

  // The number of points in the tree.
  __host__ __device__ __forceinline__ int num_points() const;

  // The range of points in the tree.
  __host__ __device__ __forceinline__ int points_begin() const;
  __host__ __device__ __forceinline__ int points_end() const;

  // Define the range for that node.
  __host__ __device__ __forceinline__ void set_range(int begin, int end);
 private:
  // The identifier of the node.
  int id;
  // The bounding box of the tree.
  BoundingBox bounding_box;
  // The range of points.
  int begin, end;
};

#endif  // QUADTREE_NODE_H
