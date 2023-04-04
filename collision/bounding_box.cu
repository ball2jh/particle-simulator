#include <cuda_runtime.h>

#include "bounding_box.cuh"

// Constructor. Create a unit box.
__host__ __device__ BoundingBox::BoundingBox() {
  min_point = make_float2(0.0f, 0.0f);
  max_point = make_float2(1.0f, 1.0f);
}

// Compute the center of the bounding-box.
__host__ __device__ void BoundingBox::compute_center(float2 &center) const {
  center.x = 0.5f * (min_point.x + max_point.x);
  center.y = 0.5f * (min_point.y + max_point.y);
}

// The points of the box.
__host__ __device__ __forceinline__ const float2 &BoundingBox::get_max() const {
  return max_point;
}

__host__ __device__ __forceinline__ const float2 &BoundingBox::get_min() const {
  return min_point;
}

// Does a box contain a point.
__host__ __device__ bool BoundingBox::contains(const float2 &p) const {
  return p.x >= min_point.x && p.x < max_point.x && p.y >= min_point.y && p.y < max_point.y;
}

// Define the bounding box.
__host__ __device__ void BoundingBox::set(float min_x, float min_y, float max_x, float max_y) {
  min_point.x = min_x;
  min_point.y = min_y;
  max_point.x = max_x;
  max_point.y = max_y;
}