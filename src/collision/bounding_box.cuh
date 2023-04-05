#ifndef BOUNDING_BOX_H
#define BOUNDING_BOX_H

#include <cuda_runtime.h>

class BoundingBox {
 public:
  // Constructor. Create a unit box.
  __host__ __device__ BoundingBox();

  // Compute the center of the bounding-box.
  __host__ __device__ void compute_center(float2 &center) const;

  // The points of the box.
  __host__ __device__ __forceinline__ const float2 &get_max() const;

  __host__ __device__ __forceinline__ const float2 &get_min() const;

  // Does a box contain a point.
  __host__ __device__ bool contains(const float2 &p) const;

  // Define the bounding box.
  __host__ __device__ void set(float min_x, float min_y, float max_x, float max_y);
 private:
    // Extreme points of the bounding box.
  float2 min_point;
  float2 max_point;
};

#endif  // BOUNDING_BOX_H
