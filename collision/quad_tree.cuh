#ifndef QUADTREE_H_
#define QUADTREE_H_

#include <thrust/device_vector.h>

class QuadTree {
public:
    QuadTree();
private:
    int num_particles;
    int warp_size;
    int max_depth;
    int min_particles_per_node;
}

#endif  // QUADTREE_H_