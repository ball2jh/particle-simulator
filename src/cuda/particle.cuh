#ifndef PARTICLE_H
#define PARTICLE_H

#include <cuda_runtime.h>

#include "vector.cuh"

class Particle {
public:
    Particle();
    Particle(const Vector& position, const Vector& velocity, float mass, float radius);

    // Getters and Setters
    __host__ __device__ const Vector& getPosition() const;
    __host__ __device__ void setPosition(const Vector& position);

    __host__ __device__ const Vector& getVelocity() const;
    __host__ __device__ void setVelocity(const Vector& velocity);

    __host__ __device__ float getMass() const;
    __host__ __device__ void setMass(float mass);

    __host__ __device__ float getRadius() const;
    __host__ __device__ void setRadius(float radius);

    // Other methods
    __host__ __device__ void updatePosition(float dt);
    __host__ void renderCircle();
    __host__ __device__ void wallBounce();
    //void updateVelocity(const Vector& force, float deltaTime);
private:
    Vector position;
    Vector velocity;
    float mass;
    float radius;
};



#endif // PARTICLE_H