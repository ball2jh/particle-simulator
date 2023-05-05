#ifndef PARTICLE_H
#define PARTICLE_H

#include <cuda_runtime.h>

#include "vector.cuh"

class Particle {
public:
    Particle();
    Particle(const Vector& position, const Vector& velocity, float mass, float radius);

    // Getters and Setters
    __device__ const Vector& getPosition() const;
    __device__ void setPosition(const Vector& position);

    __device__ const Vector& getVelocity() const;
    __device__ void setVelocity(const Vector& velocity);

    __device__ float getMass() const;
    __device__ void setMass(float mass);

    __device__ float getRadius() const;
    __device__ void setRadius(float radius);

    // Other methods
    void updatePosition(float dt);
    __host__ void renderCircle();
    __device__ void wallBounce();

    __device__ bool collidesWith(const Particle& other) const;
    __device__ void resolveCollision(Particle& other);
    //void updateVelocity(const Vector& force, float deltaTime);
private:
    Vector position;
    Vector velocity;
    float mass;
    float radius;
};



#endif // PARTICLE_H