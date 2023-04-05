#ifndef PARTICLE_H
#define PARTICLE_H

#include <cuda_runtime.h>

#include "vector.h"

class Particle {
public:
    Particle();
    Particle(const Vector& position, const Vector& velocity, float mass, float radius);

    // Getters and Setters
    const Vector& getPosition() const;
    void setPosition(const Vector& position);

    const Vector& getVelocity() const;
    void setVelocity(const Vector& velocity);

    float getMass() const;
    void setMass(float mass);

    float getRadius() const;
    void setRadius(float radius);

    // Other methods
    __host__ __device__ void updatePosition(float deltaTime);
    void updateVelocity(const Vector& force, float deltaTime);
private:
    Vector position;
    Vector velocity;
    float mass;
    float radius;
};



#endif // PARTICLE_H