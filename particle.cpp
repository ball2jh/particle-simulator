#include <cuda_runtime.h>

#include "particle.h"

Particle::Particle() : position(Vector(0, 0)), velocity(Vector(0, 0)), mass(1), radius(1) {}
Particle::Particle(const Vector& position, const Vector& velocity, float mass, float radius ) : position(position), velocity(velocity), mass(mass), radius(radius) {}

const Vector& Particle::getPosition() const {
    return position;
}

void Particle::setPosition(const Vector& position) {
    this->position = position;
    }

const Vector& Particle::getVelocity() const {
    return velocity;
}

void Particle::setVelocity(const Vector& new_velocity) {
    this->velocity = new_velocity;
}

float Particle::getMass() const {
    return mass;
}

void Particle::setMass(float mass) {
    this->mass = mass;
}

float Particle::getRadius() const {
    return radius;
}

void Particle::setRadius(float radius) {
    this->radius = radius;
}

__host__ __device__ void Particle::updatePosition(float deltaTime) {
    this->position += this->velocity * deltaTime;
}

void Particle::updateVelocity(const Vector& force, float deltaTime) {
    Vector acceleration = force / mass;
    velocity += acceleration * deltaTime;
}