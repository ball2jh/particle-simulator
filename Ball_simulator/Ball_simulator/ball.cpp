#include "ball.h"

Ball::Ball() : position(Vector(0, 0)), velocity(Vector(0, 0)), mass(1), radius(1) {}
Ball::Ball(const Vector& position, const Vector& velocity, float mass, float radius) : position(position), velocity(velocity), mass(mass), radius(radius) {}

const Vector& Ball::getPosition() const {
    return position;
}

void Ball::setPosition(const Vector& position) {
    this->position = position;
}

const Vector& Ball::getVelocity() const {
    return velocity;
}

void Ball::setVelocity(const Vector& new_velocity) {
    this->velocity = new_velocity;
}

float Ball::getMass() const {
    return mass;
}

void Ball::setMass(float mass) {
    this->mass = mass;
}

float Ball::getRadius() const {
    return radius;
}

void Ball::setRadius(float radius) {
    this->radius = radius;
}

void Ball::updatePosition(float deltaTime) {
    position += velocity * deltaTime;
}

void Ball::updateVelocity(const Vector& force, float deltaTime) {
    Vector acceleration = force / mass;
    velocity += acceleration * deltaTime;
}