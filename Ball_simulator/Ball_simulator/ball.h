#ifndef BALL_H
#define BALL_H

#include "vector.h"

class Ball {
public:
    Ball();
    Ball(const Vector& position, const Vector& velocity, float mass, float radius);

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
    void updatePosition(float deltaTime);
    void updateVelocity(const Vector& force, float deltaTime);
private:
    Vector position;
    Vector velocity;
    float mass;
    float radius;
};



#endif // BALL_H