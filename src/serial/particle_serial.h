#ifndef PARTICLE_H
#define PARTICLE_H

// #include <cuda_runtime.h>

#include "vector_serial.h"

class Particle {
    public:
        // Constructors
        Particle();
        Particle(Vector position, Vector velocity, float mass, float radius);

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
        void updatePosition(float dt);
        void renderCircle();
        void wallBounce();

        bool collidesWith(const Particle& other) const;
        void resolveCollision(Particle& other);
        // void updateVelocity(const Vector& force, float deltaTime);
    private:
        Vector position;
        Vector velocity;
        float mass;
        float radius;
};



#endif // PARTICLE_H