#include <cuda_runtime.h>
#include <GL/gl.h>     // The GL Header File
#include <GL/glut.h>   // The GL Utility Toolkit (Glut) Header
#include <math.h>

#include "particle.cuh"

#define PI 3.14159265f

Particle::Particle() : position(Vector(0, 0)), velocity(Vector(0, 0)), mass(1), radius(1) {}
Particle::Particle(const Vector& position, const Vector& velocity, float mass, float radius ) : position(position), velocity(velocity), mass(mass), radius(radius) {}

__host__ __device__ const Vector& Particle::getPosition() const {
    return position;
}

__host__ __device__ void Particle::setPosition(const Vector& position) {
    this->position = position;
    }

__host__ __device__ const Vector& Particle::getVelocity() const {
    return velocity;
}

__host__ __device__ void Particle::setVelocity(const Vector& new_velocity) {
    this->velocity = new_velocity;
}

__host__ __device__ float Particle::getMass() const {
    return mass;
}

__host__ __device__ void Particle::setMass(float mass) {
    this->mass = mass;
}

__host__ __device__ float Particle::getRadius() const {
    return radius;
}

__host__ __device__ void Particle::setRadius(float radius) {
    this->radius = radius;
}

__host__ __device__ void Particle::updatePosition(float deltaTime) {
    this->position += this->velocity * deltaTime;
}

__host__ void Particle::renderCircle() {
    GLfloat ballRadius = (GLfloat) this->radius;   // Radius of the bouncing ball
    GLfloat ballX = (GLfloat) this->position.getX();
    GLfloat ballY = (GLfloat) this->position.getY();

    glMatrixMode(GL_MODELVIEW);    // To operate on the model-view matrix
    glLoadIdentity();              // Reset model-view matrix

    glTranslatef(ballX, ballY, 0.0f);  // Translate to (xPos, yPos)
    // Use triangular segments to form a circle
    glBegin(GL_TRIANGLE_FAN);
        glColor3f(1, 0, 1);
        glVertex2f(0.0f, 0.0f);       // Center of circle
        int numSegments = 100;
        GLfloat angle;
        for (int i = 0; i <= numSegments; i++) { // Last vertex same as first vertex
            angle = (i * 2.0f * PI) / numSegments;  // 360 deg for all segments
            glVertex2f(cos(angle) * ballRadius, sin(angle) * ballRadius);
        }
    glEnd();
}

__host__ __device__ void Particle::wallBounce() {
    float x = this->position.getX();
    float y = this->position.getY();
    float dx = this->velocity.getX();
    float dy = this->velocity.getY();
    float radius = this->getRadius();
    
    if (x + radius > 1) {
        this->position.setX(1 - radius);
        this->velocity.setX(-dx);
    } else if (x - radius < -1) {
        this->position.setX(-1 + radius);
        this->velocity.setX(-dx);
    }

    if (y + radius > 1) {
        this->position.setY(1 - radius);
        this->velocity.setY(-dy);
    } else if (y - radius < -1) {
        this->position.setY(-1 + radius);
        this->velocity.setY(-dy);
    }
}
