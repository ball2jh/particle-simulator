#ifndef VECTOR_H
#define VECTOR_H

#include <cuda_runtime.h>

class Vector {
public:
    // Default constructor
    Vector();
    Vector(float x, float y);

    float getX() const;
    void setX(float x);
    float getY() const;
    void setY(float y);
    float dot(const Vector& other) const;

    Vector operator+(const Vector& other) const;
    Vector operator+=(const Vector& other);
    Vector operator-(const Vector& other) const;
    Vector operator-=(const Vector& other);
    Vector operator*(float scalar) const;
    Vector& operator*=(float scalar);
    Vector operator/(float scalar) const;
    Vector& operator/=(float scalar);
private:
    float x;
    float y;
};

#endif //VECTOR_H