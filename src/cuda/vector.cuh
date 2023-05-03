#ifndef VECTOR_H
#define VECTOR_H

#include <cuda_runtime.h>

class Vector {
public:
    // Default constructor
    __host__ __device__ Vector();
    __host__ __device__ Vector(float x, float y);

    __host__ __device__ float getX() const;
    __host__ __device__ void setX(float x);
    __host__ __device__ float getY() const;
    __host__ __device__ void setY(float y);
    __host__ __device__ float dot(const Vector& other) const;

    __host__ __device__ Vector operator+(const Vector& other) const;
    __host__ __device__ Vector operator+=(const Vector& other);
    __host__ __device__ Vector operator-(const Vector& other) const;
    __host__ __device__ Vector operator-=(const Vector& other);
    __host__ __device__ Vector operator*(float scalar) const;
    __host__ __device__ Vector& operator*=(float scalar);
    __host__ __device__ Vector operator/(float scalar) const;
    __host__ __device__ Vector& operator/=(float scalar);
private:
    float x;
    float y;
};

#endif //VECTOR_H