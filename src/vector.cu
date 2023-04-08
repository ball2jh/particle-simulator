#include "vector.cuh"

// Default constructor
__host__ __device__ Vector::Vector() : x(0.0f), y(0.0f) {}

__host__ __device__ Vector::Vector(float x, float y) : x(x), y(y) {}

__host__ __device__ float Vector::getX() const {
    return x;
}

__host__ __device__ void Vector::setX(float x) {
    this->x = x;
}

__host__ __device__ float Vector::getY() const {
    return y;
}

__host__ __device__ void Vector::setY(float y) {
    this->y = y;
}

__host__ __device__ Vector Vector::operator+(const Vector& other) const {
    return Vector(x + other.x, y + other.y);
}

__host__ __device__ Vector Vector::operator+=(const Vector& other) {
    x += other.x;
    y += other.y;
    return *this;
}

__host__ __device__ Vector Vector::operator-(const Vector& other) const {
    return Vector(x - other.x, y - other.y);
}

__host__ __device__ Vector Vector::operator-=(const Vector& other) {
    x -= other.x;
    y -= other.y;
    return *this;
}

__host__ __device__ Vector Vector::operator*(float scalar) const {
    return Vector(x * scalar, y * scalar);
}

__host__ __device__ Vector& Vector::operator*=(float scalar) {
    x *= scalar;
    y *= scalar;
    return *this;
}

__host__ __device__ Vector Vector::operator/(float scalar) const {
    return Vector(x / scalar, y / scalar);
}

__host__ __device__ Vector& Vector::operator/=(float scalar) {
    x /= scalar;
    y /= scalar;
    return *this;
}
