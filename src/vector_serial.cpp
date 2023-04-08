#include "vector_serial.h"

// Default constructor
Vector::Vector() : x(0.0f), y(0.0f) {}

Vector::Vector(float x, float y) : x(x), y(y) {}

float Vector::getX() const {
    return x;
}

void Vector::setX(float x) {
    this->x = x;
}

float Vector::getY() const {
    return y;
}

void Vector::setY(float y) {
    this->y = y;
}

float Vector::dot(const Vector& other) const {
    return x * other.x + y * other.y;
}

Vector Vector::operator+(const Vector& other) const {
    return Vector(x + other.x, y + other.y);
}

Vector Vector::operator+=(const Vector& other) {
    x += other.x;
    y += other.y;
    return *this;
}

Vector Vector::operator-(const Vector& other) const {
    return Vector(x - other.x, y - other.y);
}

Vector Vector::operator-=(const Vector& other) {
    x -= other.x;
    y -= other.y;
    return *this;
}

Vector Vector::operator*(float scalar) const {
    return Vector(x * scalar, y * scalar);
}

Vector& Vector::operator*=(float scalar) {
    x *= scalar;
    y *= scalar;
    return *this;
}

Vector Vector::operator/(float scalar) const {
    return Vector(x / scalar, y / scalar);
}

Vector& Vector::operator/=(float scalar) {
    x /= scalar;
    y /= scalar;
    return *this;
}

