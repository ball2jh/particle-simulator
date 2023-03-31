#include "vector.h"

Vector::Vector() : x(0), y(0) {}

Vector::Vector(float x, float y) : x(x), y(y) {}

float Vector::getX() const {
	return x;
}

float Vector::getY() const {
	return y;
}

void Vector::setX(float new_x) {
	x = new_x;
}

void Vector::setX(float new_y) {
	y = new_y;
}

Vector Vector::operator+(const Vector& other) const {
	return Vector(x + other.x, y + other.y);
}

Vector Vector::operator-(const Vector& other) const {
	return Vector(x - other.x, y - other.y);
}

Vector Vector::operator*(float scalar) const {
	return Vector(x * scalar, y * scalar);
}

float Vector::dot(const Vector& other) const {
	return x * other.x + y * other.y;
}

