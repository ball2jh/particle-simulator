#ifndef VECTOR_H
#define VECTOR_H

class Vector {
public:
    // Default constructor
    Vector();
    Vector(float x, float y);

    // Getters and Setters
    float getX() const;
    void setX(float x);

    float getY() const;
    void setY(float y);

    Vector operator+(const Vector& other) const;
    Vector operator-(const Vector& other) const;
    Vector operator*(float scalar) const;

    float dot(const Vector& other) const;

private:
    float x;
    float y;
};

#endif //VECTOR_H