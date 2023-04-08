#ifndef VERTEX_ARRAY_H
#define VERTEX_ARRAY_H

#include "GL/glew.h"

class VertexArray
{
public:
    VertexArray();
    ~VertexArray();

    void bind() const;
    void unbind() const;
private:
    GLuint array_id;
};


#endif // VERTEX_ARRAY_H