#ifndef VERTEX_ARRAY_H
#define VERTEX_ARRAY_H

#include "GL/glew.h"
#include "vertex_buffer_layout.h"
#include "vertex_buffer.h"

class VertexArray
{
public:
    VertexArray();
    ~VertexArray();

    void bind() const;
    void unbind() const;

    void add_buffer(const VertexBuffer& vertex_buffer, const VertexBufferLayout& layout);
private:
    GLuint array_id;
};


#endif // VERTEX_ARRAY_H