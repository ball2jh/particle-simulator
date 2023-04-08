#include "vertex_array.h"
#include "buffer.h"

VertexArray::VertexArray()
{
    glGenVertexArrays(1, &array_id);
}

VertexArray::~VertexArray()
{
    glDeleteVertexArrays(1, &array_id);
}

void VertexArray::bind() const
{
    glBindVertexArray(array_id);
}

void VertexArray::unbind() const
{
    glBindVertexArray(0);
}
