#include "vertex_buffer.h"
#include <GL/glew.h>

VertexBuffer::VertexBuffer(uint32_t size) {
    glGenBuffers(1, &buffer_id);
    glBindBuffer(GL_ARRAY_BUFFER, buffer_id);
    glBufferData(GL_ARRAY_BUFFER, size, nullptr, GL_DYNAMIC_DRAW);
}

VertexBuffer::~VertexBuffer() {
    glDeleteBuffers(1, &buffer_id);
}

void VertexBuffer::bind() const {
    glBindBuffer(GL_ARRAY_BUFFER, buffer_id);
}

void VertexBuffer::unbind() const {
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void VertexBuffer::set_data(const void* data, uint32_t size) {
    glBindBuffer(GL_ARRAY_BUFFER, buffer_id);
    glBufferSubData(GL_ARRAY_BUFFER, 0, size, data);
}