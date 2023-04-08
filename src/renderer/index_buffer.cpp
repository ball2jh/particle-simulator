#include "index_buffer.h"
#include <GL/glew.h>

IndexBuffer::IndexBuffer(uint32_t* indices, uint32_t count) : count(count) {
    glGenBuffers(1, &buffer_id);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer_id);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, count * sizeof(uint32_t), indices, GL_STATIC_DRAW);
}

IndexBuffer::~IndexBuffer() {
    glDeleteBuffers(1, &buffer_id);
}

void IndexBuffer::bind() const {
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer_id);
}

void IndexBuffer::unbind() const {
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

uint32_t IndexBuffer::get_count() const {
    return count;
}
