#include "vertex_buffer_layout.h"

template<>
void VertexBufferLayout::push<float>(unsigned int count) {
    elements_.push_back({ count, GL_FLOAT, GL_FALSE });
    stride_ += count * sizeof(float);
}

template<>
void VertexBufferLayout::push<unsigned int>(unsigned int count) {
    elements_.push_back({ count, GL_UNSIGNED_INT, GL_FALSE });
    stride_ += count * sizeof(unsigned int);
}