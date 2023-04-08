#include "vertex_array.h"

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

void VertexArray::add_buffer(const VertexBuffer& vertex_buffer, const VertexBufferLayout& layout)
{
    bind();
    vertex_buffer.bind();
    const auto& elements = layout.get_elements();
    uint32_t offset = 0;
    for (uint32_t i = 0; i < elements.size(); i++) {
        const auto& element = elements[i];
        glEnableVertexAttribArray(i);
        glVertexAttribPointer(i, element.size, element.type, element.normalized, layout.get_stride(), (const void*)offset);
        offset += element.size * VertexBufferLayout::Element::get_size_of_type(element.type);
    }
}
