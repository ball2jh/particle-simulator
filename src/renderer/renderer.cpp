#include "renderer.h"

void Renderer::draw(const VertexArray& vertex_array, const IndexBuffer& index_buffer, const Shader& shader) const {
    shader.bind();
    vertex_array.bind();
    index_buffer.bind();

    glDrawElements(GL_TRIANGLES, index_buffer.get_count(), GL_UNSIGNED_INT, nullptr);
}