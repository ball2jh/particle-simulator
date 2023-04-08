#ifndef RENDERER_H
#define RENDERER_H

#pragma once

#include <GL/glew.h>
#include <GL/freeglut.h>

#include "shader.h"
#include "vertex_array.h"
#include "vertex_buffer.h"
#include "vertex_buffer_layout.h"
#include "index_buffer.h"

class Renderer {
public:
    void draw(const VertexArray& vertex_array, const IndexBuffer& index_buffer, const Shader& shader) const;
};
#endif // RENDERER_H