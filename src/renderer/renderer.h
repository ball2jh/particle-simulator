#ifndef RENDERER_2D_H
#define RENDERER_2D_H

#pragma once

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>
#include <string>

class Renderer2D {
public:
    static void init();
    void createVBO(struct cudaGraphicsResource **vbo_res, unsigned int vbo_res_flags);
    void display();


private:
    GLuint vertex_buffer;
    GLuint shader_program;

    GLuint compileShader(GLenum shader_type, const std::string& source);
    GLuint createShader(const std::string& vertexShader, const std::string& fragmentShader);
};

#endif