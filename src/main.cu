#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <sstream>

#include "helpers/helper_cuda.h"
#include <GL/glew.h>
#include <GL/freeglut.h>

#include <cuda_gl_interop.h>

#include "renderer/shader.h"

#define MAX_PARTICLES_PER_NODE 4
// vbo variables

GLuint vertex_buffer;
struct cudaGraphicsResource *cuda_vbo_resource;
void *d_vbo_buffer = NULL;

// GL functionality
bool initGL(int *argc, char **argv);
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
               unsigned int vbo_res_flags);

void display(void) {
	glClear(GL_COLOR_BUFFER_BIT);

    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

    // Swap buffers
    glutSwapBuffers();
}

bool initGL(int *argc, char **argv)
{
    glutInit(argc, argv);
    glutInitWindowSize(1024, 768);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutCreateWindow("Particle Simulator");
    glutDisplayFunc(display);

    // Initialize GLEW
    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        fprintf(stderr, "GLEW initialization failed: %s\n", glewGetErrorString(err));
        return false;
    }

    return true;
}

void createVBO(GLuint *vertex_buffer, struct cudaGraphicsResource **vbo_res,
               unsigned int vbo_res_flags) {
    // create buffer object
    glGenBuffers(1, vertex_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, *vertex_buffer);

    float vertices[] = {
        -0.5f, -0.5f,
         0.5f, -0.5f,
         0.5f,  0.5f,
        -0.5f,  0.5f
    };
    // initialize buffer object
    unsigned int size = 8 * sizeof(float);
    glBufferData(GL_ARRAY_BUFFER, size, vertices, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 2, 0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    Shader shader("res/shaders/basic.shader");
    shader.bind();
    
    // ShaderProgramSource source = parseShader("res/shaders/basic.shader");
    // std::cout << "VERTEX" << std::endl;
    // std::cout << source.vertex_source << std::endl;
    // std::cout << "FRAGMENT" << std::endl;
    // std::cout << source.fragment_source << std::endl;

    //GLuint shader = createShader(source.vertex_source, source.fragment_source);

    // register this buffer object with CUDA
    cudaGraphicsGLRegisterBuffer(vbo_res, *vertex_buffer, vbo_res_flags);
}

int main(int argc,  char** argv) {
    const int cuda_device = findCudaDevice(argc, (const char**)argv);
    cudaDeviceProp deviceProps;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProps, cuda_device));

    initGL(&argc, argv);

    createVBO(&vertex_buffer, &cuda_vbo_resource, 0);

    glutMainLoop();

    //glDeleteProgram(shader)

    return 0;
}