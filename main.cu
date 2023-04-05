#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <sys/time.h>
#include <iostream>

#include "helpers/helper_cuda.h"
#include "helpers/helper_gl.h"
#include <GL/freeglut.h>

#include <cuda_gl_interop.h>

#define MAX_PARTICLES_PER_NODE 4
// vbo variables

GLuint vertex_buffer;
struct cudaGraphicsResource *cuda_vbo_resource;
void *d_vbo_buffer = NULL;

// GL functionality
bool initGL(int *argc, char **argv);
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
               unsigned int vbo_res_flags);
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res);

void display(void) {
	glClear(GL_COLOR_BUFFER_BIT);

    glDrawArrays(GL_POINTS, 0, 1);

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

    SDK_CHECK_ERROR_GL();

    return true;
}

void createVBO(GLuint *vertex_buffer, struct cudaGraphicsResource **vbo_res,
               unsigned int vbo_res_flags) {
    // create buffer object
    glGenBuffers(1, vertex_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, *vertex_buffer);

    float vertices[2] = {0.0f, 0.0f};
    // initialize buffer object
    unsigned int size = 2 * sizeof(float);
    glBufferData(GL_ARRAY_BUFFER, size, vertices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 2, 0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    SDK_CHECK_ERROR_GL();

    // register this buffer object with CUDA
    cudaGraphicsGLRegisterBuffer(vbo_res, *vertex_buffer, vbo_res_flags);

    SDK_CHECK_ERROR_GL();
}

int main(int argc,  char** argv) {
    int cuda_device = findCudaDevice(argc, (const char **)argv);
    cudaDeviceProp deviceProps;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProps, cuda_device));

    initGL(&argc, argv);

    glutMainLoop();

    return 0;
}