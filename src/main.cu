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
#include "renderer/vertex_buffer.h"
#include "renderer/index_buffer.h"
#include "renderer/vertex_array.h"
#include "renderer/vertex_buffer_layout.h"
#include "renderer/renderer.h"

#define MAX_PARTICLES_PER_NODE 4
// vbo variables
Renderer renderer;

VertexArray* vertex_array;
IndexBuffer* index_buffer;
Shader* shader;
struct cudaGraphicsResource *cuda_vbo_resource;
void *d_vbo_buffer = NULL;

// GL functionality
bool initGL(int *argc, char **argv);

void display(void) {
	glClear(GL_COLOR_BUFFER_BIT);

    renderer.draw(*vertex_array, *index_buffer, *shader);

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

    // Set openGl version
    glutInitContextVersion(4, 6);
    glutInitContextProfile(GLUT_CORE_PROFILE);

    // Initialize GLEW
    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        fprintf(stderr, "GLEW initialization failed: %s\n", glewGetErrorString(err));
        return false;
    }

    return true;
}

int main(int argc, char** argv) {
    const int cuda_device = findCudaDevice(argc, (const char**)argv);
    cudaDeviceProp deviceProps;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProps, cuda_device));

    initGL(&argc, argv);

    shader = new Shader("res/shaders/basic.shader");
    vertex_array = new VertexArray();
    VertexBuffer vertex_buffer = VertexBuffer(8 * sizeof(float));

    float vertices[] = {
        -0.5f, -0.5f,
         0.5f, -0.5f,
         0.5f,  0.5f,
        -0.5f,  0.5f
    };

    vertex_buffer.set_data(vertices, 8 * sizeof(float));

    VertexBufferLayout layout;
    layout.push<float>(2);
    vertex_array->add_buffer(vertex_buffer, layout);

    uint32_t indeces[] = {
        0, 1, 2,
        2, 3, 1
    };

    index_buffer = new IndexBuffer(indeces, 6);

    Renderer renderer;

    return 0;
}