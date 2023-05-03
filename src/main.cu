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

#include "vector.cuh"

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
	renderer.clear();
    renderer.draw(*vertex_array, *index_buffer, *shader);

    glutSwapBuffers();
}

bool initGL(int *argc, char **argv)
{
    glutInit(argc, argv);
    glutInitWindowSize(800, 800);
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
    initGL(&argc, argv);

    shader = new Shader("res/shaders/circle.shader");
    vertex_array = new VertexArray();
    VertexBuffer vertex_buffer = VertexBuffer(12 * sizeof(float));

    float quad_vertices[] = {
        -1.0f, -1.0f, 1000.0f,
        1.0f, -1.0f, 1000.0f,
        1.0f,  1.0f, 1000.0f,
        -1.0f,  1.0f, 1000.0f,
    };

    vertex_buffer.set_data(quad_vertices, 12 * sizeof(float));

    VertexBufferLayout layout;

    layout.push<float>(2);
    layout.push<float>(1);
    vertex_array->add_buffer(vertex_buffer, layout);

    uint32_t indices[] = {
        0, 1, 2,
        2, 3, 0,
    };

    index_buffer = new IndexBuffer(indices, 6);

    glutMainLoop();

    return 0;
}