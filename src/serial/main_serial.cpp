// #include <device_launch_parameters.h>
// #include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <cstdlib>



// #include "helpers/helper_cuda.h"
#include <GL/glew.h>
#include <GL/freeglut.h>

// #include <cuda_gl_interop.h>

#include "particle_serial.h"
#include "particle_serial.cpp"

#include "vector_serial.h"
#include "vector_serial.cpp"

#define MAX_PARTICLES_PER_NODE 4
// vbo variables
#include <math.h>
#define PI 3.14159265f
#define PARTICLE_NUM 5

GLuint vertex_buffer;
struct cudaGraphicsResource *cuda_vbo_resource;
void *d_vbo_buffer = NULL;
Particle particles[2];

// GL functionality
bool initGL(int *argc, char **argv);
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
               unsigned int vbo_res_flags);

void display() {
	glClear(GL_COLOR_BUFFER_BIT);

    for (int i = 0; i < PARTICLE_NUM; i++) {
        particles[i].renderCircle();
        // make a random number
        float dx = (float) rand();
        // scale it to be between 2 and 4
        float scaled = (dx / RAND_MAX) * 2 + 2;
        particles[i].updatePosition(scaled);
        particles[i].wallBounce();

        // check for collisions with other particles (NOT IMPLEMENTED ATM)
        for (int j = i + 1; j < PARTICLE_NUM; j++) {
            if (particles[i].collidesWith(particles[j])) {
                particles[i].resolveCollision(particles[j]);
            }
        }
    }
    // glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    // p->updatePos(2);
    // p->renderCircle();
    // p->wallBounce();
    // Swap buffers
    glutSwapBuffers();
}

void timer( int value )
{
    glutPostRedisplay();
    glutTimerFunc( 16, timer, 0 );
}

bool initGL(int *argc, char **argv)
{
    glutInit(argc, argv);
    glutInitWindowSize(800, 800);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutCreateWindow("Particle Simulator");
    glutTimerFunc( 0, timer, 0 );
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

int main(int argc, char** argv) {
    // const int cuda_device = findCudaDevice(argc, (const char**)argv);
    // cudaDeviceProp deviceProps;
    // checkCudaErrors(cudaGetDeviceProperties(&deviceProps, cuda_device));
    
    srand(time(NULL));
    for (int i = 0; i < PARTICLE_NUM; i++) {
        std::random_device rd;
        std::mt19937 gen(rd());

        std::uniform_real_distribution<float> dist(-0.0015, 0.0015);
        std::uniform_real_distribution<float> rand(-0.95, 0.95);

        // make random particle velocity        
        float dx = dist(gen) * 6;
        float dy = dist(gen) * 6;

        // make random particle position
        float x = rand(gen);
        float y = rand(gen);

        // printf("dx: %f, dy: %f\n", dx, dy);
        // printf ("x: %f, y: %f\n", x, y);

        particles[i] = Particle(Vector(x, y), Vector(dx, dy), 1, 0.1);
    }
    initGL(&argc, argv);
    //createVBO(&vertex_buffer, &cuda_vbo_resource, 0);

    // VertexBuffer buffer = VertexBuffer(8 * sizeof(float));

    // float vertices[] = {
    //     -0.5f, -0.5f,
    //      0.5f, -0.5f,
    //      0.5f,  0.5f,
    //     -0.5f,  0.5f
    // };
    // buffer.set_data(vertices, 8 * sizeof(float));

    // glEnableVertexAttribArray(0);
    // glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 2, 0);


    // Shader shader = Shader("res/shaders/basic.shader");
    // shader.bind();

    glutMainLoop();

    // //glDeleteProgram(shader)

    // return 0;
}