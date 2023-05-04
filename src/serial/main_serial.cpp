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
#include <unistd.h>


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

int num_particles;
float particle_size;
Particle* particles;

GLuint vertex_buffer;
struct cudaGraphicsResource *cuda_vbo_resource;
void *d_vbo_buffer = NULL;

// GL functionality
bool initGL(int *argc, char **argv);
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
               unsigned int vbo_res_flags);

void display() {
	glClear(GL_COLOR_BUFFER_BIT);

    for (int i = 0; i < num_particles; i++) {
        particles[i].renderCircle();
        // make a random number
        float dx = (float) rand();
        // scale it to be between 2 and 4
        float scaled = (dx / RAND_MAX) * 2 + 2;
        particles[i].updatePosition(scaled);
        particles[i].wallBounce();

        // check for collisions with other particles (NOT IMPLEMENTED ATM)
        for (int j = i + 1; j < num_particles; j++) {
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

    static int frameCount = 0;
    static int lastTime = 0;
    int currentTime = glutGet(GLUT_ELAPSED_TIME);
    frameCount++;

    if (currentTime - lastTime > 1000) {
        char title[80];
        sprintf(title, "Particle Simulator - Serial (%d fps)", frameCount);
        printf("%d\n", frameCount);
        glutSetWindowTitle(title);
        frameCount = 0;
        lastTime = currentTime;
    }

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

    // Set defaults
    num_particles = 5;
    particle_size = 0.1f;
    int opt;

    // Command line options
    while ((opt = getopt(argc, argv, "n:s:")) != -1) {
        switch (opt) {
            case 'n':
                num_particles = strtol(optarg, NULL, 10);
                break;
            case 's':
                particle_size = strtod(optarg, NULL);
                break;
            default:
                fprintf(stderr, "Usage: %s [-n num_particles] [-sp particle_size]\n", argv[0]);
                exit(EXIT_FAILURE);
        }
    }

    particles = (Particle*) calloc(num_particles, sizeof(Particle));

    srand(time(NULL));
    for (int i = 0; i < num_particles; i++) {
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

        particles[i] = Particle(Vector(x, y), Vector(dx, dy), 1, particle_size);
    } 
    
    particles[num_particles - 1].setMass(5);
    particles[num_particles - 1].setRadius(0.2f);
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