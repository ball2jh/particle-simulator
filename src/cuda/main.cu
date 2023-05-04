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
#include <stack>
#include <unistd.h>

// #include "helpers/helper_cuda.h"
#include <GL/glew.h>
#include <GL/freeglut.h>

// #include <cuda_gl_interop.h>

#include "particle.cuh"
#include "particle.cu"
#include <curand.h>
#include <curand_kernel.h>

#include "vector.cuh"
#include "vector.cu"

#include <math.h>
#define PI 3.14159265f

int num_particles;
float particle_size;
Particle* particles;
curandState* states;

GLuint vertex_buffer;
struct cudaGraphicsResource *cuda_vbo_resource;
void *d_vbo_buffer = NULL;
Particle* device_particles;

// GL functionality
bool initGL(int *argc, char **argv);
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
               unsigned int vbo_res_flags);


// A cuda kernel
__global__ void checkCollision(Particle* d_particles, int n_particles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    for (int j = i + 1; j < n_particles; j++) {
        if (d_particles[i].collidesWith(d_particles[j])) {
            d_particles[i].resolveCollision(d_particles[j]);
        }
    }
}

__global__ void updateParticles(Particle* d_particles, int n_particles, curandState* states) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_particles) {
        curand_init(i, 0, 0, &states[i]);
        float dx = (float) curand_uniform(&states[i]);
        float scaled = (dx / RAND_MAX) * 2 + 2;
        d_particles[i].updatePosition(scaled);
        d_particles[i].wallBounce();
    }
}

void display() {
	glClear(GL_COLOR_BUFFER_BIT);

    for (int i = 0; i < num_particles; i++) {
        particles[i].renderCircle(); // opengl stuff cant be cuda
        // // make a random number
        // float dx = (float) rand();
        // //// scale it to be between 2 and 4
        // float scaled = (dx / RAND_MAX) * 2 + 2;
        // particles[i].updatePosition(scaled);
        // particles[i].wallBounce();

    }

    int blockSize = 256;
    int blockCount = (num_particles + blockSize - 1) / blockSize;;

    // Send particle data to device
    cudaMemcpy(device_particles, particles, num_particles * sizeof(Particle), cudaMemcpyHostToDevice);
    updateParticles<<<blockCount, 256>>>(device_particles, num_particles, states);
    // Do the cuda stuff
    checkCollision<<<blockCount, 256>>>(device_particles, num_particles);
    // Retrieve particle data from device
    cudaMemcpy(particles, device_particles, num_particles * sizeof(Particle), cudaMemcpyDeviceToHost);

    static int frameCount = 0;
    static int lastTime = 0;
    int currentTime = glutGet(GLUT_ELAPSED_TIME);
    frameCount++;

    if (currentTime - lastTime > 1000) {
        char title[80];
        sprintf(title, "Particle Simulator (%d fps)", frameCount);
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


        // Make Particle -------------
        // make random particle velocity        
        float dx = dist(gen) * 6;
        float dy = dist(gen) * 6;
        // make random particle position
        float x = rand(gen);
        float y = rand(gen);
        particles[i] = Particle(Vector(x, y), Vector(dx, dy), 10000, particle_size);
        // ---------------------------
    }

    // Init the device particles
    cudaMalloc((void**)&device_particles, num_particles * sizeof(Particle));
    cudaMalloc((void**)&states, num_particles * sizeof(curandState));

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

    cudaDeviceSynchronize();
    cudaFree(device_particles);
    cudaFree(states);

    return 0;
}