#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <cstdlib>
#include <unistd.h>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include "particle_serial.h"
#include "particle_serial.cpp"
#include "vector_serial.h"
#include "vector_serial.cpp"

#include <math.h>
#define PI 3.14159265f

int num_particles;
float particle_size;
Particle* particles;

// GL functionality
bool initGL(int *argc, char **argv);

// OpenGL rendering
void display() {
	glClear(GL_COLOR_BUFFER_BIT);

    for (int i = 0; i < num_particles; i++) {
        // Render the particle
        particles[i].renderCircle();
        float dx = (float) rand();
        float scaled = (dx / RAND_MAX) * 2 + 2;

        // Update the particle's position, check for wall collision
        particles[i].updatePosition(scaled);
        particles[i].wallBounce();

        // Check for collisions with other particles
        for (int j = i + 1; j < num_particles; j++) {
            if (particles[i].collidesWith(particles[j])) {
                particles[i].resolveCollision(particles[j]);
            }
        }
    }

    // FPS counter
    static int frameCount = 0;
    static int lastTime = 0;
    int currentTime = glutGet(GLUT_ELAPSED_TIME);
    frameCount++;

    if (currentTime - lastTime > 1000) {
        char title[80];
        sprintf(title, "Particle Simulator Serial (%d fps) - %d particles", frameCount, num_particles);
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
    glutPositionWindow(100,100);
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

    // Set defaults
    srand(time(NULL));
    num_particles = 10;
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

    for (int i = 0; i < num_particles; i++) {
        std::random_device rd;
        std::mt19937 gen(rd());

        std::uniform_real_distribution<float> dist(-0.0015, 0.0015);
        std::uniform_real_distribution<float> rand(-0.95, 0.95);

        std::uniform_real_distribution<float> mass(1.5, 5.5);

        // make random particle velocity        
        float dx = dist(gen) * 6;
        float dy = dist(gen) * 6;

        // make random particle position
        float x = rand(gen);
        float y = rand(gen);

        particles[i] = Particle(Vector(x, y), Vector(dx, dy), mass(gen), particle_size);
    } 
    

    initGL(&argc, argv);
    glutMainLoop();

    return 0;
}