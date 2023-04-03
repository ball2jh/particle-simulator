#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>

#include "helpers/helper_cuda.h"
#include "particle.h"

#define START_TIMER(NAME) gettimeofday(&tv, NULL); \
    double NAME ## _time = tv.tv_sec+(tv.tv_usec/1000000.0);
#define STOP_TIMER(NAME) gettimeofday(&tv, NULL); \
    NAME ## _time = tv.tv_sec+(tv.tv_usec/1000000.0) - (NAME ## _time);
#define GET_TIMER(NAME) (NAME##_time)


__global__ void updateParticlePositions(Particle* particles, float deltaTime, int numParticles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParticles) {
        particles[idx].updatePosition(deltaTime);
    }
}

int main(int argc,  char** argv) {
    int devID = findCudaDevice(argc, (const char**)argv);
    printf("devID: %d\n", devID);

    int numParticles = 50000000;
    Particle* host_particles = new Particle[numParticles];
    Particle* device_particles;

    checkCudaErrors(cudaMalloc(&device_particles, numParticles * sizeof(Particle)));

    // Give each particle a random velocity
    for (int i = 0; i < numParticles; i++) {
        Vector velocity = Vector(rand() % 100, rand() % 100);
        host_particles[i] = Particle(Vector(0, 0), velocity, 1, 1);
    }

    // Copy particles from host to device
    checkCudaErrors(cudaMemcpy(device_particles, host_particles, numParticles * sizeof(Particle), cudaMemcpyHostToDevice));

    int numIterations = 1;
    float deltaTime = 1.0f;

    int blockSize = 256;
    int gridSize = (numParticles + blockSize - 1) / blockSize;

    struct timeval tv;
    START_TIMER(total);
    for (int i = 0; i < numIterations; i++) {
        updateParticlePositions<<<gridSize, blockSize>>>(device_particles, deltaTime, numParticles);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        // Copy particles from device to host
        checkCudaErrors(cudaMemcpy(host_particles, device_particles, numParticles * sizeof(Particle), cudaMemcpyDeviceToHost));

        //Uncomment the lines below if you want to print particle information
        // for (int j = 0; j < numParticles; j++) {
        //     Particle& particle = host_particles[j];
        //     Vector position = particle.getPosition();
        //     Vector velocity = particle.getVelocity();
        //     printf("Particle %d: (%f, %f) (%f, %f)\n", j, position.getX(), position.getY(), velocity.getX(), velocity.getY());
        // }
    }
    STOP_TIMER(total);

    printf("Total time: %f\n", GET_TIMER(total));

    checkCudaErrors(cudaFree(device_particles));
    delete[] host_particles;
}