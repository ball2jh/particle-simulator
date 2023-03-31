

#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "helper_cuda.h"
#include "helper_gl.h"

#include "ball.h"
#include "vector.h"

bool runTest(int argc, char** argv, char* ref_file);

//__global__ void updateBallPositions(Ball* balls, float deltaTime, int numBalls) {
//	int idx = blockIdx.x * blockDim.x + threadIdx.x;
//	if (idx < numBalls) {
//		//balls[idx].updatePosition(deltaTime);
//	}
//}

int main(int argc, char **argv)
{
#if defined(__linux__)
	setenv("DISPLAY", ":0", 0);
#endif

	printf("Hello World!\n");
	runTest(argc, argv, NULL);
	return 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
bool runTest(int argc, char** argv, char* ref_file)
{
    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    int devID = findCudaDevice(argc, (const char**)argv);
	printf("devID: %d\n", devID);

	int numBalls = 30;
	Ball* host_balls = new Ball[numBalls];

	host_balls[0] = Ball(Vector(0, 0), Vector(1, 0), 1, 1);

// 	Ball* device_balls;
// 	checkCudaErrors(cudaMalloc((void**)&device_balls, numBalls * sizeof(Ball)));

// 	checkCudaErrors(cudaMemcpy(device_balls, host_balls, numBalls * sizeof(Ball), cudaMemcpyHostToDevice));

// 	int numIterations = 1000;
// 	float deltaTime = 0.01f;

// 	for (int i = 0; i < numIterations; i++) {
// 		updateBallPositions<<<1, 1>>>(device_balls, deltaTime, numBalls);
// 		checkCudaErrors(cudaGetLastError());
// 		checkCudaErrors(cudaDeviceSynchronize());

// 		checkCudaErrors(cudaMemcpy(host_balls, device_balls, numBalls * sizeof(Ball), cudaMemcpyDeviceToHost));

// 		for (int i = 0; i < numBalls; ++i) {
// 			Ball& ball = host_balls[i];
// 			Vector position = ball.getPosition();
// 			printf("Ball %d: (%f, %f)\n", i, position.getX(), position.getY());
// 		}
// 	}

// 	checkCudaErrors(cudaFree(device_balls));
// 	delete[] host_balls;

    return true;
}

