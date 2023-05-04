objects = src/cuda/main.o
GL_LIBS = -L/usr/lib/x86_64-linux-gnu -lGLEW -lGL -lGLU -lglut

all: $(objects)
	nvcc -gencode=arch=compute_52,code=\"sm_52,compute_52\" $(objects) -o app $(GL_LIBS)
	g++ -std=c++11 -I. -o app_serial src/serial/main_serial.cpp $(GL_LIBS)

%.o: %.cpp
	nvcc -x cu -gencode=arch=compute_52,code=\"sm_52,compute_52\" -I. -dc $(GL_LIBS) $< -o $@
		
%.o: %.cu
	nvcc -x cu -gencode=arch=compute_52,code=\"sm_52,compute_52\" -I. -dc $(GL_LIBS) $< -o $@

clean:
	find . -name "*.o" -delete
	find . -name "app" -delete
	find . -name "app_serial" -delete