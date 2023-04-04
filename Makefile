objects = main.o

all: $(objects)
		nvcc -gencode=arch=compute_52,code=\"sm_52,compute_52\" $(objects) -o app -L/usr/lib/x86_64-linux-gnu -lGL -lGLU -lglut

%.o: %.cpp
		nvcc -x cu -gencode=arch=compute_52,code=\"sm_52,compute_52\" -I. -dc -L/usr/lib/x86_64-linux-gnu -lGL -lGLU -lglut $< -o $@
		
%.o: %.cu
		nvcc -x cu -gencode=arch=compute_52,code=\"sm_52,compute_52\" -I. -dc -L/usr/lib/x86_64-linux-gnu -lGL -lGLU -lglut $< -o $@

clean:
		rm -f *.o app