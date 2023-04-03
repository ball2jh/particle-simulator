objects = main.o particle.o vector.o

all: $(objects)
		nvcc -gencode=arch=compute_52,code=\"sm_52,compute_52\" $(objects) -o app

%.o: %.cpp
		nvcc -x cu -gencode=arch=compute_52,code=\"sm_52,compute_52\" -I. -dc $< -o $@
		
%.o: %.cu
		nvcc -x cu -gencode=arch=compute_52,code=\"sm_52,compute_52\" -I. -dc $< -o $@

clean:
		rm -f *.o app