objects = main.o run.o ising.o utils.o pcg/pcg_basic.o

all: $(objects)
	nvcc -arch=sm_30 $(objects) -o main -lm

%.o: %.cu
	nvcc -arch=sm_30 -dc $< -o $@

run: all
	./main read_n_run ../../input/input.dat

benchmark: all
	./main benchmark ../../input/benchmark.dat

plots:
	python3 ../python/plots.py
	eog ../../pics/

clean:
	rm -f main *.o pcg/pcg_basic.o

