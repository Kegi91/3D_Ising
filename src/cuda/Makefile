objects = main.o run.o ising.o utils.o pcg/pcg_basic.o

all: $(objects)
	nvcc -arch=sm_30 $(objects) -o main -lm

%.o: %.cu
	nvcc -arch=sm_30 -dc $< -o $@

run: all
	./main read ../../input/input.dat

benchmark: all
	./main benchmark ../../input/benchmark.dat

profile: all
	nvprof --unified-memory-profiling off ./main read ../../input/profile.dat

plots:
	python3 ../python/plots.py
	eog ../../pics/

plot_benchmark:
	python3 ../python/plot_benchmark.py

clean:
	rm -f main *.o pcg/pcg_basic.o

