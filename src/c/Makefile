SRC_FILES=main.c ising.c utils.c pcg/pcg_basic.c run.c

all: main

main: $(SRC_FILES)
	gcc -Wall -Wextra -pedantic -std=c99 -o $@ $(SRC_FILES) -lm -O3

clean:
	rm -f main

run: main
	./main read ../../input/input.dat

benchmark: main
	./main benchmark ../../input/benchmark.dat

plots:
	python3 ../python/plots.py
	eog ../../pics/
