#CC=gcc
CC=clang
CFLAGS=-c -g -fpic -Wall -Wextra -O3 -march=native -ffast-math -fopenmp -DUSE_OMP
LDFLAGS=-shared -fopenmp
SOURCES=convolver_c.c
OBJECTS=$(SOURCES:.c=.o)
TARGET=convolver_c.so

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@
	python setup_linux.py build_ext --inplace
	rm -f $(OBJECTS) convolver.html convolver.c
	rm -rf build

.c.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET) convolver.html convolver.c convolver.*.so
	rm -rf build
