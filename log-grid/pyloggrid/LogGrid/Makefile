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
	rm -f $(OBJECTS) compute_convolution_kernel.html compute_convolution_kernel.c convolver_c.o
	rm -rf convolver.cpython-*.so ../../convolver.cpython-*.so
	rm -rf build

.c.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET) compute_convolution_kernel.html compute_convolution_kernel.c *.so *.pyd *.o
	rm -rf build
