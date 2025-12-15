inputfile = mm.cu
libcuda = /usr/local/cuda/lib64/
all:

	nvcc -c ${inputfile} -o tmp.o -I./cuspis -g -G 
	g++ tmp.o -o mm -L${libcuda} -ldl -lrt -pthread -I./cuspis -lcuda -lcudart -g
	rm tmp.o

golden:
	./mm >golden_stdout.txt 2>golden_stderr.txt