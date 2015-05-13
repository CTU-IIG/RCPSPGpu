# Default C++ compiler.
CPP=g++
# Nvidia Cuda compiler.
NVCC=nvcc -lcudart

INST_PATH = /usr/local/bin/

OBJ = ConfigureRCPSP.o InputReader.o SourcesLoad.o ScheduleSolver.o CudaFunctions.o
INC = ConfigureRCPSP.h CudaConstants.h DefaultConfigureRCPSP.h InputReader.h SourcesLoad.h ScheduleSolver.cuh CudaFunctions.cuh
SRC = ConfigureRCPSP.cpp CreateHeaderFile.cpp InputReader.cpp RCPSPGpu.cpp SourcesLoad.cpp ScheduleSolver.cu CudaFunctions.cu

# If yout want to analyse performance then switch -pg (gprof) should be used. Static linkage of standard C++ library (-static-libstdc++).
ifdef DEBUG
OPTIMISATION = -g -G
else
OPTIMISATION = -O3 --maxrregcount=32
endif

CAPABILITY = --generate-code arch=compute_35,code=sm_35
GCC_OPTIONS = -march=native,-Wall,-funsafe-math-optimizations,-pipe,-fopenmp

# Compile all.
build: CreateHeaderFile RCPSPGpu

# Generate documentation.
doc:
	doxygen Documentation/doxyfilelatex; \
	doxygen Documentation/doxyfilehtml

# Compile CreateHeaderFile program.
CreateHeaderFile: InputReader.o CreateHeaderFile.o
	$(NVCC) $(CAPABILITY) $(OPTIMISATION) --compiler-options $(GCC_OPTIONS) -o $@ InputReader.o CreateHeaderFile.o

#c Compile RCPSPGpu program.
RCPSPGpu: $(OBJ) RCPSPGpu.o
	$(NVCC) $(CAPABILITY) $(OPTIMISATION) --compiler-options $(GCC_OPTIONS) -o $@ $(OBJ) RCPSPGpu.o

# Compile *.cpp files to objects.
%.o: %.cpp
	$(CPP) -O2 -march=native -Wall -pedantic -pipe -c -o $@ $<

# Compile *.cu files to objects.
%.o: %.cu
	$(NVCC) $(CAPABILITY) $(OPTIMISATION) --compiler-options $(GCC_OPTIONS) --ptxas-options=-v -dc -o $@ $<

# Dependencies among header files and object files.
${OBJ}: ${INC}

# Install programs.
install: build
	cp CreateHeaderFile $(INST_PATH);	\
	cp RCPSPGpu $(INST_PATH)

# Uninstall programs.
uninstall:
	rm -f $(INST_PATH)CreateHeaderFile;	\
	rm -f $(INST_PATH)RCPSPGpu

# Remove programs and temporary files.
clean:
	rm -f *.o
	rm -f CreateHeaderFile RCPSPGpu

# Create tarball from the project files.
distrib:
	tar -c $(SRC) $(INC) Makefile > RCPSPGpu.tar; \
	bzip2 RCPSPGpu.tar

