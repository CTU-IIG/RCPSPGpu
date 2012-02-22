# Pokud chcete ladit vykonost, pouzijte volbu -pg (gprof).
ifdef DEBUG
OPTIMISATION = --debug -G0 -O0
else
OPTIMISATION = -O3
endif

CAPABILITY = --generate-code arch=compute_20,code=sm_21 --maxrregcount=32
GCC_OPTIONS = -march=native,-Wall,-funsafe-math-optimizations,-pipe

# Defaultní volba pro příkaz make.
build: CreateHeaderFile RCPSPGpu

# Zkompiluje program.
CreateHeaderFile:
	nvcc $(CAPABILITY) $(OPTIMISATION) -o CreateHeaderFile CreateHeaderFile.cpp InputReader.cpp --compiler-bindir CudaGCC/ --compiler-options $(GCC_OPTIONS)

RCPSPGpu:
	nvcc $(CAPABILITY) $(OPTIMISATION) --ptxas-options=-v -o RCPSPGpu RCPSPGpu.cpp InputReader.cpp ScheduleSolver.cu SourcesLoad.cpp --compiler-bindir CudaGCC/ --compiler-options $(GCC_OPTIONS)

clean:
	rm -f CreateHeaderFile RCPSPGpu

