# Location of the CUDA Toolkit
CUDA_PATH ?= /usr/local/cuda

TARGET_ARCH ?= x86_64

TARGET_SIZE := 64

# operating system
HOST_OS   := $(shell uname -s 2>/dev/null | tr "[:upper:]" "[:lower:]")

HOST_COMPILER ?= g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# internal flags
NVCCFLAGS   := -m${TARGET_SIZE}
CCFLAGS     :=
LDFLAGS     :=

# Debug build flags
ifeq ($(dbg),1)
      NVCCFLAGS += -g -G
      BUILD_TYPE := debug
else
      BUILD_TYPE := release
endif

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(EXTRA_CCFLAGS))


ALL_LDFLAGS := 
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
ALL_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))

GL_LDFLAGS := -lglut -lGL -lGLU

# Common includes and paths for CUDA
INCLUDES  := -I../common
LIBRARIES :=

ALL_CCFLAGS += --threads 0 --std=c++11 


################################################################################

# Target rules
all: build

build: heat_gl heat_perf

heat_gl.o:heat_gl.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

heat_gl: heat_gl.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

heat_perf.o:heat_perf.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

heat_perf: heat_perf.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

clean:
	rm -f heat_perf heat_perf.o heat_gl heat_gl.o texture_result_file.* linear_result_file.*

