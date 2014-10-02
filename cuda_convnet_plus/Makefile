MODELNAME := _ConvNet

TBB_INCLUDE_PATH := /home/yzc/Downloads/setup/tbb42_20140601oss/include




TBB_LIB_PATH := /home/yzc/Downloads/setup/tbb42_20140601oss/build/linux_intel64_gcc_cc4.8_libc2.19_kernel3.13.0_release

INCLUDES := -DCUDA_5 -L/usr/include -I$(PYTHON_INCLUDE_PATH) -I$(NUMPY_INCLUDE_PATH) -I$(CUDA_SDK_PATH) -I./include -I./include/common -I./include/cudaconv2 -I./include/nvmatrix -I./dummyinclude/ -I$(TBB_INCLUDE_PATH)
LIB := -L/usr/lib -lpthread -L$(ATLAS_LIB_PATH) -L$(CUDA_INSTALL_PATH)/lib64 -lcblas -lGLEW -L$(TBB_LIB_PATH) -ltbb -ltbbmalloc

USECUBLAS   := 1

PYTHON_VERSION=$(shell python -V 2>&1 | cut -d ' ' -f 2 | cut -d '.' -f 1,2)
LIB += -lpython$(PYTHON_VERSION)

GENCODE_ARCH := -gencode=arch=compute_20,code=\"sm_20,compute_20\" -gencode=arch=compute_35,code=\"sm_35,compute_35\"
COMMONFLAGS := -DNUMPY_INTERFACE -DMODELNAME=$(MODELNAME) -DINITNAME=init$(MODELNAME)

EXECUTABLE	:= $(MODELNAME).so

CUFILES				:= $(shell echo src/*.cu src/cudaconv2/*.cu src/nvmatrix/*.cu)
CU_DEPS				:= $(shell echo include/*.cuh include/cudaconv2/*.cuh include/nvmatrix/*.cuh)
CCFILES				:= $(shell echo src/common/*.cpp)
C_DEPS				:= $(shell echo include/common/*.h)

include common-gcc-cuda-6.5.mk
	
makedirectories:
	$(VERBOSE)mkdir -p $(LIBDIR)
	$(VERBOSE)mkdir -p $(OBJDIR)/src/cudaconv2
	$(VERBOSE)mkdir -p $(OBJDIR)/src/nvmatrix
	$(VERBOSE)mkdir -p $(OBJDIR)/src/common
	$(VERBOSE)mkdir -p $(TARGETDIR)
