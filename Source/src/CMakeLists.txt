cmake_minimum_required(VERSION 2.6)
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
IF(COMMAND CMAKE_POLICY)
  CMAKE_POLICY(SET CMP0003 NEW)
ENDIF(COMMAND CMAKE_POLICY)

Project(demo)

set(VXL_DIR /usr/local/vxl.bin)
FIND_PACKAGE(VXL REQUIRED)
INCLUDE(${VXL_CMAKE_DIR}/UseVXL.cmake)

FIND_PACKAGE(CUDA REQUIRED)
INCLUDE(FindCUDA)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --gpu-architecture sm_21)

CUDA_ADD_EXECUTABLE(demo demoMain.cxx demoInterface.cxx demoCostFunction.cxx demoKernel.cu)

TARGET_LINK_LIBRARIES(demo vnl rrel)



