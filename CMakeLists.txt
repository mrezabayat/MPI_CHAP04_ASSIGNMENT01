cmake_minimum_required(VERSION 3.0.0)
project(MsgSendRecv VERSION 0.1.0)

include(CTest)
enable_testing()

add_executable(MsgSendRecv main.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

set(MPI_BASE_PATH "/usr/lib/x86_64-linux-gnu/" CACHE PATH "MPI Base Directory")

find_library(MPI_LIB NAMES mpi HINTS ${MPI_BASE_PATH}/lib ${MPI_BASE_PATH}/lib64)
if(MPI_LIB)
  message(STATUS "  MPI:\t Found at ${MPI_LIB}")
else()
  message(FATAL_ERROR "MPI library not found." )
endif()
set(CMAKE_CXX_COMPILER mpicxx)

include_directories(SYSTEM /usr/include/x86_64-linux-gnu/mpi/)