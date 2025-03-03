cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(dlrm)

find_package(Torch REQUIRED)

option(DOWNLOAD_MNIST "Download the MNIST dataset from the internet" ON)
if (DOWNLOAD_MNIST)
  message(STATUS "Downloading MNIST dataset")
  execute_process(
    COMMAND python ${CMAKE_CURRENT_LIST_DIR}/../tools/download_mnist.py
      -d ${CMAKE_BINARY_DIR}/data
    ERROR_VARIABLE DOWNLOAD_ERROR)
  if (DOWNLOAD_ERROR)
    message(FATAL_ERROR "Error downloading MNIST dataset: ${DOWNLOAD_ERROR}")
  endif()
endif()

add_executable(dlrm dlrm.cpp)
target_link_libraries(dlrm "${TORCH_LIBRARIES}")
set_property(TARGET dlrm PROPERTY CXX_STANDARD 14)

if(CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "Running debug build.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
endif()

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET dlrm
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR::lrm>)
endif (MSVC)
