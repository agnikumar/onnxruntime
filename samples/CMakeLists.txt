# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.13)

# Project
project(onnxruntime_samples C CXX)
if (WIN32)
 string(APPEND CMAKE_CXX_FLAGS " /W4")
else()
 string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra")
 string(APPEND CMAKE_C_FLAGS " -Wall -Wextra")
endif()

find_package(JPEG)
find_package(PNG)


add_definitions(-DUSE_CUDA)
if(JPEG_FOUND)
 add_executable(image_classifier "main.cc" "jpeg_handle.cc" "jpeg_handle.h" "jpeg_mem.cc" "jpeg_mem.h")
 target_link_libraries(image_classifier PRIVATE onnxruntime ${JPEG_LIBRARIES})
endif()

#TODO: remove this part(need to talk to Ryan and find a solution for this)
if(UNIX)
 include_directories("/usr/local/include/onnxruntime")
endif()
include_directories(../cmake/external/eigen)

add_executable(fns_candy_style_transfer "fns_candy_style_transfer.c")
target_include_directories(fns_candy_style_transfer PRIVATE "${TEST_SRC_DIR}/util/include" ${PNG_INCLUDE_DIRS})
target_link_libraries(fns_candy_style_transfer PRIVATE onnxruntime ${PNG_LIBRARIES})
set_target_properties(fns_candy_style_transfer PROPERTIES FOLDER "ONNXRuntimeTest")