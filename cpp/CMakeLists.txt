cmake_minimum_required(VERSION 3.16)
project(simple_ai_native VERSION 0.0.1 LANGUAGES CXX)

add_library(simple_ai_native STATIC src/simple_ai.cpp)
target_include_directories(simple_ai_native PUBLIC include)
target_compile_features(simple_ai_native PUBLIC cxx_std_17)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(simple_ai_native PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(simple_ai_native_demo src/main.cpp)
target_link_libraries(simple_ai_native_demo PRIVATE simple_ai_native)
