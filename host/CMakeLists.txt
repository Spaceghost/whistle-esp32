cmake_minimum_required(VERSION 3.16)
project(needle_host C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wno-unused-parameter)

set(ENGINE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../engine)

file(GLOB ENGINE_SRC ${ENGINE_DIR}/src/*.c)

add_library(needle_engine STATIC ${ENGINE_SRC})
target_include_directories(needle_engine PUBLIC ${ENGINE_DIR}/include)
target_link_libraries(needle_engine PUBLIC m)

add_executable(nd_dump nd_dump.c)
target_link_libraries(nd_dump PRIVATE needle_engine)

add_executable(nd_gtest nd_gtest.c)
target_link_libraries(nd_gtest PRIVATE needle_engine)

add_executable(wh_run wh_run.c)
target_link_libraries(wh_run PRIVATE needle_engine)

add_executable(ledcmd_test ledcmd_test.c)
target_link_libraries(ledcmd_test PRIVATE needle_engine)

add_executable(english_test english_test.c)
target_link_libraries(english_test PRIVATE needle_engine)
# Keep assertions active in release/minimum-size test builds.
target_compile_options(english_test PRIVATE -UNDEBUG)
enable_testing()
add_test(NAME english_api COMMAND english_test)
add_test(NAME english_commands COMMAND ledcmd_test)

add_executable(alloc_test alloc_test.c)
target_include_directories(alloc_test PRIVATE ../engine/include ../tests/alloc_stub)
target_compile_definitions(alloc_test PRIVATE ESP_PLATFORM=1)
target_compile_options(alloc_test PRIVATE -UNDEBUG)
add_test(NAME fast_allocation_fallback COMMAND alloc_test)

add_executable(tokenizer_memory_test tokenizer_memory_test.c ../engine/src/nd_cact.c)
target_include_directories(tokenizer_memory_test PRIVATE ../engine/include)
target_compile_options(tokenizer_memory_test PRIVATE -UNDEBUG)
add_test(NAME tokenizer_memory COMMAND tokenizer_memory_test)
