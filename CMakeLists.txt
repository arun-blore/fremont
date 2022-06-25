cmake_minimum_required (VERSION 3.21)
project(my_vector)

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_definitions(NDEBUG)
endif()

set (VEC_SOURCES test_my_vector.cpp)
set (LIST_SOURCES test_my_list.cpp)

add_executable(vec_test ${VEC_SOURCES})
target_include_directories (vec_test PUBLIC .)

add_executable(list_test ${LIST_SOURCES})
target_include_directories (list_test PUBLIC .)

#add_executable(gdb_mem gdb_mem.cpp)

set (MAT_SOURCES test_my_mat.cpp)
add_executable(mat_test ${MAT_SOURCES})
target_include_directories(mat_test PUBLIC .)
set(CMAKE_CXX_FLAGS "--std=c++17 -ftree-vectorize -fopt-info-vec-missed -march=native -ftree-vectorizer-verbose=2 -fopenmp")

if (${CMAKE_BUILD_TYPE} EQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG") # disable calls to assert()
endif()
#set_target_properties(mat_test PROPERTIES COMPILE_FLAGS "-pg")
#set_target_properties(mat_test PROPERTIES LINK_FLAGS "-pg")
#set_target_properties(mat_test PROPERTIES COMPILE_FLAGS "-S -fverbose-asm")
