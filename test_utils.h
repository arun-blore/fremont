#ifndef __TEST_UTILS_H__
#define __TEST_UTILS_H__

#include <iostream>
#include <chrono>

#define COMMA ,
#define RUN_TEST(func) func() ? std::cout<<#func" passed"<<std::endl : std::cout<<#func" failed"<<std::endl
#define RUN_TEST1(func) func ? std::cout<<#func" passed"<<std::endl : std::cout<<#func" failed"<<std::endl

std::chrono::time_point<std::chrono::high_resolution_clock> start_time, stop_time;

#define START_CLOCK start_time = std::chrono::high_resolution_clock::now()
#define STOP_CLOCK   stop_time = std::chrono::high_resolution_clock::now()
#define TIME_ELAPSED std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time).count()

#endif
