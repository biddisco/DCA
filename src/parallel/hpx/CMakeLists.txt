# HPX

configure_file("${PROJECT_SOURCE_DIR}/include/dca/config/hpx_defines.hpp.in"
  "${CMAKE_BINARY_DIR}/include/dca/config/hpx_defines.hpp"
  @ONLY)

add_library(parallel_hpx STATIC hpxthread.cpp)

