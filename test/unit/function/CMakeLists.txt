# Function library's unit tests

add_subdirectory(domains)
add_subdirectory(util)

dca_add_gtest(function_library_test
  GTEST_MAIN
  LIBS function)

dca_add_gtest(set_to_zero_test GTEST_MAIN)
