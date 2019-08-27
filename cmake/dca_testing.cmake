################################################################################
# Author: Urs R. Haehner (haehneru@itp.phys.ethz.ch)
#
# Enables testing.
# References: - https://github.com/ALPSCore/ALPSCore

include(CMakeParseArguments)

# Adds a test written with Google Test.
#
# dca_add_gtest(name
#               [FAST | EXTENSIVE | STOCHASTIC | PERFORMANCE]
#               [GTEST_MAIN]
#               [MPI [MPI_NUMPROC procs]]
#               [CUDA]
#               [STDTHREAD] [HPX]
#               [INCLUDE_DIRS dir1 [dir2 ...]]
#               [SOURCES src1 [src2 ...]]
#               [LIBS lib1 [lib2 ...]])
#
# Adds a test called 'name', the source is assumed to be 'name.cpp'.
# The type of the test can be FAST, EXTENSIVE, STOCHASTIC or PERFORMANCE (mutually exclusive
# options). If no option is specified, the default is FAST.
# MPI or CUDA may be given to indicate that the test requires these libraries. MPI_NUMPROC is the
# number of MPI processes to use for a test with MPI, the default value is 1.
function(dca_add_gtest name)
  set(options FAST EXTENSIVE STOCHASTIC PERFORMANCE GTEST_MAIN MPI CUDA STDTHREAD HPX)
  set(oneValueArgs MPI_NUMPROC)
  set(multiValueArgs INCLUDE_DIRS SOURCES LIBS)
  cmake_parse_arguments(DCA_ADD_GTEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  # FAST, EXTENSIVE and PERFORMANCE are mutually exclusive.
  if ((DCA_ADD_GTEST_FAST AND DCA_ADD_GTEST_EXTENSIVE) OR
      (DCA_ADD_GTEST_FAST AND DCA_ADD_GTEST_PERFORMANCE) OR
      (DCA_ADD_GTEST_EXTENSIVE AND DCA_ADD_GTEST_PERFORMANCE) OR
      (DCA_ADD_GTEST_STOCHASTIC AND DCA_ADD_GTEST_FAST) OR
      (DCA_ADD_GTEST_STOCHASTIC AND DCA_ADD_GTEST_EXTENSIVE) OR
      (DCA_ADD_GTEST_STOCHASTIC AND DCA_ADD_GTEST_PERFORMANCE))
    message(FATAL_ERROR "Incorrect use of dca_add_gtest.\n
                         dca_add_gtest(name\n
                                       [FAST | EXTENSIVE | STOCHASTIC | PERFORMANCE]\n
                                       [GTEST_MAIN]\n
                                       [MPI [MPI_NUMPROC procs]]\n
                                       [CUDA]\n
                                       [HPX]\n
                                       [INCLUDE_DIRS dir1 [dir2 ...]]\n
                                       [SOURCES src1 [src2 ...]]\n
                                       [LIBS lib1 [lib2 ...]])")
  endif()

  # Only build the test if the corresponding option is set.
  if (DCA_ADD_GTEST_PERFORMANCE)
    if (NOT DCA_WITH_TESTS_PERFORMANCE)
      return()
    endif()
    # Only build performance tests in Release mode.
    if (NOT (CMAKE_BUILD_TYPE STREQUAL "Release"))
      return ()
    endif()

  elseif (DCA_ADD_GTEST_EXTENSIVE)
    if (NOT DCA_WITH_TESTS_EXTENSIVE)
      return()
    endif()

  elseif (DCA_ADD_GTEST_STOCHASTIC)
    if (NOT DCA_WITH_TESTS_STOCHASTIC)
      return()
    endif()

  else()  # Default is FAST.
    if (NOT DCA_WITH_TESTS_FAST)
      return()
    endif()
  endif()

  # Only build the test if the required libraries are available.
  if (DCA_ADD_GTEST_MPI AND NOT DCA_HAVE_MPI)
    return()
  endif()

  # Only build the test if the required libraries are available.
  if (DCA_ADD_GTEST_HPX AND NOT DCA_HAVE_HPX)
    return()
  endif()

  if (DCA_ADD_GTEST_CUDA AND NOT DCA_HAVE_CUDA)
    return()
  endif()

  if (DCA_ADD_GTEST_HPX)
    set(test_name ${name}_HPX)
  else()
    set(test_name ${name})
  endif()
  add_executable(${test_name} ${name}.cpp ${DCA_ADD_GTEST_SOURCES})

  if(DCA_ADD_GTEST_HPX AND DCA_HAVE_HPX)
    hpx_setup_target(${test_name})
  endif()
  if(DCA_ADD_GTEST_STDTHREAD AND DCA_HAVE_HPX)
    message("HPX config for ${test_name}")
    hpx_setup_target(${test_name})
  endif()

  # Create a macro with the project source dir. We use this as the root path for reading files in
  # tests.
  target_compile_definitions(${test_name} PRIVATE DCA_SOURCE_DIR=\"${PROJECT_SOURCE_DIR}\")

  if (DCA_ADD_GTEST_GTEST_MAIN)
    # Use gtest main.
    target_link_libraries(${test_name} gtest_main ${DCA_ADD_GTEST_LIBS})
    if (DCA_ADD_GTEST_STDTHREAD AND DCA_HAVE_HPX)
      target_compile_definitions(${test_name} PRIVATE "DCA_HPX_MAIN")
      message("HPX command-line-options for ${test_name}")
      set(TEST_COMMAND_LINE_OPTIONS
          "${TEST_COMMAND_LINE_OPTIONS} --hpx:ini=hpx.stacks.small_size=131072")
      if (DCA_ADD_GTEST_MPI_NUMPROC GREATER 1)
          set(TEST_COMMAND_LINE_OPTIONS
              "${TEST_COMMAND_LINE_OPTIONS} --hpx:threads=1")
      endif()
    endif()
  else()
    # Test has its own main.
    target_link_libraries(${test_name} gtest ${DCA_ADD_GTEST_LIBS})
  endif()

  if (DCA_ADD_GTEST_CUDA)
    target_include_directories(${test_name} PRIVATE ${CUDA_TOOLKIT_INCLUDE})
    target_link_libraries(${test_name} ${DCA_CUDA_LIBS})
    target_compile_definitions(${test_name} PRIVATE DCA_HAVE_CUDA)
    if(DCA_HAVE_MAGMA)
      target_include_directories(${test_name} PRIVATE ${MAGMA_INCLUDE_DIR})
      target_compile_definitions(${test_name} PRIVATE DCA_HAVE_MAGMA)
    endif()
    cuda_add_cublas_to_target(${test_name})
  endif()

  target_include_directories(${test_name} PRIVATE
    ${gtest_SOURCE_DIR}/include
    ${DCA_ADD_GTEST_INCLUDE_DIRS})

  if (DCA_ADD_GTEST_MPI)
    if (NOT DEFINED DCA_ADD_GTEST_MPI_NUMPROC)
      set(DCA_ADD_GTEST_MPI_NUMPROC 1)
    endif()

    add_test(NAME ${test_name}
             COMMAND ${TEST_RUNNER} ${MPIEXEC_NUMPROC_FLAG} ${DCA_ADD_GTEST_MPI_NUMPROC}
                     ${MPIEXEC_PREFLAGS} "$<TARGET_FILE:${test_name}>"
                     "${TEST_COMMAND_LINE_OPTIONS}")
    target_link_libraries(${test_name} ${MPI_C_LIBRARIES})

  else()
    if (TEST_RUNNER)
      add_test(NAME ${test_name}
               COMMAND ${TEST_RUNNER} ${MPIEXEC_NUMPROC_FLAG} 1
                   ${MPIEXEC_PREFLAGS} "$<TARGET_FILE:${test_name}>"
                   "${TEST_COMMAND_LINE_OPTIONS}")
    else (TEST_RUNNER)
      add_test(NAME ${test_name}
               COMMAND "$<TARGET_FILE:${test_name}>"
               "${TEST_COMMAND_LINE_OPTIONS}")
    endif (TEST_RUNNER)
  endif()

endfunction()
