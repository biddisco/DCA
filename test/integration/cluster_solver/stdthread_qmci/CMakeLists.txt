set(TEST_INCLUDES ${DCA_INCLUDE_DIRS})
set(TEST_LIBS     ${DCA_LIBS})

dca_add_gtest(stdthread_ctaux_tp_test
    EXTENSIVE
    STDTHREAD
    CUDA
    HPX_LARGE_STACKS
    GTEST_MAIN
    INCLUDE_DIRS ${DCA_INCLUDE_DIRS}
    LIBS         ${DCA_LIBS}
    )

add_subdirectory(gpu)
