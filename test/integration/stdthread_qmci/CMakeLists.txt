dca_add_gtest(stdthread_ctaux_tp_test
    EXTENSIVE
    STDTHREAD
    HPX_LARGE_STACKS
    GTEST_MAIN
    INCLUDE_DIRS ${DCA_INCLUDE_DIRS}
    LIBS         ${DCA_LIBS}
    )

if (DCA_HAVE_HPX)
    dca_add_gtest(hpxthread_ctaux_tp_test
        EXTENSIVE
        STDTHREAD
        HPX_LARGE_STACKS
        GTEST_MAIN
        INCLUDE_DIRS ${DCA_INCLUDE_DIRS}
        LIBS         ${DCA_LIBS}
        )
endif()

add_subdirectory(gpu)
