// Copyright (C) 2009-2016 ETH Zurich
// Copyright (C) 2007?-2016 Center for Nanophase Materials Sciences, ORNL
// All rights reserved.
//
// See LICENSE.txt for terms of usage.
// See CITATION.txt for citation guidelines if you use this code for scientific publications.
//
// Author: Urs R. Haehner (haehneru@itp.phys.ethz.ch)
//
// This file tests mpi_concurrency.hpp.
// It is run with 4 MPI processes.

#include "dca/concurrency/mpi_concurrency/mpi_concurrency.hpp"
#include "gtest/gtest.h"
#include "dca/testing/minimalist_printer.hpp"

dca::concurrency::MPIConcurrency* concurrency_ptr;

TEST(MPIConcurrencyTest, Basic) {
  EXPECT_EQ(4, concurrency_ptr->number_of_processors());
  EXPECT_EQ(0, concurrency_ptr->first());
  EXPECT_EQ(3, concurrency_ptr->last());
}

int main(int argc, char** argv) {
  int result = 0;

  concurrency_ptr = new dca::concurrency::MPIConcurrency(argc, argv);

  ::testing::InitGoogleTest(&argc, argv);

  ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
  if (concurrency_ptr->id() != 0) {
    delete listeners.Release(listeners.default_result_printer());
    listeners.Append(new dca::testing::MinimalistPrinter);
  }

  result = RUN_ALL_TESTS();

  delete concurrency_ptr;

  return result;
}
