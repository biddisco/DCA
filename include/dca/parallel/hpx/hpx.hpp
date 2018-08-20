// Copyright (C) 2009-2016 ETH Zurich
// Copyright (C) 2007?-2016 Center for Nanophase Materials Sciences, ORNL
// All rights reserved.
//
// See LICENSE.txt for terms of usage.
// See CITATION.txt for citation guidelines if you use this code for scientific publications.
//
// Author: Peter Staar (taa@zurich.ibm.com)
//         Urs R. Haehner (haehneru@itp.phys.ethz.ch)
//
// This class provides an interface for parallelizing with HPX.
//
// TODO: Finish sum methods.

#ifndef DCA_PARALLEL_HPX_HPX_HPP
#define DCA_PARALLEL_HPX_HPX_HPP

#include <vector>
#include <thread>
#include "dca/parallel/util/threading_data.hpp"

namespace dca {
namespace parallel {

class hpxthread {
public:
  hpxthread();

  void execute(int num_threads, void* (*start_routine)(void*), void* arg)
  {
      fork(num_threads, start_routine, arg);
      join();
  }

private:
  void fork(int num_threads, void* (*start_routine)(void*), void* arg)
  {
      threads_.clear();
      data_.resize(num_threads);

      for (int id = 0; id < num_threads; id++) {
        data_[id].id = id;
        data_[id].num_threads = num_threads;
        data_[id].arg = arg;
        //
        threads_.push_back(std::thread(start_routine, (void*)(&data_[id])));
      }
  }

  void join() {
      for (int id = 0; id < threads_.size(); id++) {
        threads_[id].join();
      }
      threads_.clear();
  }

  std::vector<std::thread>   threads_;
  std::vector<ThreadingData> data_;
};

}  // parallel
}  // dca

#endif  // DCA_PARALLEL_hpxthread_hpxthread_HPP
