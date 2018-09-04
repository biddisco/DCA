// Copyright (C) 2012 Jakob Progsch, Václav Zeman
// Copyright (C) 2018 ETH Zurich
// Copyright (C) 2018 UT-Battelle, LLC
// All rights reserved.
//
// See LICENSE for terms of usage.
// See CITATION.md for citation guidelines, if DCA++ is used for scientific publications.
//
// Author: Giovanni Balduzzi (gbalduzz@itp.phys.ethz.ch)
//
// This file implements a thread pool based on the file
// https://github.com/progschj/ThreadPool/blob/master/ThreadPool.h.
// See COPYING for the license of the original work.

#ifndef DCA_PARALLEL_STDTHREAD_THREAD_POOL_THREAD_POOL_HPP
#define DCA_PARALLEL_STDTHREAD_THREAD_POOL_THREAD_POOL_HPP

// Threading includes
#include <mutex>
#include <condition_variable>
#include <future>
#include <thread>
//
#include <iostream>
#include <functional>
#include <memory>
#include <queue>
#include <stdexcept>
#include <vector>

namespace dca {
namespace parallel {

struct thread_traits {
    template <typename T>
    using future_type               = std::future<T>;
    using mutex_type                = std::mutex;
    using condition_variable_type   = std::condition_variable;
    using scoped_lock               = std::lock_guard<mutex_type>;
    using unique_lock               = std::unique_lock<mutex_type>;
    static void yield() {
        std::this_thread::yield();
    }
};

class ThreadPool {
public:
  // Creates a pool with n_threads.
  ThreadPool(size_t n_threads = 0);

  ThreadPool(const ThreadPool& other) = delete;
  ThreadPool(ThreadPool&& other) = default;

  // Enlarge the pool to n_threads if the current number of threads in the pool is inferior.
  void enlarge(std::size_t n_threads);

  // Call asynchronously the function f with arguments args. This method is thread safe.
  // Returns: a future to the result of f(args...).
  template <class F, class... Args>
  auto enqueue(F&& f, Args&&... args) -> thread_traits::future_type<typename std::result_of<F(Args...)>::type>;

  // Conclude all the pending work and destroy the threds spawned by this class.
  ~ThreadPool();

  // Returns the number of threads used by this class.
  std::size_t size() const {
    return workers_.size();
  }

  // Returns a static instance.
  static ThreadPool& get_instance() {
    static ThreadPool global_pool;
    return global_pool;
  }

private:
  void workerLoop(int id);

  // need to keep track of threads so we can join them
  std::vector<std::thread> workers_;
  // the task queue
  std::vector<std::unique_ptr<std::queue<std::packaged_task<void()>>>> tasks_;

  // synchronization
  std::vector<std::unique_ptr<thread_traits::mutex_type>> queue_mutex_;
  std::vector<std::unique_ptr<thread_traits::condition_variable_type>> condition_;
  std::atomic<bool> stop_;
  std::atomic<unsigned int> active_id_;
};

// add new work item to the pool
template <class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args)
    -> thread_traits::future_type<typename std::result_of<F(Args...)>::type> {
  using return_type = typename std::result_of<F(Args...)>::type;
  unsigned int id = active_id_++;
  id = id % size();

  auto task =
      std::packaged_task<return_type()>(std::bind(std::forward<F>(f), std::forward<Args>(args)...));

  thread_traits::future_type<return_type> res = task.get_future();
  {
    std::unique_lock<thread_traits::mutex_type> lock(*queue_mutex_[id]);

    // don't allow enqueueing after stopping the pool
    if (stop_)
      throw std::runtime_error("enqueue on stopped ThreadPool");

    tasks_[id]->emplace(std::move(task));
  }
  condition_[id]->notify_one();

  return res;
}

}  // parallel
}  // dca

#endif  // DCA_PARALLEL_STDTHREAD_THREAD_POOL_THREAD_POOL_HPP
