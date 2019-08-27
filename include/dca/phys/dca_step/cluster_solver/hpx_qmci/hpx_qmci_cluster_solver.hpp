// Copyright (C) 2009-2016 ETH Zurich
// Copyright (C) 2007?-2016 Center for Nanophase Materials Sciences, ORNL
// All rights reserved.
//
// See LICENSE.txt for terms of usage.
// See CITATION.txt for citation guidelines if you use this code for scientific publications.
//
// Author: Peter Staar (taa@zurich.ibm.com)
//         Raffaele Solca' (rasolca@itp.phys.ethz.ch)
//         Urs R. Haehner (haehneru@itp.phys.ethz.ch)
//         John Biddiscombe (biddisco@cscs.ch)
//
// An HPX MC integrator that implements a threaded MC integration independent of the MC method.

#ifndef DCA_PHYS_DCA_STEP_CLUSTER_SOLVER_STDTHREAD_QMCI_HPX_QMCI_CLUSTER_SOLVER_HPP
#define DCA_PHYS_DCA_STEP_CLUSTER_SOLVER_STDTHREAD_QMCI_HPX_QMCI_CLUSTER_SOLVER_HPP

#include <hpx/hpx.hpp>
#include <hpx/include/thread_executors.hpp>

#define hexpointer(p) \
  "0x" << std::setfill('0') << std::setw(12) << std::hex << (uintptr_t)(p) << " "
#define hexuint32(p) "0x" << std::setfill('0') << std::setw(8) << std::hex << (uint32_t)(p) << " "
#define hexlength(p) "0x" << std::setfill('0') << std::setw(6) << std::hex << (uintptr_t)(p) << " "
#define hexnumber(p) "0x" << std::setfill('0') << std::setw(4) << std::hex << p << " "
#define decnumber(p) "" << std::dec << p << " "
#define ipaddress(p)                                                               \
  "" << std::dec << (int)((uint8_t*)&p)[0] << "." << (int)((uint8_t*)&p)[1] << "." \
     << (int)((uint8_t*)&p)[2] << "." << (int)((uint8_t*)&p)[3] << " "

#include <hpx/util/logging.hpp>
#include <hpx/lcos/local/mutex.hpp>
#include <hpx/include/future.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/parallel/executors.hpp>
#include <hpx/include/thread_executors.hpp>
#include <hpx/runtime/threads/thread_enums.hpp>
#include <hpx/runtime/threads/executors/default_executor.hpp>
//
#include <boost/chrono/chrono.hpp>
//
#include <dca/config/config_defines.hpp>
#include <dca/config/hpx_defines.hpp>
//
#include <cassert>
#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <map>
#include <queue>
//
#include "dca/phys/dca_step/cluster_solver/hpx_qmci/hpx_qmci_accumulator.hpp"
#include "dca/phys/dca_step/cluster_solver/thread_task_handler.hpp"
#include "dca/profiling/events/time.hpp"
#include "dca/util/print_time.hpp"

namespace dca {
namespace phys {
namespace solver {
// dca::phys::solver::

template <class qmci_integrator_type>
struct HPXQmciClusterSolver
    : public hpx::components::simple_component_base<HPXQmciClusterSolver<qmci_integrator_type>>,
      protected qmci_integrator_type
{
  using Data = typename qmci_integrator_type::DataType;
  typedef typename qmci_integrator_type::ParametersType parameters_type;
  //
  using random_number_generator = typename parameters_type::random_number_generator;
  //
  typedef typename qmci_integrator_type::Profiler        profiler_type;
  typedef typename qmci_integrator_type::Concurrency     concurrency_type;
  //
  typedef typename qmci_integrator_type::Walker          walker_type;
  typedef typename qmci_integrator_type::Accumulator     accumulator_type;
  //
  typedef HPXQmciClusterSolver<qmci_integrator_type>          this_type;
  typedef hpxqmci::hpx_qmci_accumulator<accumulator_type>     HPX_accumulator_type;
  //
  typedef typename std::shared_ptr<walker_type>               walker_ptr;
  typedef typename std::shared_ptr<HPX_accumulator_type>      accumulator_ptr;
  //
  typedef hpx::lcos::local::mutex      mutex_type;
  typedef std::lock_guard<mutex_type>  scoped_lock;
  typedef std::unique_lock<mutex_type> unique_lock;
  typedef hpx::future<walker_ptr>      walker_future;
  typedef hpx::future<accumulator_ptr> accumulator_future;

public:
  HPXQmciClusterSolver(parameters_type& parameters_ref, Data& data_ref);

  ~HPXQmciClusterSolver();

  template <typename Writer>
  void write(Writer& reader);

  void initialize(int dca_iteration_);

  void integrate();

  template <typename dca_info_struct_t>
  double finalize(dca_info_struct_t& dca_info_struct);

private:
  using qmci_integrator_type::parameters_;
  using qmci_integrator_type::data_;
  using qmci_integrator_type::concurrency_;
  using qmci_integrator_type::total_time_;
  using qmci_integrator_type::dca_iteration_;
  using qmci_integrator_type::accumulator_;

  walker_ptr walker_initialize(int id);
  walker_ptr walker_sweep(walker_ptr walker);
  walker_ptr walker_sweep_completed(walker_ptr walker);

  accumulator_ptr accumulator_initialize(int id);
  accumulator_ptr accumulator_try_measure(accumulator_ptr accum);
  accumulator_ptr accumulator_measure(accumulator_ptr accum, walker_ptr walker);

  /*
  // needed if we want to execute start_walker remotely
  struct walker_action : hpx::actions::make_action<
    walker_ptr (HPXQmciClusterSolver<qmci_integrator_type>::*)(int),
    &HPXQmciClusterSolver<qmci_integrator_type>::start_walker, walker_action>
  {};

  // needed if we want to execute accumulator_initialize remotely
  struct accumulator_action : hpx::actions::make_action<
    accumulator_ptr (HPXQmciClusterSolver<qmci_integrator_type>::*)(int),
    &HPXQmciClusterSolver<qmci_integrator_type>::accumulator_initialize, accumulator_action>
  {};
  */

  void warm_up(walker_ptr walker, int id);

  // TODO: Are the following using statements redundant and can therefore be removed?
  using qmci_integrator_type::computeErrorBars;
  //using qmci_integrator_type::symmetrize_measurements;

private:
  //
  std::atomic<int> walkers_active;
//  std::atomic<int> dca_iteration_;
  std::atomic<int> acc_finished;

  hpx::lcos::local::promise<void> accumulators_ready;
  hpx::future<void> accumulators_finished;

  const int nr_walkers;
  const int nr_accumulators;

  ThreadTaskHandler thread_task_handler_;
  std::vector<random_number_generator> rng_vector;
  hpx::threads::executors::default_executor large_stack_executor;

  // When an accumulator is waiting for work but no walkers have finished
  // it will be pushed onto this queue where it will be popped off by a
  // walker when it is ready
  mutex_type                  accumulators_queue_mutex;
  std::queue<accumulator_ptr> accumulators_queue;

  // When a walker finishes and has no extra steps to perform, it will be pushed
  // onto this queue and the next accumulator to finish will pop it off
  // and start measuring on it
  mutex_type             walkers_queue_mutex;
  std::queue<walker_ptr> walkers_queue;

  typedef std::pair<HPX_accumulator_type*, std::atomic<int>> map_pair;
  std::map<HPX_accumulator_type*, std::atomic<int>>          accumulator_counter;

  mutex_type merge_mutex;
};

//----------------------------------------------------------------------------
template <class qmci_integrator_type>
HPXQmciClusterSolver<qmci_integrator_type>::HPXQmciClusterSolver(parameters_type& parameters_ref,
                                                                 Data& data_ref)
    : qmci_integrator_type(parameters_ref, data_ref),
      walkers_active(0),
      acc_finished(0),
      nr_walkers(parameters_.get_walkers()),
      nr_accumulators(parameters_.get_accumulators()),
      thread_task_handler_(nr_walkers, nr_accumulators),
      large_stack_executor(hpx::threads::thread_stacksize_large)
{
  DCA_LOG("HPXQmciClusterSolver constructor");

  if (nr_walkers < 1 || nr_accumulators < 1) {
    throw std::logic_error(
        "Both the number of walkers and the number of accumulators must be at least 1.");
  }

  for (int i = 0; i < nr_walkers; ++i) {
    auto seed = parameters_.get_seed();
    std::cout << "Seeding walker "<< i << " with seed " << seed << std::endl;
    rng_vector.emplace_back(concurrency_.id(), concurrency_.number_of_processors(), seed);
  }
}

//----------------------------------------------------------------------------
template <class qmci_integrator_type>
HPXQmciClusterSolver<qmci_integrator_type>::~HPXQmciClusterSolver() {}

//----------------------------------------------------------------------------
template <class qmci_integrator_type>
template <typename Writer>
void HPXQmciClusterSolver<qmci_integrator_type>::write(Writer& writer)
{
  DCA_LOG("HPXQmciClusterSolver write");
  qmci_integrator_type::write(writer);
  // accumulator.write(writer);
}

//----------------------------------------------------------------------------
template <class qmci_integrator_type>
void HPXQmciClusterSolver<qmci_integrator_type>::initialize(int dca_iteration_)
{
  DCA_LOG("HPXQmciClusterSolver initialize");
  profiler_type profiler(__FUNCTION__, "HPX-MC-Integration", __LINE__);

//  dca_iteration_ = dca_iteration_0;
  qmci_integrator_type::initialize(dca_iteration_);
  acc_finished = 0;
}

//----------------------------------------------------------------------------
template <class qmci_integrator_type>
void HPXQmciClusterSolver<qmci_integrator_type>::integrate()
{
  profiler_type profiler(__FUNCTION__, "HPX-MC-Integration", __LINE__);
  DCA_LOG("HPXQmciClusterSolver integrate " << nr_walkers << " " << nr_accumulators);
  //
  // Reset the promise on each iteration because we can only get the future 'once'
  accumulators_ready = hpx::lcos::local::promise<void>();
  accumulator_counter.clear();

  if (concurrency_.id() == concurrency_.first()) {
    std::cout << "HPX QMC integration has started: " << dca::util::print_time() << "\n"
              << std::endl;
  }

  {
    accumulators_finished = accumulators_ready.get_future();

    if (concurrency_.id() == concurrency_.first())
      thread_task_handler_.print();

    dca::profiling::WallTime start_time;

    // walker and accumulator tasks use a lot of temp stack space,
    // so use a large stack size executor

    // prepare an accumulator to go onto the queue
    for (int i = 0; i < 1/*nr_accumulators*/; ++i) {
        DCA_LOG("starting an async accumulator");
        auto init = hpx::async(large_stack_executor, [=] { return accumulator_initialize(i); });
        auto measure = init.then(large_stack_executor, [=](accumulator_future&& accum) {
          return this->accumulator_try_measure(accum.get());
        });
    }
    for (int i = 0; i < nr_walkers; ++i) {
        DCA_LOG("starting an async walker");
        auto init = hpx::async(large_stack_executor, [=] { return this->walker_initialize(i); });
        auto sweep = init.then(large_stack_executor, [=](walker_future&& finit) {
          return this->walker_sweep(finit.get());
        });
        auto done = sweep.then(large_stack_executor, [=](walker_future&& fsweep) {
          return this->walker_sweep_completed(fsweep.get());
        });
    }

    // wait until accumulators have done their work and complete is made ready
    accumulators_finished.get();

    dca::profiling::WallTime end_time;

    dca::profiling::Duration duration(end_time, start_time);
    total_time_ = duration.sec + 1.e-6 * duration.usec;
  }

  if (concurrency_.id() == concurrency_.first()) {
    std::cout << "HPX on-node integration has ended: " << dca::util::print_time()
              << "\n\nTotal number of measurements: " << parameters_.get_measurements()
              << "\nQMC-time\t" << total_time_ << std::endl;
  }

  qmci_integrator_type::accumulator_.finalize();
}

//----------------------------------------------------------------------------
template <class qmci_integrator_type>
template <typename dca_info_struct_t>
double HPXQmciClusterSolver<qmci_integrator_type>::finalize(dca_info_struct_t& dca_info_struct)
{
  profiler_type profiler(__FUNCTION__, "HPX-MC-Integration", __LINE__);
  DCA_LOG("HPXQmciClusterSolver finalize");

  if (dca_iteration_ == parameters_.get_dca_iterations() - 1)
    computeErrorBars();

  double L2_Sigma_difference = qmci_integrator_type::finalize(dca_info_struct);
  return L2_Sigma_difference;
}

//----------------------------------------------------------------------------
template <class qmci_integrator_type>
typename HPXQmciClusterSolver<qmci_integrator_type>::walker_ptr HPXQmciClusterSolver<
    qmci_integrator_type>::walker_initialize(int id)
{
  DCA_LOG("HPXQmciClusterSolver walker_initialize");

  if (id == 0) {
    if (concurrency_.id() == concurrency_.first())
      std::cout << "\n\t\t HPX QMCI starts\n\n";
  }

  const int rng_index = thread_task_handler_.walkerIDToRngIndex(id);
  DCA_LOG("Creating a walker");

  walker_ptr walker = std::make_shared<walker_type>(parameters_, data_, rng_vector[rng_index], id);

  DCA_LOG("Initializing walker");
  walker->initialize();

  {
    profiler_type profiler("thermalization", "HPX-MC-walker", __LINE__, id);
    DCA_LOG("Warm up walker " << walker->get_thread_id());
    warm_up(walker, id);
  }
  DCA_LOG("Warm up done : returning " << walker);
  return walker;
}

//----------------------------------------------------------------------------
template <class qmci_integrator_type>
typename HPXQmciClusterSolver<qmci_integrator_type>::walker_ptr HPXQmciClusterSolver<
    qmci_integrator_type>::walker_sweep(walker_ptr walker)
{
  DCA_LOG("HPXQmciClusterSolver walker_sweep");
  //
  auto id = walker->get_thread_id();
  //
  if (acc_finished < nr_accumulators) {
    profiler_type profiler("HPX-MC-walker updating", "HPX-MC-walker", __LINE__, id);
    DCA_LOG("Sweep (1) walker " << walker->get_thread_id());
    walker->doSweep();
  }

  return walker;
}

//----------------------------------------------------------------------------
template <class qmci_integrator_type>
typename HPXQmciClusterSolver<qmci_integrator_type>::walker_ptr HPXQmciClusterSolver<
    qmci_integrator_type>::walker_sweep_completed(walker_ptr walker)
{
  DCA_LOG("walker_sweep_completed " << walker->get_thread_id());
  //
  if (acc_finished < parameters_.get_measurements()) {
    accumulator_ptr accum = nullptr;
    {
      scoped_lock lock(accumulators_queue_mutex);
      if (!accumulators_queue.empty()) {
        accum = accumulators_queue.front();
        accumulators_queue.pop();
      }
    }
    if (accum) {
      DCA_LOG("walker " << walker->get_thread_id() << " popped an accumulator " << accum);
      accum->update_from(*walker);
      // Start an accumulator measure task
      auto measure =
          hpx::async(large_stack_executor, [=] { return this->accumulator_measure(accum, walker); });
      // when the measure task completes, it will restart the walker sweeping
      return nullptr;
    }
    else {
        throw std::runtime_error("This code is deprecated, probably");
    }
  }
  else {
      DCA_LOG("walker " << walker->get_thread_id() << " did not need to restart accumulator");
  }
  accumulators_ready.set_value();
  return walker;
}

//----------------------------------------------------------------------------
template <class qmci_integrator_type>
void HPXQmciClusterSolver<qmci_integrator_type>::warm_up(walker_ptr walker, int id)
{
  DCA_LOG("HPXQmciClusterSolver warm_up");

  if ((id == 0) && concurrency_.id() == concurrency_.first()) {
    std::cout << "\n\t\t HPX warm-up starts\n" << std::endl;
  }

  for (int i = 0; i < parameters_.get_warm_up_sweeps(); i++) {
    walker->doSweep();
    walker->updateShell(i, parameters_.get_warm_up_sweeps());
  }

  walker->is_thermalized() = true;

  if ((id == 0) && concurrency_.id() == concurrency_.first()) {
    std::cout << "\n\t\t HPX warm-up ends\n" << std::endl;
  }
}

//----------------------------------------------------------------------------
template <class qmci_integrator_type>
typename HPXQmciClusterSolver<qmci_integrator_type>::accumulator_ptr HPXQmciClusterSolver<
    qmci_integrator_type>::accumulator_initialize(int id)
{
  DCA_LOG("HPXQmciClusterSolver accumulator_initialize");

  accumulator_ptr accum = std::make_shared<HPX_accumulator_type>(parameters_, data_, id);
  accum->initialize(dca_iteration_);
  accumulator_counter[accum.get()] = 0;
  return accum;
}

//----------------------------------------------------------------------------
template <class qmci_integrator_type>
typename HPXQmciClusterSolver<qmci_integrator_type>::accumulator_ptr HPXQmciClusterSolver<
    qmci_integrator_type>::accumulator_try_measure(accumulator_ptr accum)
{
  walker_ptr walker = nullptr;
  {
    scoped_lock lock(walkers_queue_mutex);
    if (!walkers_queue.empty()) {
      walker = walkers_queue.front();
      walkers_queue.pop();
    }
  }
  if (walker) {
    DCA_LOG("Accumulator got a walker from the waiting queue");
    return accumulator_measure(accum, walker);
  }
  else {
    DCA_LOG("Accumulator found no walkers ready, ading itself to accumulator queue");
    {
      scoped_lock lock(accumulators_queue_mutex);
      accumulators_queue.push(accum);
    }
  }
  return nullptr;
}
//----------------------------------------------------------------------------
template <class qmci_integrator_type>
typename HPXQmciClusterSolver<qmci_integrator_type>::accumulator_ptr HPXQmciClusterSolver<
    qmci_integrator_type>::accumulator_measure(accumulator_ptr accum, walker_ptr walker)
{
  int id = accum->get_thread_id();
  int measurement_num = accumulator_counter[accum.get()];
  {
    profiler_type profiler("HPX-accumulator accumulating", "HPX-MC-accumulator", __LINE__, id);
    //if (id == 1) {
       walker->updateShell(measurement_num, parameters_.get_measurements());
    //}
    //
    DCA_LOG("Accumulator " << id << " calling measure");
    accum->measure();

    // if the accumulator has done all the measurements it needs, then we can quit
    if (++accumulator_counter[accum.get()] >= parameters_.get_measurements()) {
      DCA_LOG("Accumulator " << id << " calling sum_to : finished");
      scoped_lock lock(merge_mutex);
      accum->sum_to(accumulator_);
      acc_finished++;
      accumulators_ready.set_value();
    }
    else {
      DCA_LOG("Accumulator " << id << " going back to queue "
                     << accumulator_counter[accum.get()]);
      scoped_lock lock(accumulators_queue_mutex);
      accumulators_queue.push(accum);
    }
  }

  // we are finished with the walker, so restart the sweep
  DCA_LOG("Accumulator " << id << " restarting walker");
  auto sweep = hpx::async(large_stack_executor, [=] { return this->walker_sweep(walker); });
  auto done = sweep.then(large_stack_executor, [=](walker_future&& sweep) {
    walker_ptr completed_walker = sweep.get();
    return this->walker_sweep_completed(completed_walker);
  });

  return nullptr;
}

}  // namespace solver
}  // namespace phys
}  // namespace dca

#endif
