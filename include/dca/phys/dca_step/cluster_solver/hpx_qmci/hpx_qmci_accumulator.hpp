// Copyright (C) 2009-2016 ETH Zurich
// Copyright (C) 2007?-2016 Center for Nanophase Materials Sciences, ORNL
// All rights reserved.
//
// See LICENSE.txt for terms of usage.
// See CITATION.txt for citation guidelines if you use this code for scientific publications.
//
// Author: Peter Staar (taa@zurich.ibm.com)
//         Raffaele Solca' (rasolca@itp.phys.ethz.ch)
//         John Biddiscombe (biddisco@cscs.ch)
//
// An HPX implementation of an MC accumulator independent of the MC method.

#ifndef DCA_QMCI_HPX_JACKET_FOR_MC_ACCUMULATION_H
#define DCA_QMCI_HPX_JACKET_FOR_MC_ACCUMULATION_H

#include <hpx/hpx.hpp>
#include <hpx/util/logging.hpp>

#include <dca/config/hpx_defines.hpp>

#include <queue>
#include <iostream>

namespace dca {
namespace phys {
namespace solver {
namespace hpxqmci {

template <class qmci_accumulator_type>
class hpx_qmci_accumulator : protected qmci_accumulator_type {
  typedef typename qmci_accumulator_type::ParametersType parameters_type;
  using Data = typename qmci_accumulator_type::DataType;

  typedef hpx_qmci_accumulator<qmci_accumulator_type> this_type;

public:
  hpx_qmci_accumulator(parameters_type& parameters_ref, Data& data_ref, int id);

  ~hpx_qmci_accumulator();

  using qmci_accumulator_type::initialize;
  using qmci_accumulator_type::finalize;
  // using qmci_accumulator_type::to_JSON;
  using qmci_accumulator_type::get_configuration;

  inline int get_thread_id() const { return thread_id; }
  template <typename walker_type>
  void update_from(walker_type& walker);

  void measure();

  // Sums all accumulated objects of this accumulator to the equivalent objects of the 'other'
  // accumulator.
  void sum_to(qmci_accumulator_type& other);

public:
  using qmci_accumulator_type::get_number_of_measurements;
protected:
  using qmci_accumulator_type::get_Gflop;

private:
  using qmci_accumulator_type::parameters_;
  using qmci_accumulator_type::data_;

  int thread_id;
};

//----------------------------------------------------------------------------
template <class qmci_accumulator_type>
hpx_qmci_accumulator<qmci_accumulator_type>::hpx_qmci_accumulator(
  parameters_type& parameters_ref,
  Data& data_ref, int id)
  : qmci_accumulator_type(parameters_ref, data_ref, id), thread_id(id)
{
  DCA_LOG("hpx_qmci_accumulator constructor");
}

//----------------------------------------------------------------------------
template <class qmci_accumulator_type>
hpx_qmci_accumulator<qmci_accumulator_type>::~hpx_qmci_accumulator()
{
  DCA_LOG("hpx_qmci_accumulator destructor");
}

//----------------------------------------------------------------------------
template <class qmci_accumulator_type>
template <typename walker_type>
void hpx_qmci_accumulator<qmci_accumulator_type>::update_from(walker_type& walker) {
  {
    DCA_LOG("hpx_qmci_accumulator update_from");
    qmci_accumulator_type::updateFrom(walker);
  }
}

//----------------------------------------------------------------------------
template <class qmci_accumulator_type>
void hpx_qmci_accumulator<qmci_accumulator_type>::measure()
{
  DCA_LOG("hpx_qmci_accumulator measure");
  qmci_accumulator_type::measure();
}

//----------------------------------------------------------------------------
template <class qmci_accumulator_type>
void hpx_qmci_accumulator<qmci_accumulator_type>::sum_to(qmci_accumulator_type& other)
{
  DCA_LOG("hpx_qmci_accumulator sum_to");
  qmci_accumulator_type::sumTo(other);
}

}}}}

#endif  // DCA_QMCI_HPX_JACKET_FOR_MC_ACCUMULATION_H
