// Copyright (C) 2018 ETH Zurich
// Copyright (C) 2018 UT-Battelle, LLC
// All rights reserved.
//
// See LICENSE for terms of usage.
// See CITATION.md for citation guidelines, if DCA++ is used for scientific publications.
//
// Author: Raffaele Solca' (rasolca@itp.phys.ethz.ch)
//
// This file provides cuda related utilities to cudaStreams.

#ifndef DCA_LINALG_UTIL_STREAM_FUNCTIONS_HPP
#define DCA_LINALG_UTIL_STREAM_FUNCTIONS_HPP

#include <cuda_runtime.h>
#include "dca/linalg/util/def.hpp"
#include "dca/linalg/util/stream_container.hpp"

namespace dca {
namespace linalg {
namespace util {
// dca::linalg::util::

inline StreamContainer<DCA_MAX_THREADS, DCA_STREAMS_PER_THREAD>& getStreamContainer() {
  static StreamContainer<DCA_MAX_THREADS, DCA_STREAMS_PER_THREAD> stream_container;
  return stream_container;
}

// Preconditions: 0 <= thread_id < DCA_MAX_THREADS,
//                0 <= stream_id < DCA_STREAMS_PER_THREADS.
inline cudaStream_t getStream(int thread_id, int stream_id) {
  return getStreamContainer()(thread_id, stream_id);
}

// Preconditions: 0 <= thread_id < DCA_MAX_THREADS,
//                0 <= stream_id < DCA_STREAMS_PER_THREADS.
inline void syncStream(int thread_id, int stream_id) {
  getStreamContainer().sync(thread_id, stream_id);
}

}  // util
}  // linalg
}  // dca

#endif  // DCA_LINALG_UTIL_STREAM_FUNCTIONS_HPP
