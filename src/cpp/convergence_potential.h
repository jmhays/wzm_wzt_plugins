//
// Created by Jennifer Hays on 6/12/18.
//

#ifndef GROMACS_CONVERGENCE_POTENTIAL_H
#define GROMACS_CONVERGENCE_POTENTIAL_H

#include <array>
#include <iostream>
#include <mutex>
#include <vector>

#include "gmxapi/gromacsfwd.h"
#include "gmxapi/md/mdmodule.h"
#include "gmxapi/session.h"

#include "gromacs/restraint/restraintpotential.h"
#include "gromacs/utility/real.h"

#include "make_unique.h"
#include "restraint.h"
#include "sessionresources.h"

namespace plugin
{

struct convergence_input_param_type
{
  double alpha{0};
  double tolerance{0.5};
  double target{0};
  double sample_period{0};
  std::string logging_filename;
};

std::unique_ptr<convergence_input_param_type>
makeConvergenceParams(double alpha, double target, double tolerance,
                      double sample_period, std::string logging_filename);
//                   double sample_period)

class Convergence
{
public:
  using input_param_type = convergence_input_param_type;

  explicit Convergence(const input_param_type &params);

  Convergence(double alpha, double target, double tolerance, double sample_period,
              std::string filename);

  // If dispatching this virtual function is not fast enough, the compiler may
  // be able to better optimize a free function that receives the current
  // restraint as an argument.
  gmx::PotentialPointData calculate(gmx::Vector v, gmx::Vector v0,
                                    gmx_unused double t);

  void writeParameters(double t, const double R);

  // An update function to be called on the simulation master rank/thread
  // periodically by the Restraint framework.
  void callback(gmx::Vector v, gmx::Vector v0, double t,
                const Resources &resources);

  double getTime() { return time_; }

private:
  bool initialized_{false};
  double time_{0};

  double alpha_;
  double tolerance_;

  /// target distance
  double target_;

  // Sample interval
  double sample_period_;
  double startTime_{0};
  double nextSampleTime_{0};
  unsigned int currentSample_{0};

  std::string logging_filename_;
  std::unique_ptr<RAIIFile> logging_file_{nullptr};
  bool stop_not_called_{true};
};

// Just declare the template instantiation here for client code.
// We will explicitly instantiate a definition in the .cpp file where the
// input_param_type is defined.
extern template class RestraintModule<Restraint<Convergence>>;
} // end namespace plugin

#endif // GROMACS_CONVERGENCE_POTENTIAL_H
