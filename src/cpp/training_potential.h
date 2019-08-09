//
// Created by Jennifer Hays on 3/28/2018
//

#ifndef GROMACS_TRAINING_POTENTIAL_H
#define GROMACS_TRAINING_POTENTIAL_H

/*! \file
 * \brief Provide EBMetaD MD potential for GROMACS plugin.
 */
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

struct training_input_param_type
{
  /// learned coupling constant
  double alpha{0};
  double alpha_prev{0};
  double alpha_max{0};

  /// keep track of mean and variance
  double mean{0};
  double variance{0};

  /// parameters for training coupling constant (Adagrad)
  double A{0};
  double tau{0};
  double g{0};
  double gsqrsum{0};
  double eta{0};
  bool converged{0};
  double tolerance{0.05};

  /// target distance
  double target{0};

  /// Number of samples to store during each tau window.
  unsigned int n_samples{0};
  unsigned int currentSample{0};
  double samplePeriod{0};
  double windowStartTime{0};

  std::string parameter_filename;
};

// \todo We should be able to automate a lot of the parameter setting stuff
// by having the developer specify a map of parameter names and the
// corresponding type, but that could get tricky. The statically compiled fast
// parameter structure would be generated with a recursive variadic template the
// way a tuple is. ref
// https://eli.thegreenplace.net/2014/variadic-templates-in-c/

std::unique_ptr<training_input_param_type>
makeTrainingParams(double A, double tau, double tolerance, double target,
                   unsigned int n_samples, std::string parameter_filename);
//                   double samplePeriod)

class Training
{
public:
  using input_param_type = training_input_param_type;

  //        EnsembleHarmonic();

  explicit Training(const input_param_type &params);

  Training(double alpha, double alpha_prev, double alpha_max, double mean,
           double variance, double A, double tau, double g, double gsqrsum,
           double eta, bool converged, double tolerance, double target,
           unsigned int n_samples, std::string parameter_filename);
  // If dispatching this virtual function is not fast enough, the compiler may
  // be able to better optimize a free function that receives the current
  // restraint as an argument.

  gmx::PotentialPointData calculate(gmx::Vector v, gmx::Vector v0,
                                    gmx_unused double t);

  void writeparameters(double t, const double R);

  // An update function to be called on the simulation master rank/thread
  // periodically by the Restraint framework.
  void callback(gmx::Vector v, gmx::Vector v0, double t,
                const Resources &resources);

  double getAlphaMax() { return alpha_max_; }
  double getTarget() { return target_; }

private:
  bool initialized_{false};

  /// learned coupling constant
  double alpha_;
  double alpha_prev_;
  double alpha_max_;

  /// keep track of mean and variance
  double mean_;
  double variance_;

  /// parameters for training coupling constant (Adagrad)
  double A_;
  double tau_;
  double g_;
  double gsqrsum_;
  double eta_;
  bool converged_;
  double tolerance_;

  /// target distance
  double target_;

  // Sampling parameters determined by the user
  unsigned int n_samples_;
  double samplePeriod_;

  unsigned int currentSample_{0};
  // Sampling parameters that are dependent on t and thus set upon
  // initialization of the plugin For now, since we don't have access to t,
  // we'll set them all to zero.
  double nextSampleTime_{0};
  double windowStartTime_{0};
  double nextUpdateTime_{0};

  std::string parameter_filename_;
  std::unique_ptr<RAIIFile> parameter_file_{nullptr};
};

// Just declare the template instantiation here for client code.
// We will explicitly instantiate a definition in the .cpp file where the
// input_param_type is defined.
extern template class RestraintModule<Restraint<Training>>;

} // end namespace plugin

#endif // GROMACS_TRAINING_POTENTIAL_H
