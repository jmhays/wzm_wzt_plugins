#ifndef GROMACS_PORDUCTION_POTENTIAL_H
#define GROMACS_PRODUCTION_POTENTIAL_H

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

struct production_input_param_type
{
  double alpha{0};
  double target{0};
  double sample_period{0};
  std::string logging_filename{""};
};

std::unique_ptr<production_input_param_type>
makeProductionParams(double alpha, double target, double sample_period,
                     std::string logging_filename);

class Production
{
public:
  using input_param_type = production_input_param_type;

  explicit Production(const input_param_type &params);

  Production(double alpha, double target, double sample_period,
             std::string filename);

  gmx::PotentialPointData calculate(gmx::Vector v, gmx::Vector v0,
                                    gmx_unused double t);

  void writeparameters(double t, const double R);

  void callback(gmx::Vector v, gmx::Vector v0, double t,
                const Resources &resources);

private:
  bool initialized_{FALSE};
  double alpha_;

  /// target distance
  double target_;

  // Sample interval
  double sample_period_;
  double startTime_{0};
  double nextSampleTime_{0};
  unsigned int currentSample_{0};

  std::string logging_filename_;
  std::unique_ptr<RAIIFile> logging_file_{nullptr};
};

extern template class RestraintModule<Restraint<Production>>;
} // namespace plugin

#endif // GROMACS_PRODUCTION_POTENTIAL_H
