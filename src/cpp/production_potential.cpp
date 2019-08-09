#include "gmxapi/md/mdsignals.h"
#include "gmxapi/session.h"
#include "production_potential.h"

#include <cmath>

#include <vector>

namespace plugin {
std::unique_ptr<production_input_param_type>
makeProductionParams(double alpha, double target, double sample_period,
                 std::string logging_filename) {
  using gmx::compat::make_unique;
  auto params = make_unique<production_input_param_type>();
  params->alpha = alpha;
  params->target = target;
  params->sample_period = sample_period;
  params->logging_filename = logging_filename;
  return params;
};
Production::Production(double alpha, double target, double sample_period,
               std::string logging_filename)
    : alpha_{alpha}, target_{target}, sample_period_{sample_period},
      logging_filename_{logging_filename} {};

Production::Production(const input_param_type &params)
    : Production(params.alpha, params.target, params.sample_period,
             params.logging_filename) {}

void Production::writeparameters(double t, const double R) {
  if (logging_file_) {
    fprintf(logging_file_->fh(), "%f\t%f\t%f\t%f\n", t, R, target_, alpha_);
    fflush(logging_file_->fh());
  }
}

void Production::callback(gmx::Vector v, gmx::Vector v0, double t,
                      const Resources &resources) {

  // Update distance
  auto rdiff = v - v0;
  const auto Rsquared = dot(rdiff, rdiff);
  const auto R = sqrt(Rsquared);

  if (!initialized_) {
    startTime_ = t;
    nextSampleTime_ = startTime_ + sample_period_;
    logging_file_ =
        gmx::compat::make_unique<RAIIFile>(logging_filename_.c_str(), "w");
    if (logging_file_) {
      fprintf(logging_file_->fh(), "time\tR\ttarget\talpha\n");
      writeparameters(t, R);
    }
    initialized_ = true;
  }

  // If the simulation has not converged, keep running and log
  if (t >= nextSampleTime_) {
    writeparameters(t, R);
    currentSample_++;
    nextSampleTime_ = (currentSample_ + 1) * sample_period_ + startTime_;
  }
}

gmx::PotentialPointData Production::calculate(gmx::Vector v, gmx::Vector v0,
                                          gmx_unused double t) {
  // Our convention is to calculate the force that will be applied to v.
  // An equal and opposite force is applied to v0.
  auto rdiff = v - v0;
  const auto Rsquared = dot(rdiff, rdiff);
  const auto R = sqrt(Rsquared);

  gmx::PotentialPointData output;

  output.energy = real(alpha_ * R / target_);
  // Direction of force is ill-defined when v == v0
  if (R != 0 && R != target_) {
    if (R > target_)
      output.force = real(-(alpha_ / target_ / double(R))) * rdiff;
    else
      output.force = real((alpha_ / target_ / double(R))) * rdiff;
  }

  return output;
}

template class ::plugin::RestraintModule<Restraint<Production>>;
} // namespace plugin
