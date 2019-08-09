//
// Created by Jennifer Hays on 6/12/18.
//

#include "gmxapi/md/mdsignals.h"
#include "gmxapi/session.h"
#include "convergence_potential.h"

#include <cmath>

#include <vector>

namespace plugin {
std::unique_ptr<convergence_input_param_type>
makeConvergenceParams(double alpha, double target, double tolerance,
                     double sample_period, std::string logging_filename) {
  using gmx::compat::make_unique;
  auto params = make_unique<convergence_input_param_type>();
  params->alpha = alpha;
  params->tolerance = tolerance;
  params->target = target;
  params->sample_period = sample_period;
  params->logging_filename = logging_filename;

  return params;
};
Convergence::Convergence(double alpha, double target, double tolerance,
                       double sample_period, std::string logging_filename)
    : alpha_{alpha}, tolerance_{tolerance}, target_{target},
      sample_period_{sample_period}, logging_filename_{logging_filename} {};

Convergence::Convergence(const input_param_type &params)
    : Convergence(params.alpha, params.target, params.tolerance,
                 params.sample_period, params.logging_filename) {}

void Convergence::writeParameters(double t, const double R) {
  if (logging_file_) {
    fprintf(logging_file_->fh(), "%f\t%f\t%f\t%f\n", t, R, target_, alpha_);
    fflush(logging_file_->fh());
  }
}

void Convergence::callback(gmx::Vector v, gmx::Vector v0, double t,
                          const Resources &resources) {

  // Update distance
  auto rdiff = v - v0;
  const auto Rsquared = dot(rdiff, rdiff);
  const auto R = sqrt(Rsquared);

  bool converged = std::abs(R - target_) < tolerance_;
  //        printf("===SUMMARY===\nR = %f\ntarget_ = %f\ntolerance =
  //        %f\nconverged = %d\n======\n",
  //               R, target_, tolerance_, converged);
  // Open logs at the beginning of the simulation
  if (!initialized_) {
    startTime_ = t;
    nextSampleTime_ = startTime_ + sample_period_;
    //            printf("startTime_ = %f, nextSampleTime_ = %f, sample_period_ =
    //            %f\n",
    //                   startTime_, nextSampleTime_, sample_period_);
    logging_file_ =
        gmx::compat::make_unique<RAIIFile>(logging_filename_.c_str(), "w");
    if (logging_file_) {
      fprintf(logging_file_->fh(), "time\tR\ttarget\talpha\n");
      writeParameters(t, R);
    }
    initialized_ = true;
    //            printf("initialized_ = %d\n", initialized_);
  }

  // If the simulation has not converged, keep running and log
  if (!converged && (t >= nextSampleTime_)) {
    writeParameters(t, R);
    currentSample_++;
    nextSampleTime_ = (currentSample_ + 1) * sample_period_ + startTime_;
  }

  if (converged) {
    if (stop_not_called_) {
      stop_not_called_ = false;
      writeParameters(t, R);
      //                fprintf(logging_file_->fh(), "Simulation converged at t
      //                == %f", t);
      logging_file_->close();
      logging_file_.reset(nullptr);
      resources.getHandle().stop();
    } else {
      // Do nothing until all stops have been called
    }
  }
}

gmx::PotentialPointData Convergence::calculate(gmx::Vector v, gmx::Vector v0,
                                              gmx_unused double t) {
  // Our convention is to calculate the force that will be applied to v.
  // An equal and opposite force is applied to v0.
  time_ = t;
  auto rdiff = v - v0;
  const auto Rsquared = dot(rdiff, rdiff);
  const auto R = sqrt(Rsquared);
  // TODO: find appropriate math header and namespace

  // In White & Voth, the additional energy is alpha * f(r)/favg

  gmx::PotentialPointData output;

  output.energy = alpha_ / target_ * double(R);
  // Direction of force is ill-defined when v == v0
  if (R != 0 && R != target_) {
    if (R > target_)
      output.force = real(-(alpha_ / target_ / double(R))) * rdiff;
    else
      output.force = real((alpha_ / target_ / double(R))) * rdiff;
  }

  //    history.emplace_back(magnitude - R0);
  return output;
}

// Explicitly instantiate a definition.
template class ::plugin::RestraintModule<Restraint<Convergence>>;
} // end namespace plugin
