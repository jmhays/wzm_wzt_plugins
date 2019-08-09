//
// Created by Jennifer Hays on 3/27/2018
//

/*! \file
 * \brief Code to implement the potential declared in training_potential.h
 *
 * This file currently contains boilerplate that will not be necessary in future
 * gmxapi releases, as well as additional code used in implementing the BRER
 * workflow.
 *
 */

#include <cassert>

#include "training_potential.h"
#include <cmath>

#include <vector>

#include "gmxapi/context.h"
#include "gmxapi/md/mdsignals.h"
#include "gmxapi/session.h"

#include "sessionresources.h"
#include <fstream>

namespace plugin {

Training::Training(double alpha, double alpha_prev, double alpha_max, double mean,
           double variance, double A, double tau, double g, double gsqrsum,
           double eta, bool converged, double tolerance, double target,
           unsigned int n_samples, std::string parameter_filename)
    : alpha_{alpha}, alpha_prev_{alpha_prev},
      alpha_max_{alpha_max}, mean_{mean}, variance_{variance}, A_{A}, tau_{tau},
      g_{g}, gsqrsum_{gsqrsum}, eta_{eta}, converged_{converged},
      tolerance_{tolerance}, target_{target}, n_samples_{n_samples},
      samplePeriod_{tau / n_samples}, parameter_filename_{parameter_filename} {};

Training::Training(const input_param_type &params)
    : Training(params.alpha, params.alpha_prev, params.alpha_max, params.mean,
           params.variance, params.A, params.tau, params.g, params.gsqrsum,
           params.eta, params.converged, params.tolerance, params.target,
           params.n_samples, params.parameter_filename) {}

void Training::writeparameters(double t, const double R) {
  if (parameter_file_) {
    fprintf(parameter_file_->fh(), "%f\t%f\t%f\t%d\t%f\t%f\t%f\t%f\n", t, R,
            target_, converged_, alpha_, alpha_max_, g_, eta_);
    fflush(parameter_file_->fh());
  }
}

void Training::callback(gmx::Vector v, gmx::Vector v0, double t,
                    const Resources &resources) {

  if (!converged_) {
    auto rdiff = v - v0;
    const auto Rsquared = dot(rdiff, rdiff);
    const auto R = sqrt(Rsquared);
    if (!initialized_) {
      nextSampleTime_ = t + samplePeriod_;
      windowStartTime_ = t;
      nextUpdateTime_ = t + tau_;

      mean_ = R;

      // We expect that the amount of energy we need to add to the system will
      // be APPROXIMATELY proportional to the difference in R and the target.
      //      A_ *= std::fabs(target_ - R);
      // Similarly, the tolerance should be adjusted so that it is essentially a
      // percentage of the maximum energy input
      tolerance_ *= A_;

      parameter_file_ =
          gmx::compat::make_unique<RAIIFile>(parameter_filename_.c_str(), "w");
      if (parameter_file_) {
        fprintf(parameter_file_->fh(),
                "time\tR\ttarget\tconverged\talpha\talpha_max\tg\teta\n");
        writeparameters(t, R);
      }
      initialized_ = true;
    }

    if (t >= nextSampleTime_) {
      // update mean and variance
      int j = currentSample_ + 1;
      auto difference = (R - mean_);
      auto diffsqr = difference * difference;
      variance_ = variance_ + (j - 1) * diffsqr / j;
      mean_ = mean_ + difference / j;
      currentSample_++;
      nextSampleTime_ = (currentSample_ + 1) * samplePeriod_ + windowStartTime_;
    }

    if (t >= nextUpdateTime_) {
      assert(currentSample_ == n_samples_);
      g_ = (1 - mean_ / target_) * variance_;
      gsqrsum_ = gsqrsum_ + g_ * g_;
      eta_ = A_ / sqrt(gsqrsum_);
      alpha_prev_ = alpha_;
      alpha_ = alpha_prev_ - eta_ * g_;
      if (fabs(alpha_) > alpha_max_)
        alpha_max_ = fabs(alpha_);

      // Reset mean and variance
      mean_ = R;
      variance_ = 0;
      windowStartTime_ = t;
      nextUpdateTime_ = n_samples_ * samplePeriod_ + windowStartTime_;

      // Reset sample buffering.
      currentSample_ = 0;
      // Reset sample times.
      nextSampleTime_ = t + samplePeriod_;
      if (parameter_file_) {
        writeparameters(t, R);
      }

      if (fabs(alpha_ - alpha_prev_) < tolerance_) {
        converged_ = TRUE;
        if (parameter_file_) {
          writeparameters(t, R);
        }
        // Release filehandle and close file.
        parameter_file_->close();
        parameter_file_.reset(nullptr);
        // Issue stop signal exactly once.
        resources.getHandle().stop();
      }
    }
  } else {
    // Do nothing after convergence but wait for the simulation to end.
  }
}

gmx::PotentialPointData Training::calculate(gmx::Vector v, gmx::Vector v0,
                                        gmx_unused double t) {
  // Our convention is to calculate the force that will be applied to v.
  // An equal and opposite force is applied to v0.
  auto rdiff = v - v0;
  const auto Rsquared = dot(rdiff, rdiff);
  const auto R = sqrt(Rsquared);
  // TODO: find appropriate math header and namespace

  // In White & Voth, the additional energy is alpha * f(r)/favg

  gmx::PotentialPointData output;

  output.energy = alpha_ * double(R) / target_;
  // Direction of force is ill-defined when v == v0
  if (R != 0) {
    // For harmonic: output.force = k * (double(R0)/R - 1.0)*rdiff;
    // For Training: outpu.force = - alpha/target * (unit vector in direction v-v0).
    output.force = real(-(alpha_ / target_ / double(R))) *
                   rdiff; // Why is there a double cast here?
  }

  //    history.emplace_back(magnitude - R0);
  return output;
}

std::unique_ptr<training_input_param_type>
makeTrainingParams(double A, double tau, double tolerance, double target,
               unsigned int n_samples, std::string parameter_filename) {
  using gmx::compat::make_unique;
  auto params = make_unique<training_input_param_type>();
  params->A = A;
  params->tau = tau;
  params->tolerance = tolerance;
  params->target = target;
  params->n_samples = n_samples;
  params->parameter_filename = parameter_filename;
  return params;
};

// Explicitly instantiate a definition.
template class ::plugin::RestraintModule<Restraint<Training>>;
} // end namespace plugin
