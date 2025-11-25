#ifndef LLG_MODEL_INLINE_HPP
#define LLG_MODEL_INLINE_HPP

#include "config/config.hpp"

#ifdef MAGNETODYNAMICS_LLG_MODEL

#include "Particle.hpp"
#include "random.hpp"
#include "integrate.hpp"
#include "rotation.hpp"
#include "thermostat.hpp"

#include <utils/Vector.hpp>

#include <cmath>
#include <iostream>

// all calculations are performed in the space-fixed/lab frame
// update the angular velocity of the particle dipole
inline Utils::Vector3d propagate_dip_omega(Particle &p, Utils::Vector3d dip) {
  auto const Galpha = p.llg_model_params().Galpha;  // Gilbert damping
  return p.llg_model_params().gyromag / (1. + Galpha*Galpha) *
    (p.heff() + Galpha * vector_product(dip, p.heff()));
}

// calculating the change in magnetic momentum
// via the Landau-Lifshitz-Gilbert equation
// the LLG as a result of the cross product of angular velocity and dipole moment
inline Utils::Vector3d llg(Particle &p, Utils::Vector3d dip) {
  return vector_product(propagate_dip_omega(p, dip),dip);
}

// single step propagation of the dipole
inline void propagate_dipu_particle(Particle &p,double time_step) {
  // updating the direction of the dipole moment
  p.dipu() += llg(p, p.dipu())*time_step;
  p.dipu().normalize();
}

// multi step propagation of the dipole
inline void propagate_dipu_particle_multi_step(Particle &p,double time_step) {
  auto magdt  = p.llg_model_params().magdt;   // magnetic time step
  Utils::Vector3d dip = p.dipu();
  Utils::Vector3d dm_intermediate = {0.0,0.0,0.0};
  Utils::Vector3d dip_intermediate;
  
  // perform multiple steps for the magnetic problem
  // during one step of the mechanical problem
  double remaining_time = time_step;
  do {
    if (magdt < remaining_time - 1e-12) {
      remaining_time -= magdt;
    } else {
      magdt = remaining_time;
      remaining_time = 0.0;
    }
    // updating the direction of the dipol moment
    // Heun's method
    dm_intermediate = llg(p, dip);
    dip_intermediate = dip + magdt*dm_intermediate;
    dip += (dm_intermediate + llg(p, dip_intermediate))*magdt/2;
    dip.normalize();
  } while (remaining_time > 0.0);

  p.dipu() = dip;
}

inline void update_mag_field_and_edh_torque(ParticleRange const &particles) {
  Utils::Vector3d dip;
  Utils::Vector3d easy_axis;
  for (auto &p : particles) {
    dip = p.dipu();
    easy_axis = p.calc_director();

    // the external fields is already added in the Constraints
    // thermal, crystalline anisotropy, and Barnett field
    p.heff() += p.htherm() + p.llg_model_params().Hani * (dip*easy_axis) * easy_axis
      - vector_product(dip,convert_vector_body_to_space(p, p.omega()))
      * p.llg_model_params().Galpha/p.llg_model_params().gyromag;
    
#ifdef DIPOLE_FIELD_TRACKING
    // dipolar interaction field
    p.heff() += p.dip_fld();
#endif // DIPOLE_FIELD_TRACKING
    // TODO add the field energy terms to the system energy

    // Einstein-de-Haas effect
    p.torque() -= 1./p.llg_model_params().gyromag * llg(p, dip) * p.dipm();
    // TODO check again (equation, sign)!
  }
}
#endif // MAGNETODYNAMICS_LLG_MODEL
#endif // LLG_MODEL_INLINE_HPP