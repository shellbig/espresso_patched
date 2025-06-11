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
inline Utils::Vector3d llg(Utils::Vector3d dip, Particle &p) {
  // calculating the change in magnetic momentum
  // via the Landau-Lifshitz-Gilbert equation
  auto const Galpha   = p.llg_model_params().Galpha;  // Gilbert damping
  auto const easy_axis= p.calc_director();

  // calculate the anisotropy field to later add to the effective field
  Utils::Vector3d const hani = p.llg_model_params().Hani * (dip*easy_axis) * easy_axis;
  
  p.dip_omega() = p.llg_model_params().gyromag / (1. + Galpha*Galpha) *
    (p.heff() + hani + Galpha * vector_product(dip, p.heff() + hani));

  return vector_product(p.dip_omega(),dip);
}

inline void propagate_dipu_particle(Particle &p,double time_step) {
  // updating the direction of the dipole moment
  p.dipu() += llg(p.dipu(), p)*time_step;
  p.dipu().normalize();
}

inline void propagate_dipu_particle_multi_step(Particle &p,double time_step) {
  auto magdt  = p.llg_model_params().magdt;   // magnetic time step
  Utils::Vector3d dip = p.dipu();
  
  // perform multiple steps for the magnetic problem
  // during one step of the mechanical problem
  double remaining_time = time_step;
  do {
    if (magdt < remaining_time) {
      remaining_time -= magdt;
    } else {
      magdt = remaining_time;
      remaining_time = 0.0;
    }
    // updating the direction of the dipol moment
    // Heun's method
    auto const dm_intermediate = llg(dip, p);
    auto const dip_intermediate = dip + magdt*dm_intermediate;
    auto const dm_final = (dm_intermediate + llg(dip_intermediate, p))/2;
    dip += dm_final*magdt;
    dip.normalize();
  } while (remaining_time > 0.0);

  p.dipu() = dip;

  // Utils::Vector3d const easy_axis= p.calc_director();
  // auto const hani = p.llg_model_params().Hani * (dip*easy_axis) * easy_axis;
  // p.heff() += hani;
  // anisotropy_energy = dip * hani;

  // reset the effective field
  p.heff() = {0.,0.,0.};
}

inline void apply_magnetic_torque(Particle &p, double time_step) {
  Utils::Vector3d const dip = p.dipu();
  auto const gyromag  = p.llg_model_params().gyromag; // gyromagnetic ratio

  // summing the field components that are constant during one mechanical step
  // dipole-dipole-interaction field, thermal field, external field, Barnett field
  p.heff() += p.dip_fld() + p.htherm()
    - vector_product(dip,convert_vector_body_to_space(p, p.omega()))
    * p.llg_model_params().Galpha/gyromag;
  // Einstein-de-Haas effect
  p.torque() += 1./gyromag * llg(dip, p) * p.dipm();
}
#endif // MAGNETODYNAMICS_LLG_MODEL
#endif // LLG_MODEL_INLINE_HPP