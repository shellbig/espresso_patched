/*
 * Copyright (C) 2010-2022 The ESPResSo project
 *
 * This file is part of ESPResSo.
 *
 * ESPResSo is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ESPResSo is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef CONSTRAINTS_ALTERNATINGMAGNETICFIELD_HPP
#define CONSTRAINTS_ALTERNATINGMAGNETICFIELD_HPP

#include "Constraint.hpp"
#include "Observable_stat.hpp"
#include "Particle.hpp"

#include <utils/Vector.hpp>

namespace Constraints {

class AlternatingMagneticField : public Constraint {
public:
  AlternatingMagneticField() 
  : m_amplitude({0., 0., 0.}),
    m_omega(0.),
    m_phase(0.) {}

  void set_H0(Utils::Vector3d const &H0) { m_amplitude = H0; }
  void set_omega(double const &w) { m_omega = w; }
  void set_phase(double const &w) { m_phase = w; }


  Utils::Vector3d const &H0() const { return m_amplitude; }
  double const &omega() const { return m_omega; }
  double const &phase() const { return m_phase; }

  void add_energy(const Particle &p, const Utils::Vector3d &, double,
                  Observable_stat &energy) const override;

  Utils::Vector3d add_magnetic_field(const Particle &p,
                  const Utils::Vector3d &, double t) const;

  ParticleForce force(const Particle &p, const Utils::Vector3d &,
                      double) override;

  bool fits_in_box(Utils::Vector3d const &) const override { return true; }

private:
  Utils::Vector3d m_amplitude;
  double m_omega;
  double m_phase;
};

} /* namespace Constraints */

#endif
