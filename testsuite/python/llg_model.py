#
# Copyright (C) 2023 The ESPResSo project
#
# This file is part of ESPResSo.
#
# ESPResSo is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ESPResSo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

import espressomd
import espressomd.observables
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import scipy.special as spcl

import unittest as ut
import unittest_decorators as utx
import tests_common

class Magnetodynamics(ut.TestCase):
    system = espressomd.System(box_l=[1.0, 1.0, 1.0])
    system.time_step = 0.001
    system.cell_system.skin = 1.3
    n_part = 4000

    np.random.seed(1)
    
    def set_stoner_wohlfarth_particle(self):
        n_part = 1
        self.system.part.add(type=n_part * [0],
                    pos=np.random.random((n_part, 3)) * self.system.box_l,
                    rotation=n_part * [(False, False, False)],
                    fix=n_part * [(True, True, True)],
                    director=n_part * [(0,1,0)],
                    dip=n_part * [(0,1,0)],
                    #dipm=n_part * [1.],
                    llg_model_params=n_part*[[
                        True,
                        2,
                        0.1,
                        70,
                        5e-5
                    ]]
                    )

    def set_langevin_particles(self):
        n_part = self.n_part
        self.system.part.add(type=n_part * [0],
                    pos=np.random.random((n_part, 3)) * self.system.box_l,
                    rotation=n_part * [(False, False, False)],
                    fix=n_part * [(True, True, True)],
                    llg_model_params=n_part*[[
                        True,
                        2,
                        0.1,
                        47.69599836457976,
                        0.00013526476494994482
                    ]])

        orientor_list = np.random.standard_normal((n_part, 3))
        orientor_list_normalized = np.array(
            [(x / np.linalg.norm(x, axis=0)) for x in orientor_list])
        dip_mom = orientor_list_normalized
        self.system.part.all().director = orientor_list_normalized
        self.system.part.all().dip = dip_mom
        self.system.part.all().gamma_mag = 3*[0.00017298778441412224]

    def tearDown(self):
        self.system.part.clear()
        self.system.thermostat.turn_off()
        self.system.integrator.set_vv()
        self.system.constraints.clear()
        self.system.time = 0.
        self.system.auto_update_accumulators.clear()

    def setUp(self):
        system = self.system
        system.cell_system.skin = 0.4
        system.periodicity = [False, False, False]
        system.time = 0.
    
    @utx.skipIfMissingFeatures(["MAGNETODYNAMICS_LLG_MODEL", "DIPOLES", "EXTERNAL_FORCES"])
    def test_stoner_wohlfarth(self):
        def stoner_wohlfarth(phi0, theta, h):
            """Stoner Wohlfarth minimizer"""
            """Author: Florian Bruckner"""
            def eta(phi, theta, h):
                '''
                eta = E / (2*Ku*V)
                phi     ... angle between external field and magnetization
                theta   ... angle between external field and anisotropy axis
                h       ... reduced external field = Hext/Hani
                '''
                return 0.25 - 0.25*np.cos(2*(phi-theta)) - h*np.cos(phi)

            def deta(phi, theta, h):
                return 0.5*np.sin(2*(phi-theta))+h*np.sin(phi)

            eps_phi = (phi0-theta)*1e-2
            # larger eps for larger H, best general choice 1e-2
            f  = lambda phi: eta (phi, theta, h)   # update,
            df = lambda phi: deta(phi, theta, h)   # jacobian, gradient vector

            sol = minimize(f, x0=phi0+eps_phi, jac=df, method="BFGS", tol=1e-15, options={'gtol':1e-15})['x'][0]
            return np.cos(sol)

        self.set_stoner_wohlfarth_particle()

        dip_obs = espressomd.observables.MagneticDipoleMoment(
            ids=self.system.part.all().id)
        self.system.thermostat.set_langevin(
            kT=0.,
            gamma=1.0,
            gamma_magnet=0.0001,
            seed=42
        )

        Hext = 0.5
        H_constraint = espressomd.constraints.HomogeneousMagneticField(H=[Hext,0,0])
        self.system.constraints.add(H_constraint)

        self.system.integrator.set_vv()

        self.system.integrator.run(steps=1000)
        dips = dip_obs.calculate()

        np.testing.assert_allclose(
            dips[0],
            stoner_wohlfarth(
                # initial angle between external field and magnetization
                phi0=np.deg2rad(90),

                # initial angle between external field and anisotropy axis
                theta=np.deg2rad(90),

                # reduced external field = Hext/Hani
                h=Hext/2
            ),
            atol=1e-5)
        self.tearDown()
    
    @utx.skipIfMissingFeatures(["MAGNETODYNAMICS_LLG_MODEL", "DIPOLES", "EXTERNAL_FORCES"])
    def test_langevin(self):
        """
        Check the alignment of the dipole with the field (1024 particles,
        non-interacting, mu^2=1, no PBC).
        """
        def gen_ani_integral():
            """langevin solution for fixed particle"""
            """Author: Andrey Kuznetsov"""
            z_int = lambda x,s,p,t: np.exp(-s*np.sin(t)**2) * np.cosh(x*np.cos(t)*np.cos(p)) * spcl.i0(x*np.sin(t)*np.sin(p))*np.sin(t)
            dz_int = lambda x,s,p,t: np.exp(-s*np.sin(t)**2) * (
                np.cosh(x*np.cos(t)*np.cos(p)) * spcl.i1(x*np.sin(t)*np.sin(p)) * np.sin(t) * np.sin(p) + 
                np.sinh(x*np.cos(t)*np.cos(p)) * spcl.i0(x*np.sin(t)*np.sin(p)) * np.cos(t) * np.cos(p)
            ) * np.sin(t)
            Z = lambda x,s,p: quad(lambda t: z_int(x,s,p,t),0,np.pi/2)[0]
            dZ = lambda x,s,p: quad(lambda t: dz_int(x,s,p,t),0,np.pi/2)[0]
            m = lambda x,s: quad(lambda p: dZ(x,s,p)/Z(x,s,p) * np.sin(p),0,np.pi/2)[0]
            return m
        
        self.set_langevin_particles()
        dip_obs = espressomd.observables.MagneticDipoleMoment(
            ids=self.system.part.all().id)

        # check auto-update accumulator
        acc = espressomd.accumulators.TimeSeries(obs=dip_obs)
        self.system.auto_update_accumulators.add(acc)

        self.system.thermostat.set_langevin(
            kT=0.2,
            gamma=1.0,
            gamma_magnet=0.00017298778441412224,
            seed=42
        )

        H_constraint = espressomd.constraints.HomogeneousMagneticField(H=[1,0,0])
        self.system.constraints.add(H_constraint)

        self.system.integrator.set_vv()
        self.system.integrator.run(steps=1000)
        time_series = acc.time_series()

        n_part = self.n_part
        avg_mx = np.average(time_series[-100:,0])/n_part
        xsi = 5
        sigma = 5
        np.testing.assert_allclose(avg_mx, gen_ani_integral()(xsi,sigma),
            atol=5e-3)
        self.tearDown()


if __name__ == "__main__":
    ut.main()
