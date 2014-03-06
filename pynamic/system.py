__author__ = 'nmearl'

import numpy as np
import pylab
from astropy.constants.si import G, M_sun, R_sun, sigma_sb
from astropy import units as unit
from numba import autojit, vectorize, guvectorize, float64
import time
import math
from numpy.polynomial.legendre import legval
from numpy.linalg import norm

# Define some global variables
sigma = sigma_sb.value  # sb constant


class System():
    def __init__(self):
        self.all_bodies = []
        self.mu = 0.0
        self.semimajor_axis= 0.0
        self.com = np.zeros(3)  # * unit.meter
        self.total_mass = 0.0
        self.total_luminosity = 0.0
        self.total_flux = 0.0
        self.luminosity_history = []

    def add_body(self, mass, radius, temperature, a, ecc, inc, arg_peri, right_asc, true_anomaly):
        # Add star to list of all bodies
        new_body = Body(mass, radius, temperature, a, ecc, inc, arg_peri, right_asc, true_anomaly)
        self.all_bodies.append(new_body)

        # Get new total luminosity of the system
        self.total_luminosity = np.sum(b.luminosity for b in self.all_bodies)

        # Get total flux of the system
        self.total_flux = np.sum(b.flux for b in self.all_bodies)

        # Get new center of mass
        self.com = 1.0/np.sum(body.mass for body in self.all_bodies) * \
                   np.sum(body.mass * body.position for body in self.all_bodies)

        # Calculate total mass
        self.tot_mass = np.sum(b.mass for b in self.all_bodies)

        # self.all_bodies[0].position = +self.com

        # # Calculate Jacobian coordinates
        # for i in range(1, len(self.all_bodies)):
        #     body = self.all_bodies[i]
        #     body.jposition = body.position - self.com - (1.0/np.sum(self.all_bodies[j].mass for j in range(i))) * \
        #         np.sum(self.all_bodies[k].mass * self.all_bodies[k].position for k in range(i))\

    def prepare(self):
        pass


    def run(self, dt, nsteps):
        # Run loop
        for i in range(nsteps):
            # Calculate force and acceleration on body
            self.gforce()
            for body in self.all_bodies:
                # Save the position for plotting purposes
                body.save_position()

                # Advance the position
                body.position = body.position + body.velocity * dt + 0.5 * body.acceleration * dt ** 2
                # body.jposition = body.jposition + body.velocity * dt + 0.5 * body.acceleration * dt ** 2

            # Sort bodies by closest to furthest
            xvals = np.array([body.position[0] for body in self.all_bodies])
            sorted_bodies = np.array(self.all_bodies)[np.argsort(xvals)]
            sorted_bodies = np.array([body.pickleable() for body in sorted_bodies], dtype='float64')
            # sorted_bodies = sorted_bodies.reshape(len(self.all_bodies), 6)

            # Get the total flux at this time
            dS = eclipse(sorted_bodies)[0]

            current_luminosity = self.total_luminosity * (1.0 - dS / self.total_flux)
            self.luminosity_history.append(current_luminosity)

            # Calculate new forces
            self.gforce()
            for body in self.all_bodies:
                # Advance the velocity
                body.velocity += 0.5 * (body.acceleration + body.last_acceleration) * dt

        for body in self.all_bodies:
            x, y, z = zip(*body.position_history)
            pylab.plot(x, y, label=body.mass)
            pylab.plot(x[0], y[0], 'o')
            pylab.legend(loc=0)
        pylab.show()

        for body in self.all_bodies:
            x, y, z = zip(*body.position_history)
            # pylab.plot(z, y, '--')
            pylab.plot(z, y)
        pylab.show()

        pylab.plot(self.luminosity_history)
        pylab.show()

    def gforce_old(self):
        for body in self.all_bodies:
            body.force = np.zeros(3)  # * unit.kg * unit.meter / unit.second**2
            body.last_acceleration = body.acceleration

            for other_body in self.all_bodies:
                if other_body == body:
                    continue

                rvec = body.position - other_body.position
                distance = np.linalg.norm(body.position - other_body.position)  # * unit.meter
                body.force += - G.value * body.mass * other_body.mass * rvec / distance ** 3

            body.acceleration = body.force / body.mass  # - G * other_body.mass * rvec / np.abs(rvec)

    def gforce(self):
        all_bodies = self.all_bodies
        tot_mass = self.tot_mass
        for i in range(1, len(all_bodies)):
            body = all_bodies[i]
            body.last_acceleration = body.acceleration

            # r = body.position - (1/np.sum(all_bodies[j].mass for j in range(i))) * \
            #     np.sum(all_bodies[k].mass * all_bodies[k].position for k in range(i))

            r = [b.position for b in all_bodies]
            mu = [b.mass/tot_mass for b in all_bodies]
            rij = [mu[m-1] * r[m-1] + r[m] if m > 0 else np.zeros(3) for m in range(len(all_bodies))]

            # rjk = mu[i-2] * r[i-2] + r[i-1]
            # rij = mu[i-1] * r[i-1] + r[i]
            body.acceleration = -G.value * tot_mass * rij[i] / ((1.0 - mu[i]) * norm(rij[i])**3)

            if i == 1 and len(all_bodies) > 2:
                body.acceleration -= G.value * tot_mass * mu[i+1] * (r[i] - rij[i+1]) / norm(r[i] - rij[i+1])**3
                body.acceleration -= G.value * tot_mass * mu[i+1] * rij[i+1] / norm(rij[i+1])**3
            else:
                body.acceleration -= G.value * tot_mass * mu[i-1] * (rij[i] - rij[i-1]) / norm(rij[i] - rij[i-1])**3

    def gforce_new(self):
        all_bodies = self.all_bodies
        for i in range(1, len(all_bodies)):
            body = all_bodies[i]
            body.last_acceleration = body.acceleration
            body.acceleration = np.zeros(3)

            all_mass = np.array(body.mass for body in all_bodies)
            all_position = np.array(body.position for body in all_bodies)

            # Get mass interior to the body in question
            int_mass = np.sum(all_bodies[j].mass for j in range(len(all_bodies)) if j < i)

            # Calculate the jacobian postion of the body
            com = 1.0/np.sum(all_mass[j] for j in range(len(all_bodies)) if j < i) * \
                   np.sum(all_mass[j] * all_position[j] for j in range(len(all_bodies)) if j < i)
            jpos = body.position * com
            jpos_norm = np.linalg.norm(jpos)
            # body.position - (1.0 / int_mass) * np.sum(all_bodies[k].mass * all_bodies[k].position for k in range(len(all_bodies)))

            # Calculate the jacobian acceleration of body
            epsilon_k = np.sum(all_mass[k] * np.sum(all_mass[:k-1+1]) / (np.sum(all_mass[:k+1]) * np.sum(all_mass[:i-1+1])) *
                               (norm(all_position[k] - all_position[k-1]) / jpos_norm)**2 for k in range(1, i-1))
            c_leg_k = np.sum(0.5 * (3 * (np.dot((all_position[k] - all_position[k-1]), jpos) / (norm(all_position[k] - all_position[k-1]) * jpos_norm))**2 - 1) for k in range(i-1))
            epsilon_l = np.sum(all_mass[l] * np.sum(all_mass[:l-1+1]) / (np.sum(all_mass[:l+1]) * np.sum(all_mass[:i-1+1])) *
                               (norm(all_position[l] - all_position[l-1]) / jpos_norm)**2 for l in range(1+i, len(all_bodies)))
            c_leg_l = np.sum(0.5 * (3 * (np.dot((all_position[l] - all_position[l-1]), jpos) / (norm(all_position[l] - all_position[l-1]) * jpos_norm))**2 - 1) for l in range(1+i, len(all_bodies)))

            body.acceleration = G.value * body.mass * np.gradient() * (1/jpos_norm) * (1 + epsilon_k * c_leg_k + epsilon_l * c_leg_l)

# @autojit
@guvectorize(['void(float64[:,:], float64[:])'], '(m,n)->(m)')
def eclipse(sorted_bodies, dS):

    # Number of steps for r' and theta' integrations
    Nr = 100
    Ntheta = 500
    dtheta_prime = math.pi / Ntheta

    # Define luminosity of differential area
    dS[0] = 0.0

    # for i in range(2):
    #     bb_position_y = sorted_bodies[i, 4]

    # Loop through the sorted bodies, starting with the nearest one. Find the angle between y' and the
    # projected line between the centers of the bodies in the (y', z') plane. The polar coordinate
    # integration will be centered on this line to take advantage of spherical symmetry.
    for i in range(len(sorted_bodies)):
        for j in range(len(sorted_bodies)):
            if i == j:
                continue

            # front_body = sorted_bodies[i]
            # back_body = sorted_bodies[j]

            # fb_position_x = front_body[3]
            fb_position_y = sorted_bodies[i, 4] #front_body[4]
            fb_position_z = sorted_bodies[i, 5] #front_body[5]

            # bb_position_x = back_body[3]
            bb_position_y = sorted_bodies[j, 4] #back_body[4]
            bb_position_z = sorted_bodies[j, 5] #back_body[5]

            # fb_mass = sorted_bodies[i, 0] #front_body[0]
            # bb_mass = sorted_bodies[j, 0] #back_body[0]

            fb_radius = sorted_bodies[i, 1] #front_body[1]
            bb_radius = sorted_bodies[j, 1] #back_body[1]

            # fb_temperature = sorted_bodies[i, 2] #front_body[2]
            bb_temperature = sorted_bodies[j, 2] #back_body[2]

            # Check if stars are close enough to eclipse
            distance = math.sqrt((fb_position_y - bb_position_y) * (fb_position_y - bb_position_y) +
                                 (fb_position_z - bb_position_z) * (fb_position_z - bb_position_z))
            if not distance <= bb_radius + fb_radius:
                continue

            theta0_prime = math.atan2((fb_position_z - bb_position_z),
                                      (fb_position_y - bb_position_y))

            # Determine starting radius for integration
            if distance < bb_radius - fb_radius:
                r_prime = distance + fb_radius
                r_stop = 0.0
            else:
                r_prime = bb_radius
                r_stop = 0.0

            dr_prime = r_prime / Nr

            # Do the surface integration
            while r_prime >= r_stop:
                theta_prime = theta0_prime

                while theta_prime - theta0_prime <= math.pi:
                    dA_y = r_prime * math.cos(theta_prime + dtheta_prime) + bb_position_y
                    dA_z = r_prime * math.sin(theta_prime + dtheta_prime) + bb_position_z

                    if math.sqrt((dA_y - fb_position_y) * (dA_y - fb_position_y) +
                                 (dA_z - fb_position_z) * (dA_z - fb_position_z)) > fb_radius:
                        break

                    theta_prime += dtheta_prime

                # Determine limb darkening (Van Hamme, 1993)
                x = 0.648; y = 0.207
                mu = math.cos((r_prime - dr_prime * 0.5) / bb_radius * math.pi * 0.5)
                limb_dark = 1.0 - x * (1.0 - mu) - y * mu * math.log(mu)

                # # Calculate and return the flux
                flux = sigma * bb_temperature * bb_temperature * bb_temperature * bb_temperature * limb_dark


                # Add the luminosity change for differential area
                dS[0] += 2.0 * flux * (r_prime - dr_prime * 0.5) * dr_prime * (theta_prime - theta0_prime)

                # Check to see that there is no remaining overlap or if center of disk has been reached
                r_prime = r_prime - dr_prime

    # return dS


class Body():
    def __init__(self, mass, radius, temperature, a, ecc, inc, arg_peri, right_asc, true_anomaly):
        # Stellar parameters
        self.mass = (mass * unit.solMass).to(unit.kg).value
        self.radius = (radius * unit.solRad).to(unit.meter).value
        self.temperature = temperature
        
        # Orbital elements
        self.semimajor_axis = (a * unit.AU).to(unit.meter).value
        self.eccentricity = ecc
        self.inclination = np.deg2rad(inc)
        self.arg_periastron = np.deg2rad(arg_peri)
        self.right_ascension = np.deg2rad(right_asc)
        self.true_anomaly = np.deg2rad(true_anomaly)
        # self.epoch = epoch

        # Secondary orbital elements
        self.period = 0.0

        # Kinematics
        self.jposition = np.zeros(3)
        self.position = np.zeros(3)  # (pos * unit.au).to(unit.meter).value
        self.velocity = np.zeros(3)  # (vel * unit.km / unit.second).to(unit.meter / unit.second).value
        self.acceleration = np.zeros(3)  # * unit.meter / unit.second**2
        self.last_acceleration = np.zeros(3)  # * unit.meter / unit.second**2
        self.position_history = []
        self.jposition_history = []

        # Intrinsic characteristics
        self.kinetic_energy = 0.0
        self.luminosity = 4 * np.pi * radius ** 2 * sigma_sb.value * temperature ** 4
        self.flux = 0.0  # * unit.J / (unit.meter * unit.second)

        # Calculate the total flux of the object
        self.total_flux()

        # Calculate initial kinematics
        if self.semimajor_axis > 0.0:
            self.get_kinematics()

    def save_position(self):
        self.position_history.append(self.position)

    def save_jposition(self):
        self.jposition_history.append(self.jposition)

    def get_kinetic_energy(self):
        self.kinetic_energy = 0.5 * self.mass * np.sum(v ** 2 for v in self.velocity)

    def total_flux(self, Nr=100):
        rS = 0.0
        drS = self.radius / Nr

        for i in range(Nr):
            rS += self.radius / Nr
            self.flux += 2 * np.pi * self.get_flux(rS - drS / 2.0) * (rS - drS / 2.0) * drS

    def get_flux(self, r_prime):
        # Determine limb darkening (Van Hamme, 1993)
        x, y = 0.648, 0.207
        mu = np.cos(r_prime / self.radius * np.pi / 2.0)
        limb_dark = 1.0 - x * (1.0 - mu) - y * mu * np.log(mu)

        # Calculate and return the flux
        return sigma_sb.value * self.temperature ** 4 * limb_dark
    
    def pickleable(self):
        return np.array([self.mass, self.radius,
                         self.temperature,
                         self.position[0],
                         self.position[1],
                         self.position[2]], dtype='float64')

    def _get_eccentric_anomaly(self, me, ecc, tol=1.0e-8):
        ea_temp = me
        ratio = 1.0

        while abs(ratio) > tol:
            f_ea = ea_temp - ecc * np.sin(ea_temp) - me
            f_ea_p = 1.0 - ecc * np.cos(ea_temp)
            ratio = f_ea / f_ea_p

            if abs(ratio) > tol:
                ea_temp -= ratio
            else:
                ea = ea_temp

        return ea

    def get_kinematics(self):
        mu = G.value * self.mass
        a, e, i = self.semimajor_axis, self.eccentricity, self.inclination
        W, w = self.right_ascension, self.arg_periastron
        t, T = 0.0, 212.12316

        mean_anomaly = self.true_anomaly # np.sqrt(mu / a**3) * (t - T)
        eccentric_anomaly = self._get_eccentric_anomaly(mean_anomaly, e)

        # Unrotated positions and velocities
        p = (1.0 - e**2)
        mean_motion = np.sqrt(mu/a**3)
        position = np.array([
            a * (np.cos(eccentric_anomaly) - e),
            p * a * np.sin(eccentric_anomaly),
            0.0
        ])

        velocity = np.array([
            -a * mean_motion * np.sin(eccentric_anomaly) / (1.0 - e * np.cos(eccentric_anomaly)),
            p * a * mean_motion * np.cos(eccentric_anomaly) / (1.0 - e * np.cos(eccentric_anomaly)),
            0.0
        ])

        # Rotate by argument of periastron in orbit plane
        pposition = np.array([
            position[0] * np.cos(w) - position[1] * np.sin(w),
            position[0] * np.sin(w) + position[1] * np.cos(w),
            position[2]
        ])

        pvelocity = np.array([
            velocity[0] * np.cos(w) - velocity[1] * np.sin(w),
            velocity[0] * np.sin(w) + velocity[1] * np.cos(w),
            position[2]
        ])

        # Rotation by incliantion about x axis
        position = np.array([
            pposition[0],
            pposition[1] * np.cos(i) - pposition[2] * np.sin(i),
            pposition[1] * np.sin(i) + pposition[2] * np.cos(i)
        ])

        velocity = np.array([
            pvelocity[0],
            pvelocity[1] * np.cos(i) - pvelocity[2] * np.sin(i),
            pvelocity[1] * np.sin(i) + pvelocity[2] * np.cos(i)
        ])

        # Rotate by longitude of the node about the z axis
        position = np.array([
            position[0] * np.cos(W) - position[1] * np.sin(W),
            position[0] * np.sin(W) + position[1] * np.cos(W),
            position[2]
        ])

        velocity = np.array([
            velocity[0] * np.cos(W) - velocity[1] * np.sin(W),
            velocity[0] * np.sin(W) + velocity[1] * np.cos(W),
            velocity[2]
        ])
        self.position = position
        self.velocity = velocity

    def get_kinematics_new(self):
        mu = G.value * self.mass
        a, e, i = self.semimajor_axis, self.eccentricity, self.inclination
        Omega, omega = self.right_ascension, self.arg_periastron
        t, T = 0.0, 30.0

        mean_anomaly = self.true_anomaly #.sqrt(mu / a**3) * (t - T)
        eccentric_anomaly = self._get_eccentric_anomaly(mean_anomaly, e)
        true_anomaly = 2.0 * np.arctan(((1.0 + e)/(1.0 - e))**0.5 * np.tan(eccentric_anomaly * 0.5))
        true_anomaly = self.true_anomaly
        p = a * (1.0 - e**2)
        r = p / (1.0 + e * np.cos(true_anomaly))

        h = (mu * a * (1.0 - e**2))**0.5



        # self.position = np.array([
        #     r * (np.cos(Omega) * np.cos(omega + true_anomaly) - np.sin(Omega) * np.sin(omega + true_anomaly) * np.cos(i)),
        #     r * (np.sin(Omega) * np.cos(omega + true_anomaly) + np.cos(Omega) * np.sin(omega + true_anomaly) * np.cos(i)),
        #     r * (np.sin(i) * np.sin(omega + true_anomaly))
        # ])
        # self.velocity = np.array([
        #     (h * e / (r * p)) * np.sin(true_anomaly) - (h/r) * (np.cos(Omega) * np.sin(omega + true_anomaly) + np.sin(Omega) * np.cos(omega + true_anomaly) * np.cos(i)),
        #     (h * e / (r * p)) * np.sin(true_anomaly) - (h/r) * (np.sin(Omega) * np.sin(omega + true_anomaly) - np.cos(Omega) * np.cos(omega + true_anomaly) * np.cos(i)),
        #     (h * e / (r * p)) * np.sin(true_anomaly) + (h/r) * np.sin(i) * np.cos(omega + true_anomaly)
        # ])

        self.position = np.array([
            # self.position[2] * np.cos(self.inclination) +
            self.position[0] * np.sin(self.inclination),
            self.position[1],
            # self.position[2] * np.sin(self.inclination)
            - self.position[0] * np.cos(self.inclination)
        ])

        self.velocity = np.array([
            # self.velocity[2] * np.cos(self.inclination) +
            self.velocity[0] * np.sin(self.inclination),
            self.velocity[1],
            # self.velocity[2] * np.sin(self.inclination)
            - self.velocity[0] * np.cos(self.inclination)
        ])

        print(self.velocity)

    def get_kinematics_old(self):
        mu = G.value * self.mass
        p = self.semimajor_axis * (1.0 - self.eccentricity**2)
        r = p / (1.0 + self.eccentricity * np.cos(self.true_anomaly))
        v = np.sqrt(G.value * self.mass * (2.0/r - 1.0/self.semimajor_axis))
        vr = -v * np.sin(self.inclination) * np.sin(self.true_anomaly + self.arg_periastron)



        position = np.array([
            (p * np.cos(self.true_anomaly)) / (1 + self.eccentricity * np.cos(self.true_anomaly)),
            (p * np.sin(self.true_anomaly)) / (1 + self.eccentricity * np.cos(self.true_anomaly)),
            0.0
        ])

        velocity = np.array([
             -np.sqrt(mu / p) * np.sin(self.true_anomaly),
            np.sqrt(mu / p) * (self.eccentricity + np.cos(self.true_anomaly)),
           0.0
        ])

if __name__ == '__main__':
    system = System()
    # mass, radius, temperature, a, ecc, inc, arg_peri, right_asc, true anomaly
    # system.add_body(5.9, 3.2, 15200.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                    # np.array([-0.80971, 0.0, 0.0]), np.array([0.0, -36.2, 0.0]))
    # system.add_body(5.6, 2.9, 13700.0, 0.043166, 0.1573, 88.89, 214.6, 0.0, 180.0)
    #                 np.array([0.80971, 0.0, 0.0]), np.array([0.0, 36.2, 0.0]))

    system.add_body(0.6897, 0.6489, 4450.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    system.add_body(0.20255, 0.22623, 3000.0, 0.22431, 0.15944, 90.30401, 263.464, 0.0, 188.884)
    system.add_body(0.000317770494, 0.07497, 170.0, 0.7048, 0.0069, 90.032, 318.0, 0.003, 137.1126)

    t0 = time.time()
    system.run(864.0, 10000)
    print(time.time() - t0)
    # system_run = numba.autojit(system.run)
    # system_run(864.0, 1000)