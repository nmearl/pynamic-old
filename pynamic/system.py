__author__ = 'nmearl'

import numpy as np
import pylab
from astropy.constants.si import G, M_sun, R_sun
from astropy import units as unit


class System():

    def __init__(self):
        self.all_bodies = []
        self.mu = 0.0
        self.a = 0.0

    def add_body(self, mass, radius, temperature, period, eccentricity, inclination, periastron, pos, vel):
        # Add star to list of all bodies
        self.all_bodies.append(Body(mass, radius, temperature, period, eccentricity, inclination, periastron, pos, vel))

    def run(self, dt, nsteps):
        # Run loop
        for i in range(nsteps):
            # Calculate force and acceleration on body
            for body in self.all_bodies:
                self.gforce(body)

            for body in self.all_bodies:
                # Save the position for plotting purposes
                body.save_position()

                # Advance the position
                body.position = body.position + body.velocity * dt + 0.5 * body.acceleration * dt**2

            # Calculate new forces
            for body in self.all_bodies:
                self.gforce(body)

                # Advance the velocity
                body.velocity += 0.5*(body.acceleration + body.last_acceleration)*dt

        for body in self.all_bodies:
            x, y, z = zip(*body.position_history)
            pylab.plot([i.value for i in x], [i.value for i in y])

        pylab.show()

    def gforce(self, body):
        body.force = np.zeros(3) * unit.kg * unit.meter / unit.second**2
        body.last_acceleration = body.acceleration

        for other_body in self.all_bodies:
            if other_body == body:
                continue

            rvec = body.position - other_body.position
            distance = np.linalg.norm(body.position - other_body.position) * unit.meter
            body.force += - G * body.mass * other_body.mass * rvec / np.abs(distance)**3

        body.acceleration = body.force / body.mass  # - G * other_body.mass * rvec / np.abs(rvec)


class Body():

    def __init__(self, mass, radius, temperature, period, eccentricity, inclination, periastron, pos, vel):
        self.mass = (mass * unit.solMass).to(unit.kg)
        self.radius = (radius * unit.solRad).to(unit.m)
        self.inclination = inclination
        self.a = 0.0
        self.position = (pos * unit.au).to(unit.meter) #np.zeros(3)
        self.velocity = (vel * unit.km / unit.second).to(unit.meter/unit.second) #np.zeros(3)
        self.acceleration = np.zeros(3) * unit.meter / unit.second**2
        self.last_acceleration = np.zeros(3) * unit.meter / unit.second**2
        self.position_history = []

    def save_position(self, skypos=False):
        self.position_history.append(self.position)

    def kinetic_energy(self):
        self.ke = 0.5 * self.mass * np.sum(v**2 for v in self.velocity)

    def sky_position(self):
        return self.position * np.array([np.sin(self.inclination), 1.0, -np.cos(self.inclination)])

if __name__ == '__main__':
    system = System()
    system.add_body(1.5, 1.0, 3000.0, 1.0, 0.5, np.deg2rad(90), 0.0, np.array([0.0, 0.0, 0.0]), np.array([0.0, 5.0, 0.0]))
    system.add_body(1.0, 1.0, 3000.0, 1.0, 0.5, np.deg2rad(90), 0.0, np.array([-0.05, 0.0, 0.0]), np.array([0.0, -50.0, 0.0]))
    system.run(3600.0 * unit.second, 1000)