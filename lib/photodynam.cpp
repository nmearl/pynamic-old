/*
  Josh Carter, 2013

  Demonstration of 'photodynamics' code.

  Call code as photodynam <input_file> <report_times> [> <output_file>].

  <input_file> is file of initial coordinates and properties in
  following format:

  <N> <time0>
  <step_size> <orbit_error>

  <mass_1> <mass_2> ... <mass_N>
  <radius_1> <radius_2> ... <radius_N>
  <flux_1> <flux_2> ... <flux_N>
  <u1_1> <u1_2> ... <u1_N>
  <u2_1> <u2_2> ... <u2_N>

  <a_1> <e_1> <i_1> <o_1> <l_1> <m_1>
  ...
  <a_(N-1)> <e_(N-1)> <i_(N-1)> <o_(N-1)> <l_(N-1)> <m_(N-1)>

  where the Keplerian coordinates
  (a = semimajor axis, e = eccentricity, i = inclination,
  o = argument periapse, l = nodal longitude, m = mean anomaly) are the
  N-1 Jacobian coordinates associated with the masses as ordered above.
  Angles are assumed to be in radians. The observer is along the positive z axis.
  Rotations are performed according to Murray and Dermott.

  For example, for Kepler-16, kepler16_pd_input.txt:

  3 212.12316
  0.01 1e-16

  0.00020335520 5.977884E-05    9.320397E-08
  0.00301596700 0.00104964500   0.00035941463
  0.98474961000 0.01525038700   0.00000000000
  0.65139908000 0.2     0.0
  0.00587581200 0.3     0.0

  2.240546E-01 1.595442E-01 1.576745E+00 4.598385E+00 0.000000E+00 3.296652E+00
  7.040813E-01 7.893413E-03 1.571379E+00 -5.374484E-01 -8.486496E-06 2.393066E+00

  <report_times> is a list of times to report the outputs.
  First line is a space-separated list of single character-defined
  output fields according to:

  t = time
  F = flux
  a = semi-major axes
  e = eccentricities
  i = sky-plane inclinations
  o = arguments of periapse
  l = nodal longitudes
  m = mean anomalies
  K = full keplerian osculating elements
  x = barycentric, light-time corrected coordinates
  v = barycentric, light-time corrected velocities
  M = masses
  E = fractional energy change from t0
  L = fraction Lz change from t0

  For example, the first line could be

  t F E

  and the output would have three columns of time flux and
  fractional energy loss.

  Output is written to standard out.

*/

#include "n_body_state.h"
#include "n_body_lc.h"
#include "omp.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

using namespace std;

extern "C"
{
  void start(double *out_fluxes, double *rv, int N, double t0, double maxh, double orbit_error, int in_times_size,
            double *in_times, double *mass, double *radii, double *flux, double *u1, double *u2, double *a, double *e,
            double *inc, double *om, double *ln, double *ma);
}

void start(double *out_fluxes, double *rv, int N, double t0, double maxh, double orbit_error, int in_times_size,
            double *in_times, double *mass, double *radii, double *flux, double *u1, double *u2, double *a, double *e,
            double *inc, double *om, double *ln, double *ma)
{

    // Instantiate state.  Time t0 is epoch of above coordinates
    NBodyState state(mass,a,e,inc,om,ln,ma,N,t0);
    int status;
    double t;

    // Evaluate the flux at time t0 using the getBaryLT() member method
    //  of NBodyState which returns NX3 array of barycentric, light-time
    //  corrected coordinates

    //  flux = occultn(state.getBaryLT(),radii,u1,u2,flux,N);

    // Now integrate forward in time to time t0+100 with stepsize 0.01 days orbit
    //  error tolerance of 1e-20 and minimum step size of 1e-10 days

    #pragma omp parallel for
    for (int i = 0; i < in_times_size; i++)
    {
        status = state(in_times[i],maxh,orbit_error,1.0e-4);

        // Now get the flux at the new time
        out_fluxes[i] = occultn(state.getBaryLT(),radii,u1,u2,flux,N);
        rv[i] = state.V_Z_LT(2);
    }
}
