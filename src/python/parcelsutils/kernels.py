from parcels import JITParticle, ScipyParticle, ParcelsRandom, Variable
import numpy as np
import math

# Functions
#     edna_decay(particle, fieldset, time):
#     AdvectionRK43D_DiffusionM12D(particle, fieldset, time)
#     smagdiff(particle, fieldset, time)
#     VertDiffusionKh(particle, fieldset, time)
#     DeleteParticle(particle, fieldset, time)
#     cleanup(particle, fieldset, time)

class eDNAParticle(JITParticle):
    edna_conc = Variable('edna_conc', dtype=np.float32, initial=1.)
    to_delete = Variable('to_delete', dtype=np.int32, initial=0)


class eDNAScipyParticle(ScipyParticle):
    edna_conc = Variable('edna_conc', dtype=np.float32, initial=1.)
    to_delete = Variable('to_delete', dtype=np.int32, initial=0)


def edna_decay(particle, fieldset, time):

    # Numerical approximation using Crank Nicolson

    # eDNA decay model from Lamb et al 2022
    a0 = -0.0776
    aW = 0.0620
    aT = 0.0069
    # Delete particle when concentration below this value
    MIN_EDNA_CONC = 0.001

    temp = fieldset.T[time, particle.depth, particle.lat, particle.lon]
    decayr = a0 + aW + aT * temp
    # JIT compilation doesn't support max([0,])
    if decayr < 0:
        decayr = 0
    dthr = particle.dt / 3600  # convert particle dt in seconds to hrs
    # Crank Nicolson approximation to exponential decay
    facCN = (1 - decayr / 2 * dthr) / (1 + decayr / 2 * dthr)
    particle.edna_conc = particle.edna_conc * facCN

    # Delete particle when concentration is too low
    if particle.edna_conc < MIN_EDNA_CONC:
        particle.to_delete = 1


# New Advection Diffusion Algorithm
# (Bug in Milstein + template for 3D Advection) #858
# Contributed by ElizaJayne11 (Wendy)
# Removed conversion of m to degrees in particle displacement
def AdvectionRK43D_DiffusionM12D(particle, fieldset, time):
    # Kernel for 3D advection-solved using fourth order Runge-Kutta (RK4)
    # and 3D Horizontal diffusion -- solved using the Milstein scheme at first order (M1).
    # Note that this approach does not consider the interaction of advection and diffusion within the time step
    # Milstein requires computation of the local gradient in Kh in both directions
    # These gradients are estimated using a centered finite different approach of O(h^2)
    # Where h = dres (set in the main code). this should be at least an order of magnitude
    # less than the typical grid resolution, but not so small as to introduce catestrophic cancellation
    # Parcels 2.2 has used a uniform random number to determine the random kick
    # Mathematically, a normal distribution is more accurate for a given time step as noted in Parcels 2.2
    # Parcels 2.2 argue that uniform is more efficient, which is generally true when comparing individual calls
    # However, since the uniform only converges on the correct distribution after about 5 steps,
    # it is not overall more efficient than a normal distribution.
    # Parcel 2.2. also argue that the random increments remain bounded.
    # It is not immediately apparent to me why "boundedness" is an advantage.
    # Despite that,I have kept the code below using a uniform distribution
    # in part b/c horizontal diffusion is already very approximate (our diffusivities are not precise!)
    # There is also a long history of Ocean Modelers using a uniformly distributed random number
    # (e.g. Visser, Ross & Sharples etc)

    # Current space-time location of particle
    lonp = particle.lon
    latp = particle.lat
    depthp = particle.depth
    dt = particle.dt

    # RK4 for Advection
    # Changed variable naming from Parcels 2.2 to 
    # (1) improve efficiency as we were instructed that it is faster to rename variables that are "." 
    # than to recall and 
    # (2) to make the code easier to read
    # Note: I have followed Parcels 2.2. and left the RK4 to be computing slopes within the time step 
    # instead of displacements
    # Both approaches are used in the literature 

    
    #start at current space-time point using those values to get slope
    (u1, v1, w1) = fieldset.UVW[time, depthp, latp, lonp]
    
    #Get estimate for slope at midpoint using previous estimate for slope
    lon1, lat1, dep1 = (lonp + u1*.5*dt, latp + v1*.5*dt, depthp + w1*.5*dt) 
    (u2, v2, w2) = fieldset.UVW[time + .5*dt, dep1, lat1, lon1]
    
    #Get improved estimate for slope at midpoint using previous estimate for slope
    lon2, lat2, dep2 = (lonp + u2*.5*dt, latp + v2 *.5*dt, depthp + w2*.5*dt)
    (u3, v3,w3) = fieldset.UVW[time + .5 * dt, dep2, lat2, lon2]
    
    #Get estimate for slope at endppoint using previous estimate for slope at midpoint
    lon3, lat3, dep3 = (lonp + u3*dt, latp + v3*dt, depthp+w3*dt)
    (u4, v4, w4) = fieldset.UVW[time + dt, dep3, lat3, lon3]
    
    # Calculate particle displacement due to local advection
    # This assumes that fieldset has already converted u,v,w to degrees
    Advect_lon = ((u1 + 2 * u2 + 2 * u3 + u4) / 6.) * dt
    Advect_lat = ((v1 + 2 * v2 + 2 * v3 + v4) / 6.) * dt
    Advect_dep = ((w1 + 2 * w2 + 2 * w3 + w4) / 6.) * dt

    # Milstein for Horizontal Diffusion
    Khdifferencedist = fieldset.dres  # in degrees
    # Note Kh is in m^2/s here, unlike built-in that has already converted to degrees
    # To save on repeated conversions, only done after compute displace ent

    # Sample random number (Wiener increment) with zero mean and std of sqrt(dt) from uniform distribution
    #  dWx = ParcelsRandom.uniform(-1., 1.) * math.sqrt(math.fabs(dt) * 3)
    #  dWy = ParcelsRandom.uniform(-1., 1.) * math.sqrt(math.fabs(dt) * 3)
    dWx = ParcelsRandom.normalvariate(0., math.sqrt(math.fabs(dt)))
    dWy = ParcelsRandom.normalvariate(0., math.sqrt(math.fabs(dt)))

    # Get estimate of random kick in x-direction based on local diffuvisity (neglects spatial variation)
    bx = math.sqrt(2 * fieldset.Kh_zonal[time, depthp, latp, lonp])
    # Get estimate of randome kick in y-direction based on local diffusivity (neglects spatial variation)
    by = math.sqrt(2 * fieldset.Kh_meridional[time, depthp, latp, lonp])

    # Get estimate of zonal diffusivity gradient at current location using finite centered finite differenece
    # This derivative is used to correct basic random kick due to variable diffusivity
    Kxp1 = fieldset.Kh_zonal[time, depthp, latp, lonp + Khdifferencedist]
    Kxm1 = fieldset.Kh_zonal[time, depthp, latp, lonp - Khdifferencedist]
    dKdx = (Kxp1 - Kxm1) / (2 * Khdifferencedist)
 
    #Get estimate of meridional gradient at current location using fininte centered difference
    #This derivative is used to correct basic random kick due to variable diffusivity 
    Kyp1 = fieldset.Kh_meridional[time, depthp, latp + Khdifferencedist, lonp]
    Kym1 = fieldset.Kh_meridional[time, depthp, latp - Khdifferencedist, lonp]
    dKdy = (Kyp1 - Kym1) / (2 * Khdifferencedist) 
    
    #Calculate particle horizontal displacement due to local diffusiona and diffusivity gradient
    DiffH_lon = bx*dWx + 0.5 * dKdx * (dWx**2 + 1)*dt #Wonky units
    DiffH_lat = by*dWy + 0.5 * dKdy * (dWy**2 + 1)*dt #Wonky units
    #to_lat = 1/1000./1.852/60.
    #to_lon = to_lat/math.cos(particle.lat*math.pi/180)
    #DiffH_lon=DiffH_lon*to_lon
    #DiffH_lat=DiffH_lat*to_lat

    #Particle positions are updated only after evaluating all terms (i.e. Advection + Diffusion simulataneous)
    #Note this approach does not consider the interaction of adv and diff on estimated particle displacements 
    #within the time step)
    particle.lon += Advect_lon + DiffH_lon
    particle.lat += Advect_lat + DiffH_lat
    particle.depth += Advect_dep


def VertDiffusionKh(particle, fieldset, time):
    """Kernel for simple 3D diffusion where diffusivity (Kh) is assumed uniform.

    Assumes that fieldset has uniform diffusivity, `Kh_vertical`.
    These should  added via:
        fieldset.add_constant("Kh_vertical", 0.01)

    This kernel assumes diffusivity gradients are zero and is therefore more efficient.
    Since the perturbation due to diffusion is in this case isotropic independent, this
    kernel contains no advection and can be used in combination with a seperate
    advection kernel.

    The Wiener increment `dW` is normally distributed with zero
    mean and a standard deviation of sqrt(dt).
    """
    # Wiener increment with zero mean and std of sqrt(dt)
    dWz = ParcelsRandom.normalvariate(0, math.sqrt(math.fabs(particle.dt)))

    bz = math.sqrt(2.0 * fieldset.Kh_vertical)

    particle.depth += bz * dWz
 
    # If particle is above water surface, this must be due to diffusion,
    # so reflect it downwards.
    if particle.depth < 0:
        particle.depth = math.fabs(particle.depth)

    # If particle is below seabed, delete.
    # eDNA will not degrade in sediment so this last position must be considered
    if particle.depth > fieldset.bathy[particle]:
        particle.to_delete = 1


def VertDiffusionKh_visser(particle, fieldset, time):
    """Kernel for simple 3D diffusion where diffusivity (Kh) is provided by model (Kh_v3d).

    Follows:
    Visser 1997 Using random walk models to simulate the vertical distribution
    of particles in a turbulent water column.
    """
    
    deltat = math.fabs(particle.dt)
    # Deterministic term
    determterm = fieldset.dKh_v3d[particle] * deltat
    # Uniform with variance  1/3
    R = ParcelsRandom.uniform(-1, 1)
    # Variance of random distribution
    # Must be represented in decimal format for JIT kernel to run
    r = 0.3333333
    # point at which K will be evaluated
    displdepth = particle.depth + 0.5 * fieldset.dKh_v3d[particle] * deltat
    # random term 
    randterm = R * (2*fieldset.Kh_v3d[particle.time, displdepth, particle.lat, particle.lon] * deltat / r )** 0.5
    
    particle.depth += determterm + randterm
    

# Replacement for a reflective boundary condition at the surface
def SinkParticle(particle, fieldset, time):
    particle.depth += 0.5

    
# Replacement for a reflective boundary condition at the seabed
def RaiseParticle(particle, fieldset, time):
    # Extreme random excursion as 3*std. dev. of wiener increment
    #sigma3 =  math.sqrt(math.fabs(particle.dt))*3
    particle.depth -=  0.5
    
    
# Passed to execute as recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle} 
def DeleteParticle(particle, fieldset, time):
    particle.delete()


# cleanup kernel should be last to run because of JIT mode limitations
def cleanup(particle, fieldset, time):
    if particle.to_delete > 0:
        particle.delete()
