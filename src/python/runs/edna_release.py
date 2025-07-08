#!/usr/bin/env python3
from parcelsutils.utils import plot_traj, plot_traj_decay, traj_to_grid
from parcelsutils.kernels import AdvectionRK43D_DiffusionM12D, edna_decay
import argparse
from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4_3D, AdvectionDiffusionM1
from parcels import Variable, AdvectionRK4
from glob import glob
import numpy as np
from datetime import timedelta as delta
from datetime import datetime
import time
import xarray as xr
import pandas as pd
import json
import shutil
import sys
import os
import psutil
import shutil
sys.path.insert(1, '../')
from parcelsutils import kernels as pkernels

datapath = '/gpfs/scratch/uck09rvu/AMM7/2012/201201/'
fpattern = '201201*'
dstart   = '2012-01-01 09:00:00'
dend     = '2012-01-15 09:00:00'
# releases_fpath = '../data/hypothetical_observations_v2_wgs84.geojson'
# releases_fpath = '../data/release_locations_70km_buffer_v2_wgs84.geojson'
releases_fpath = '../data/hypothetical_observations_v2_wgs84.geojson'

runtimedays = 25
outputdthours = 3  # hours
# Time step
dtminutes = 5 / 60
# Directory for mpi parallel zarrs
# temppath = '/tmp/'
temppath = '/gpfs/scratch/uck09rvu/C8389-NIS-eDNA-modelling/runs/nompi/'


def prep_fieldset():

    tfiles = sorted(glob(datapath + fpattern + 'grid_T.nc'))
    ufiles = sorted(glob(datapath + fpattern + 'grid_U.nc'))
    vfiles = sorted(glob(datapath + fpattern + 'grid_V.nc'))
    wfiles = sorted(glob(datapath + fpattern + 'grid_W_davmdz.nc'))
    for tfile, ufile, vfile, wfile in zip(tfiles, ufiles, vfiles, wfiles):
        print(tfile)
        print(ufile)
        print(vfile)
        print(wfile)

    meshmask = datapath + 'coordinates.nc'

    filenames = {'T':        {'lon': meshmask, 'lat': meshmask, 'depth': tfiles, 'data': tfiles},
                 'U':        {'lon': meshmask, 'lat': meshmask, 'depth': ufiles, 'data': ufiles},
                 'V':        {'lon': meshmask, 'lat': meshmask, 'depth': vfiles, 'data': vfiles},
                 'W':        {'lon': meshmask, 'lat': meshmask, 'depth': wfiles, 'data': wfiles},
                 'Kh_v3d':   {'lon': meshmask, 'lat': meshmask, 'depth': wfiles, 'data': wfiles},
                 'dKh_v3d':  {'lon': meshmask, 'lat': meshmask, 'depth': wfiles, 'data': wfiles},
                 'depthw4d': {'lon': meshmask, 'lat': meshmask, 'depth': wfiles, 'data': wfiles},
                 }

    variables = {'T':        'votemper',
                 'U':        'vozocrtx',
                 'V':        'vomecrty',
                 'W':        'wo',
                 'Kh_v3d':   'avm',
                 'dKh_v3d':  'davmdz',
                 'depthw4d': 'depthw4d',
                 }

    dimensions = {'T':        {'lon': 'glamt', 'lat': 'gphit', 'depth': 'not_yet_set', 'time': 'time_counter'},
                  'U':        {'lon': 'glamf', 'lat': 'gphif', 'depth': 'not_yet_set', 'time': 'time_counter'},
                  'V':        {'lon': 'glamf', 'lat': 'gphif', 'depth': 'not_yet_set', 'time': 'time_counter'},
                  'W':        {'lon': 'glamf', 'lat': 'gphif', 'depth': 'not_yet_set', 'time': 'time_counter'},
                  'Kh_v3d':   {'lon': 'glamf', 'lat': 'gphif', 'depth': 'not_yet_set', 'time': 'time_counter'},
                  'dKh_v3d':  {'lon': 'glamf', 'lat': 'gphif', 'depth': 'not_yet_set', 'time': 'time_counter'},
                  'depthw4d': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'not_yet_set', 'time': 'time_counter'},
                  }

    # could also work with cs='auto'
    # cs = {'time': ('time', 1),
    #      'depth': ('depth', 51),
    #      'lat': ('latitude', 330),
    #      'lon': ('longitude', 233)
    #      }
    #cs = {'time': ('time', 1),
    #      'depth': ('depth', 51),
    #      'lat': ('latitude', 90),
    #      'lon': ('longitude', 120)
    #      }
    cs = 'auto'

    fieldset = FieldSet.from_nemo(
        filenames,
        variables,
        dimensions,
        # To allow for release on 1st of month 00:00:00
        allow_time_extrapolation=True,
        chunksize=cs
    )
    fieldset.T.set_depth_from_field(fieldset.depthw4d)
    fieldset.U.set_depth_from_field(fieldset.depthw4d)
    fieldset.V.set_depth_from_field(fieldset.depthw4d)
    fieldset.W.set_depth_from_field(fieldset.depthw4d)
    fieldset.Kh_v3d.set_depth_from_field(fieldset.depthw4d)
    fieldset.dKh_v3d.set_depth_from_field(fieldset.depthw4d)
    # The default seems to be nearest neighboor which results
    # in K=0 in the top lavel, and particles not moving vertically
    fieldset.Kh_v3d.interp_method = 'cgrid_velocity'

    # Required for Adv-diff Milstein and for DiffusionUniformKh
    fieldset.add_constant_field("Kh_zonal", 10, mesh='spherical')  # in m/s, conversions done automatically
    fieldset.add_constant_field("Kh_meridional", 10, mesh='spherical')
    fieldset.add_constant('dres', 5e-5)  # In Degrees ~ 5km.
    # From experience, its size of dres should be smaller than the spatial
    # resolution of the data, but within reasonable limits of machine precision
    # to avoid numerical errors. We are working on a method to compute
    # gradients differently so that specifying dres is not necessary anymore.
    #
    # Required for Smagorinsky derived diffusivity
    # https://nbviewer.org/github/OceanParcels/parcels/blob/master/parcels/examples/tutorial_diffusion.ipynb
    # x = fieldset.U.grid.lon
    # y = fieldset.U.grid.lat
    # cellareas = Field(name='cellareas', data=fieldset.U.cellareas(), lon=x, lat=y)
    # fieldset.add_field(cellareas)
    # fieldset.add_constant('Cs', 0.1) # Smagorisnky coefficient

    return fieldset


def get_mpi_rank(mpimode):
    # Determine mpi_rank
    if mpimode:
        comm = MPI.COMM_WORLD
        return comm.Get_rank()
    else:
        return 0


def merge_zarrs(inputdir, outputzarr, chunksizes):
    """
    In MPI mode each process will write its own zarr file
    names proc<NN>.zarr where NN is the process MPI rank.
    At the end it is necessary to merge into a simple zarr
    storage
    """

    # Save storage space by recasting as float32
    varType = {
        'lat': np.dtype('float32'),
        'lon': np.dtype('float32'),
        'time': np.dtype('float64'),  # to avoid bug in xarray
        'z': np.dtype('float32'),
        'edna_conc': np.dtype('float32')
    }

    # Make sure all processes have finished writing to file
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    print(f'************************* Process {rank} is at the barrier', flush=True)
    comm.Barrier()

    if rank == 0:
        tic = time.time()
        print(f'Rank {rank} is merging files ', flush=True)
        files = glob(os.path.join(inputdir, "proc*"))
        dataIn = xr.concat([xr.open_zarr(f, decode_times=False) for f in files],
                           dim='trajectory',
                           compat='no_conflicts',
                           coords='minimal')

        print('merged data:', dataIn, flush=True)

        # Recast to save space
        for v in varType.keys():
            dataIn[v] = dataIn[v].astype(varType[v])

        # Reset chunking across processors
        for v in dataIn.variables:
            if 'chunks' in dataIn[v].encoding:
                del dataIn[v].encoding['chunks']
        dataIn = dataIn.chunk(chunksizes)
        dataIn.to_zarr(outputzarr)

        exectime = time.time() - tic
        print(f'    time2: {exectime:.0f} s mem: {mem_used_MB():.0f} MB')


def mem_used_MB():
    # Assess memory usage

    process = psutil.Process(os.getpid())
    mem_B_used = process.memory_info().rss
    return mem_B_used / (1024 * 1024)


def main():
    # Jamie Pringle recommends 1e6x10 for very large outputs
    # First dimension (trajectory is reduced to number of particles
    # released in the first time step, which can be as litle as 1
    chunksizes = {'trajectory': 5 * int(1e4), 'obs': 100}

    # Initialise random number generator with a new seed
    ParcelsRandom.seed(os.getpid())

    # Default is JIT
    ParticleClass = JITParticle  # ScipyParticle

    parser = argparse.ArgumentParser()
    parser.add_argument('-N', '--numparticles',
                        nargs=1,
                        type=int,
                        required=True,
                        help='number of particles to release')
    parser.add_argument('--outfname', '-o',
                        nargs=1,
                        type=str,
                        required=True,
                        help='output file path')
    parser.add_argument('-C', '--conf',
                        type=int,
                        choices=[1, 10, 20, 30, 40, 11, 21, 31, 41, 42, 43, 44, 45],
                        required=False,
                        default=10,
                        help='kernel and class configuration')
    args = parser.parse_args()
    npart = args.numparticles[0]
    conf = args.conf
    outfname = args.outfname[0]

    try:
        from mpy4py import MPI
        mpimode = True
    except ModuleNotFoundError:
        mpimode = False

    if get_mpi_rank(mpimode) == 0:
        if os.path.exists(outfname):
            print(f"Error: {outfname} already exists.")
            sys.exit(1)

    if mpimode:
        base = os.path.basename(outfname)
        tempfname = temppath + base
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        if os.path.exists(tempfname) and (rank == 0):
            print(f"Deleting {tempfname}.")
            shutil.rmtree(tempfname)
        print(f'************************* Process {rank} is at the barrier', flush=True)
        # Wait until process rank=0 deletes temp files
        comm.Barrier()
    else:
        tempfname = outfname

    if conf == 1:
        ParticleClass = pkernels.eDNAParticle
        kernels = [pkernels.edna_decay, pkernels.cleanup]
    elif conf == 10:
        ParticleClass = JITParticle
        kernels = [AdvectionRK4]
    elif conf == 20:
        ParticleClass = JITParticle
        kernels = [AdvectionRK4, AdvectionDiffusionM1]
    elif conf == 30:
        ParticleClass = JITParticle
        kernels = [AdvectionRK4_3D, AdvectionDiffusionM1]
    elif conf == 40:
        ParticleClass = JITParticle
        kernels = [AdvectionRK43D_DiffusionM12D]
    elif conf == 11:
        ParticleClass = JITParticle
        kernels = [AdvectionRK4, edna_decay]
    elif conf == 21:
        ParticleClass = JITParticle
        kernels = [AdvectionRK4, AdvectionDiffusionM1, edna_decay]
    elif conf == 31:
        ParticleClass = JITParticle
        kernels = [AdvectionRK4_3D, AdvectionDiffusionM1, edna_decay]
    elif conf == 41:
        ParticleClass = JITParticle
        kernels = [AdvectionRK43D_DiffusionM12D, edna_decay]
    elif conf == 42:
        ParticleClass = pkernels.eDNAParticle
        kernels = [pkernels.AdvectionRK43D_DiffusionM12D,
                   pkernels.edna_decay,
                   pkernels.cleanup]
    elif conf == 43:
        ParticleClass = pkernels.eDNAParticle
        kernels = [pkernels.AdvectionRK43D_DiffusionM12D,
                   pkernels.VertDiffusionKh_visser,
                   pkernels.edna_decay,
                   pkernels.cleanup
                   ]
    elif conf == 44:
        ParticleClass = pkernels.eDNAScipyParticle
        kernels = [pkernels.AdvectionRK43D_DiffusionM12D,
                   pkernels.VertDiffusionKh_visser,
                   pkernels.edna_decay,
                   pkernels.cleanup
                   ]
    elif conf == 45:
        ParticleClass = pkernels.eDNAParticle
        kernels = [AdvectionRK4_3D,
                   pkernels.VertDiffusionKh_visser,
                   pkernels.edna_decay,
                   pkernels.cleanup
                   ]

    # read release loction from file
    with open(releases_fpath) as f:
        data = json.load(f)
    lats = [i['geometry']['coordinates'][1] for i in data['features']]
    lons = [i['geometry']['coordinates'][0] for i in data['features']]
    nlocs = len(lats)
    ntimereleases = (npart // nlocs)
    npart = ntimereleases * nlocs
    times = []
    # Generate sequence of timestamps and convert to array of datetime
    uniquetimes = pd.date_range(start=dstart,
                                end=dend,
                                periods=ntimereleases).to_pydatetime()
    for utime in uniquetimes:
        times = times + [utime] * nlocs

    print(f'Number of locations:{nlocs}')
    print(f'Number of time releases:{ntimereleases}')
    print(f'Number of particles adjusted to {npart} to be multiple of number of locations')

    start = {
        'lat': lats * ntimereleases,
        'lon': lons * ntimereleases,
        'time': times,
        'depth': [500] * npart,
    }

    # prepare arguments
    fieldset = prep_fieldset()
    runtime = delta(days=runtimedays)
    outputdt = delta(hours=outputdthours)

    pset = ParticleSet(
        fieldset=fieldset,
        pclass=ParticleClass,
        lon=start['lon'],
        lat=start['lat'],
        time=start['time'],
        depth=start['depth']
    )

    if get_mpi_rank(mpimode) == 0:
        print('** pset ini')
        print(pset[0])
        print(pset[-1])

    kernelobj = pset.Kernel(kernels[0])
    for i in range(1, len(kernels)):
        # casting the kernel function to a kernel object
        kernelobj += pset.Kernel(kernels[i])

    outputfile = pset.ParticleFile(
        name=tempfname,
        outputdt=outputdt,
        chunks=(chunksizes['trajectory'], chunksizes['obs'])
    )

    print('Executing run')
    tic = time.time()
    pset.execute(
        kernelobj,
        runtime=runtime,
        dt=dtminutes * 60,
        output_file=outputfile,
        verbose_progress=True,
        recovery={ErrorCode.ErrorThroughSurface: pkernels.SinkParticle,
                  ErrorCode.ErrorOutOfBounds: pkernels.RaiseParticle},
    )
    outputfile.close()
    exectime = time.time() - tic

    print(f'    time: {exectime:.0f} s mem: {mem_used_MB():.0f} MB')

    if mpimode:
        merge_zarrs(tempfname, outfname, chunksizes)


if __name__ == "__main__":
    main()
