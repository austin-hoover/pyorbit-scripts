import argparse
import math
import pathlib
import sys
import time

import numpy as np

from orbit.core import orbit_mpi
from orbit.core.bunch import Bunch
from orbit.core.bunch import BunchTwissAnalysis
from orbit.core.spacecharge import SpaceChargeCalc3D
from orbit.bunch_generators import GaussDist3D
from orbit.bunch_generators import TwissContainer
from orbit.lattice import AccActionsContainer
from orbit.space_charge.sc3d import setSC3DAccNodes
from orbit.teapot import TEAPOT_Lattice
from orbit.teapot import DriftTEAPOT
from orbit.utils.consts import charge_electron
from orbit.utils.consts import mass_proton

from orbit_tools.bunch import get_bunch_cov


# Setup
# --------------------------------------------------------------------------------------

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--nparts", type=int, default=100_000)
parser.add_argument("--mass", type=float, default=mass_proton)
parser.add_argument("--current", type=float, default=0.050)
parser.add_argument("--kin-energy", type=float, default=0.0025)

parser.add_argument("--scale-x", type=float, default=0.001)
parser.add_argument("--scale-y", type=float, default=0.001)
parser.add_argument("--scale-z", type=float, default=0.001)

parser.add_argument("--grid-x", type=int, default=64)
parser.add_argument("--grid-y", type=int, default=64)
parser.add_argument("--grid-z", type=int, default=64)

parser.add_argument("--distance", type=float, default=1.000)
parser.add_argument("--delta-s", type=float, default=0.001)

parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--save", type=int, default=0)
args = parser.parse_args()


# MPI
_mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
_mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
_mpi_size = orbit_mpi.MPI_Comm_size(_mpi_comm)


# Create output directory
path = pathlib.Path(__file__)
if _mpi_rank  == 0:
    if args.save:
        output_dir = os.path.join("outputs", path.stem)
        os.makedirs(output_dir, exist_ok=True)
        

# Lattice
# --------------------------------------------------------------------------------------

# Create a drift lattice with required step size
lattice = TEAPOT_Lattice()
for _ in range(int(args.distance / args.delta_s)):
    node = DriftTEAPOT()
    node.setLength(args.delta_s)
    lattice.addNode(node)
lattice.initialize()

# Add 3D space charge nodes
sc_calc = SpaceChargeCalc3D(args.grid_x, args.grid_y, args.grid_z)
sc_nodes = setSC3DAccNodes(lattice, args.delta_s, sc_calc)


# Bunch
# --------------------------------------------------------------------------------------

# Create empty bunch
bunch = Bunch()
bunch.mass(args.mass)
bunch.getSyncParticle().kinEnergy(args.kin_energy)

# Add particles to bunch
rng = np.random.default_rng(args.seed)
for _ in range(int(args.nparts / _mpi_size)):
    x = rng.normal(scale=args.scale_x)
    y = rng.normal(scale=args.scale_y)
    z = rng.normal(scale=args.scale_z)
    xp = 0.0
    yp = 0.0
    de = 0.0
    bunch.addParticle(x, xp, y, yp, z, de)
    
# Set bunch macrosize
frequency = 402.5e6  # [Hz]
charge = args.current / frequency
intensity = charge / (abs(bunch.charge()) * charge_electron)

size_global = bunch.getSizeGlobal()
macro_size = intensity / size_global
bunch.macroSize(macro_size)


# Tracking
# --------------------------------------------------------------------------------------

class Monitor:
    """Monitors bunch size during simulation."""
    def __init__(self) -> None:
        self.start_time = None

    def __call__(self, params_dict: dict) -> None:   
        # Update parameters
        if self.start_time is None:
            self.start_time = time.time()

        time_ellapsed = time.time() - self.start_time
        position = params_dict["path_length"]

        # Get bunch
        bunch = params_dict["bunch"]
        bunch_twiss_analysis = BunchTwissAnalysis()

        # Compute covariance matrix
        cov_matrix = get_bunch_cov(bunch)
        x_rms = np.sqrt(cov_matrix[0, 0]) * 1000.0
        y_rms = np.sqrt(cov_matrix[2, 2]) * 1000.0
        z_rms = np.sqrt(cov_matrix[4, 4]) * 1000.0

        # Print update
        if _mpi_rank == 0:
            print(
                "time={:.3f} s={:.3f} xrms={:.3f} yrms={:.3f} zrms={:.3f}"
                .format(time_ellapsed, position, x_rms, y_rms, z_rms)
            )
            sys.stdout.flush()

        
monitor = Monitor()
action_container = AccActionsContainer()
action_container.addAction(monitor, AccActionsContainer.EXIT)

if (_mpi_rank == 0) and args.save:
    bunch.dumpBunch(os.path.join(output_dir, "bunch_00.dat"))

lattice.trackBunch(bunch, actionContainer=action_container)

if (_mpi_rank == 0) and args.save:
    bunch.dumpBunch(os.path.join(output_dir, "bunch_01.dat"))
