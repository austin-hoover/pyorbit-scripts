"""Track bunch with space charge + test particles in FODO lattice.

Test particles have macrosize=0, so they contribute nothing to the charge density
but still respond to the beam's electric field.

Example usage:
    python track_test_particles.py --test-mismatch=5.0
"""

import argparse
import os
import pathlib

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm import trange

from orbit.core import orbit_mpi
from orbit.core.bunch import Bunch
from orbit.core.spacecharge import SpaceChargeCalc2p5D
from orbit.lattice import AccLattice
from orbit.teapot import TEAPOT_Lattice
from orbit.teapot import TEAPOT_MATRIX_Lattice
from orbit.space_charge.sc2p5d.scLatticeModifications import setSC2p5DAccNodes
from orbit.utils.consts import mass_proton

from tools import make_bunch
from tools import make_lattice
from tools import get_bunch_coords


# Arguments
# --------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--size", type=int, default=100_000, help="bunch size (number of macroparticles)")
parser.add_argument("--dist", type=str, default="waterbag", choices=["kv", "waterbag", "gaussian"])
parser.add_argument("--intensity", type=float, default=50.0, help="bunch intensity / 1e14")
parser.add_argument("--kin-energy", type=float, default=1.000, help="bunch kinetic energy [GeV]")
parser.add_argument("--test-size", type=int, default=20_000, help="number of test particles")
parser.add_argument("--test-mismatch", type=float, default=0.75, help="mismatch test particles (0=matched)")
parser.add_argument("--phase-advance", type=float, default=110.0, help="lattice phase advance")
parser.add_argument("--periods", type=int, default=10, help="number of lattice periods to track")
parser.add_argument("--seed", type=int, default=1234, help="random seed")
args = parser.parse_args()


# Setup
# --------------------------------------------------------------------------------------

path = pathlib.Path(__file__)
output_dir = os.path.join("outputs", path.stem)
os.makedirs(output_dir, exist_ok=True)

rng = np.random.default_rng(args.seed)


# Create lattice
# --------------------------------------------------------------------------------------

# Create FODO lattice with specified phase advance.
lattice = make_lattice(
    mux=np.radians(args.phase_advance),
    muy=np.radians(args.phase_advance),
    length=5.0,
    mass=mass_proton,
    kin_energy=args.kin_energy,
    start="quad",
    verbose=2,
)

# Calculate periodic lattice parameters (alpha, beta).
bunch = Bunch()
bunch.mass(mass_proton)
bunch.getSyncParticle().kinEnergy(args.kin_energy)
matrix_lattice = TEAPOT_MATRIX_Lattice(lattice, bunch)
lattice_params = matrix_lattice.getRingParametersDict()

alpha_x = lattice_params["alpha x"]
alpha_y = lattice_params["alpha y"]
beta_x = lattice_params["beta x [m]"]
beta_y = lattice_params["beta y [m]"]

# Add 2.5D space charge nodes.
sc_calc = SpaceChargeCalc2p5D(64, 64, 1)
sc_path_length_min = 1.00e-06
sc_nodes = setSC2p5DAccNodes(lattice, sc_path_length_min, sc_calc)
if not args.intensity:
    for node in sc_nodes:
        node.switcher = False


# Create bunch
# --------------------------------------------------------------------------------------

# Generate bunch matched to bare lattice (zero space charge)
make_bunch_kws = {
    "mass": mass_proton,
    "kin_energy": args.kin_energy,
    "alpha_x": alpha_x,
    "alpha_y": alpha_y,
    "beta_x": beta_x,
    "beta_y": beta_y,
    "eps_x": 20.0e-06,
    "eps_y": 20.0e-06,
    "length": 100.0,
    "dist": args.dist,
    "seed": args.seed,
} 
bunch = make_bunch(size=args.size, **make_bunch_kws)

# Set the global macrosize. This is a bunch-level parameter. Later we will set
# particle parameters to change individual particle macrosizes. The Grid classes
# in all space charge nodes will check if the individual macrosize parameter exists,
# and if so, it will override the bunch-level parameter.
intensity = args.intensity * 1.00e14
macrosize = intensity / bunch.getSize()
bunch.macroSize(macrosize)


# Track bunch
# --------------------------------------------------------------------------------------

bunch_out = Bunch()
bunch.copyBunchTo(bunch_out)
for _ in trange(args.periods):
    lattice.trackBunch(bunch_out)

bunch_out.dumpBunch(os.path.join(output_dir, "bunch_01.dat"))
    
x1 = get_bunch_coords(bunch_out) * 1000.0


# Track bunch with test particles
# --------------------------------------------------------------------------------------

# Create test bunch. If args.test_match=True, we generate the same distrbiution as
# the main bunch. This means the test bunch should have the same evolution as the
# main bunch. Otherwise, we mismatch the initial Twiss parameters.
if args.test_mismatch:
    make_bunch_kws["alpha_x"] = +0.75 * args.test_mismatch
    make_bunch_kws["alpha_y"] = -0.75 * args.test_mismatch

bunch_test = make_bunch(size=args.test_size, **make_bunch_kws)

# Add test particles to bunch.
for i in range(bunch_test.getSize()):
    x = bunch_test.x(i)
    y = bunch_test.y(i)
    z = bunch_test.z(i)
    xp = bunch_test.xp(i)
    yp = bunch_test.yp(i)
    de = bunch_test.dE(i)
    bunch.addParticle(x, xp, y, yp, z, de)

# Set macrosize=0 for all test particles.
bunch.addPartAttr("macrosize")  # sets macrosize=0 for all particles!
attribute_array_index = 0
for i in range(bunch.getSize()):
    macrosize_loc = macrosize
    if i >= bunch.getSize() - args.test_size:
        macrosize_loc = 0.0
    bunch.partAttrValue("macrosize", i, attribute_array_index, macrosize_loc)

# Print particle macrosizes.
for i in range(bunch.getSize()):
    macrosize = bunch.partAttrValue("macrosize", i, attribute_array_index)
    print(f"i={i} macrosize={macrosize}")

# Track the new bunch.
bunch_out = Bunch()
bunch.copyBunchTo(bunch_out)
for _ in trange(args.periods):
    lattice.trackBunch(bunch_out)

bunch_out.dumpBunch(os.path.join(output_dir, "bunch_02.dat"))

x2 = get_bunch_coords(bunch_out) * 1000.0


# Analysis
# --------------------------------------------------------------------------------------

n = x1.shape[0]
axis = (0, 1)
bins = 64

xmax = np.max(x1[:, axis], axis=0)
xmax = xmax * 1.5
limits = list(zip(-xmax, xmax))

fig, axs = plt.subplots(ncols=2, figsize=(7.0, 3.0), constrained_layout=True)
for (ax, x) in zip(axs, [x1, x2]):
    ax.hist2d(
        x[:n, axis[0]], 
        x[:n, axis[1]],
        bins=bins,
        range=limits,
        cmap="Blues",
    )
axs[1].scatter(x2[n:, axis[0]], x2[n:, axis[1]], c="red", s=0.1, ec="none")
axs[0].set_title("Bunch")
axs[1].set_title("Bunch (custom macrosizes)")
plt.savefig(os.path.join(output_dir, "fig_compare.png"), dpi=300)
plt.show()

