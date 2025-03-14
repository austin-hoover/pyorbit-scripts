import numpy as np
import scipy.optimize

from orbit.core.bunch import Bunch
from orbit.bunch_generators import TwissContainer
from orbit.bunch_generators import GaussDist2D
from orbit.bunch_generators import KVDist2D
from orbit.bunch_generators import WaterBagDist2D
from orbit.lattice import AccLattice
from orbit.lattice import AccNode
from orbit.teapot import teapot
from orbit.teapot import TEAPOT_Lattice
from orbit.teapot import TEAPOT_MATRIX_Lattice
from orbit.utils.consts import mass_proton


def split_node(node: AccNode, max_part_length: float = None) -> AccNode:
    """Split node into parts."""
    if max_part_length is not None and max_part_length > 0.0:
        if node.getLength() > max_part_length:
            node.setnParts(1 + int(node.getLength() / max_part_length))
    return node


def get_transfer_matrix(lattice: AccLattice, mass: float, kin_energy: float) -> np.ndarray:
    """Compute 4x4 transfer matrix from periodic lattice."""
    bunch = Bunch()
    bunch.mass(mass)
    bunch.getSyncParticle().kinEnergy(kin_energy)

    matrix_lattice = TEAPOT_MATRIX_Lattice(lattice, bunch)
    matrix = matrix_lattice.oneTurnMatrix

    M = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            M[i, j] = matrix.get(i, j)
    return M


def get_phase_advances(M: np.ndarray) -> np.ndarray:
    """Compute x/y phase advances from 4x4 transfer matrix."""
    return np.arccos(np.linalg.eigvals(M).real)[[0, 2]]


def make_lattice(
    mux: float,
    muy: float,
    length: float,
    mass: float,
    kin_energy: float,
    fill_factor: float = 0.5,
    angle: float = 0.0,
    start: str = "drift",
    fringe: bool = False,
    reverse: bool = False,
    verbose: bool = False,
) -> AccLattice:
    """Create FODO lattice with specified phase advances.

    Parameters
    ----------
    mux{y}: float
        The x{y} lattice phase advance [rad].
    length : float
        The length of the lattice [m].
    mass, kin_energy : float
        Mass [GeV/c^2] and kinetic energy [GeV] of synchronous particle.
    fill_fac : float
        The fraction of the lattice occupied by quadrupoles.
    angle : float
        The skew or tilt angle of the quads [deg]. The focusing quad is
        rotated clockwise and the defocusing quad is rotated counterclockwise.
    fringe : bool
        Whether to include nonlinear fringe fields in the lattice.
    start : str
        If 'drift', the lattice will be O-F-O-O-D-O. If 'quad' the lattice will
        be (F/2)-O-O-D-O-O-(F/2).
    reverse : bool
        If True, reverse the lattice elements.

    Returns
    -------
    TEAPOT_Lattice
    """
    angle = np.radians(angle)

    def make_lattice(k1: float, k2: float) -> AccLattice:
        """Create FODO lattice with specified focusing strengths.

        k1 and k2 are the focusing strengths of the
        focusing (1st) and defocusing (2nd) quads, respectively.
        """
        # Instantiate elements
        lattice = TEAPOT_Lattice()
        drift1 = teapot.DriftTEAPOT("drift1")
        drift2 = teapot.DriftTEAPOT("drift2")
        drift_half1 = teapot.DriftTEAPOT("drift_half1")
        drift_half2 = teapot.DriftTEAPOT("drift_half2")
        qf = teapot.QuadTEAPOT("qf")
        qd = teapot.QuadTEAPOT("qd")
        qf_half1 = teapot.QuadTEAPOT("qf_half1")
        qf_half2 = teapot.QuadTEAPOT("qf_half2")
        qd_half1 = teapot.QuadTEAPOT("qd_half1")
        qd_half2 = teapot.QuadTEAPOT("qd_half2")

        # Set lengths
        half_nodes = (drift_half1, drift_half2, qf_half1, qf_half2, qd_half1, qd_half2)
        full_nodes = (drift1, drift2, qf, qd)
        for node in half_nodes:
            node.setLength(length * fill_factor / 4.0)
        for node in full_nodes:
            node.setLength(length * fill_factor / 2.0)

        # Set quad focusing strengths
        for node in (qf, qf_half1, qf_half2):
            node.addParam("kq", +k1)
        for node in (qd, qd_half1, qd_half2):
            node.addParam("kq", -k2)

        # Create lattice
        if start == "drift":
            lattice.addNode(drift_half1)
            lattice.addNode(qf)
            lattice.addNode(drift2)
            lattice.addNode(qd)
            lattice.addNode(drift_half2)
        elif start == "quad":
            lattice.addNode(qf_half1)
            lattice.addNode(drift1)
            lattice.addNode(qd)
            lattice.addNode(drift2)
            lattice.addNode(qf_half2)

        if reverse:
            lattice.reverseOrder()

        for node in lattice.getNodes():
            node.setUsageFringeFieldIN(fringe)
            node.setUsageFringeFieldOUT(fringe)

        lattice.initialize()

        for node in lattice.getNodes():
            name = node.getName()
            if "qf" in name:
                node.setTiltAngle(+angle)
            elif "qd" in name:
                node.setTiltAngle(-angle)

        return lattice

    def loss_function(k: np.ndarray) -> float:
        (k1, k2) = k
        lattice = make_lattice(k1, k2)
        transfer_matrix = get_transfer_matrix(lattice, mass=mass, kin_energy=kin_energy)
        phase_adv_calc = get_phase_advances(transfer_matrix)
        phase_adv_targ = [mux, muy]
        loss = np.abs(np.subtract(phase_adv_calc, phase_adv_targ))
        return loss

    k0 = np.array([0.5, 0.5])  # ~ 80 deg phase advance
    result = scipy.optimize.least_squares(loss_function, k0, verbose=verbose)
    k1, k2 = result.x
    lattice = make_lattice(k1, k2)

    for node in lattice.getNodes():
        node = split_node(node, 0.05)

    if verbose:
        transfer_matrix = get_transfer_matrix(lattice, mass=mass, kin_energy=kin_energy)
        phase_adv_calc = get_phase_advances(transfer_matrix)
        phase_adv_targ = np.array([mux, muy])
        phase_adv_calc *= 180.0 / np.pi
        phase_adv_targ *= 180.0 / np.pi
        print(f"mux = {phase_adv_calc[0]} (target={phase_adv_targ[0]})")
        print(f"muy = {phase_adv_calc[1]} (target={phase_adv_targ[1]})")

    return lattice


def make_bunch(
    mass: float,
    kin_energy: float,
    alpha_x: float,
    alpha_y: float,
    beta_x: float,
    beta_y: float,
    eps_x: float,
    eps_y: float,
    length: float = 100.0,
    size: int = 10_000,
    seed: int = None,
    dist: str = "waterbag",
) -> Bunch:
    """Make 4D bunch with specified distribution and Twiss parameters."""
    bunch = Bunch()
    bunch.mass(mass)
    bunch.getSyncParticle().kinEnergy(kin_energy)

    twiss_x = TwissContainer(alpha_x, beta_x, eps_x)
    twiss_y = TwissContainer(alpha_y, beta_y, eps_y)

    dist_name = dist
    dist = None
    if dist_name == "waterbag":
        dist = WaterBagDist2D(twiss_x, twiss_y)
    if dist_name == "kv":
        dist = KVDist2D(twiss_x, twiss_y)
    if dist_name == "gaussian":
        dist = GaussDist2D(twiss_x, twiss_y)

    rng = np.random.default_rng(seed)
    for i in range(size):
        (x, xp, y, yp) = dist.getCoordinates()
        z = rng.uniform(-0.5 * length, 0.5 * length)
        de = 0.0
        bunch.addParticle(x, xp, y, yp, z, de)

    return bunch


def get_bunch_coords(bunch: Bunch) -> np.ndarray:
    """Extract [N, 6] phase space coordinate array from bunch."""
    size = bunch.getSize()
    X = np.zeros((size, 6))
    for i in range(size):
        X[i, 0] = bunch.x(i)
        X[i, 1] = bunch.xp(i)
        X[i, 2] = bunch.y(i)
        X[i, 3] = bunch.yp(i)
        X[i, 4] = bunch.z(i)
        X[i, 5] = bunch.dE(i)
    return X
