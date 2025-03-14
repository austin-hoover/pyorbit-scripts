"""Set particle macrosize and print results."""

import numpy as np

from orbit.core.bunch import Bunch
from orbit.utils.consts import mass_proton


# Setup
seed = 0
rng = np.random.default_rng(seed)

# Create distribution
size = 10
X = rng.normal(size=(size, 6))

# Create bunch
bunch = Bunch()
bunch.charge(1.0)
bunch.mass(mass_proton)
for i in range(size):
    x, xp, y, yp, z, de = X[i, :]
    bunch.addParticle(x, xp, y, yp, z, de)

# Set global macrosize
intensity = 1.00e14
macrosize = intensity / size
bunch.macroSize(macrosize)
bunch.dumpBunch()

# Add particle macrosize attribute (defaults to zero)
bunch.addPartAttr("macrosize")
bunch.dumpBunch()

# Update particle macrosize attributes
atribute_array_index = 0
for i in range(bunch.getSize()):
    macrosize_loc = macrosize * i
    bunch.partAttrValue("macrosize", i, atribute_array_index, macrosize_loc)
bunch.dumpBunch()

# Print new macrosizes
for i in range(bunch.getSize()):
    macrosize = bunch.partAttrValue("macrosize", i, atribute_array_index)
    print(f"i={i} macrosize={macrosize}")

