from __future__ import absolute_import

# -- IMPORT VERBOSITY -- #
from .display import Verbose

# -- INITIALIZE VERBOSE-- #
__verbose__ = Verbose(True)
__verbose__.set_colour
print('------------------')
print('Interpretability Toolbox.')
print('------------------')
print('Importing Modules...')

# -- IMPORT UTILITIES SUBPACKAGE -- #
from . import utils

# -- IMPORT METHOD SUBPACKAGES -- #
from . import surrogate
from . import function
from . import signal
from . import attribution
from . import localization
from . import distance

# -- REMOVE __PYCACHE__ -- #
import subprocess
print('Removing Cache...')
subprocess.call('find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf',shell=True)

# -- RESET COLOR -- #
print('========== DONE ==========')
__verbose__.reset_colour

# -- VERSION -- #
__version__ = '1.1.0'

# -- AUTHOR -- #
__author__ = 'Pau Domingo'

# -- ALL -- #
__all__ = ['attribution','distance','function','localization','signal','surrogate','utils']