from __future__ import absolute_import

# -- IMPORT PRINT_MSG -- #
from .display import Verbose

# -- INITIALIZE VERBOSE -- #
__verbose__ = Verbose(True)
__verbose__.set_colour
__verbose__.print_msg('Interpretability Toolbox.')

# -- IMPORT UTILITIES SUBPACKAGE -- #
#from . import utils

# -- IMPORT METHOD SUBPACKAGES -- #
#from . import surrogate
#from . import function
#from . import signal
#from . import attribution
#from . import localization
#from . import distance

# -- RESET COLOR -- #
__verbose__.reset_colour

# -- VERSION -- #
__version__ = '1.0.1'

# -- ALL -- #
__all__ = ['attribution','distance','function','localization','signal','surrogate','utils']