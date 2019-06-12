from __future__ import absolute_import

# -- IMPORT PRINT_MSG -- #
from .display import print_msg,set_colour,reset_colour

# -- MESSAGE -- #
print_msg('Interpretability Toolbox.')
set_colour()

# -- IMPORT UTILITIES SUBPACKAGE -- #
from . import utils

# -- IMPORT METHOD SUBPACKAGES -- #
from . import surrogate
from . import function
from . import signal
from . import attribution
from . import localization
from . import distance
reset_colour()

# -- VERSION -- #
__version__ = '1.0.0'

# -- ALL -- #
__all__ = ['attribution','datapoints','function','localization','signal','surrogate','utils']