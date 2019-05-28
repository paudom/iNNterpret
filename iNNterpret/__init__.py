from __future__ import absolute_import

# -- MESSAGE -- #
print('Interpretability Toolbox.')

# -- IMPORT METHOD SUBPACKAGES -- #
from . import attribution
from . import datapoints
from . import function
from . import localization
from . import signal

# -- IMPORT UTILITIES SUBPACKAGE -- #
from . import utils

# -- VERSION -- #
__version__ = '1.0.0'

# -- ALL -- #
__all__ = ['attribution','datapoints','function','localization','signal','utils']