from __future__ import absolute_import

# -- IMPORT PRINT_MSG -- #
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

# -- RESET COLOR -- #
print('========== DONE ==========')
__verbose__.reset_colour

# -- VERSION -- #
__version__ = '1.1.0'

# -- AUTHOR -- #
__author__ = 'Pau Domingo'

# -- COPYRIGHT -- #
__license__ = """Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""

# -- ALL -- #
__all__ = ['attribution','distance','function','localization','signal','surrogate','utils']