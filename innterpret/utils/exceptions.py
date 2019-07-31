from __future__ import absolute_import

# -- IMPORTS -- #
import traceback
import sys

# -- MODULE EXCEPTIONS -- #
class OptionNotSupported(Exception):
    """EXCEPTION::OptionNotSupported.
        ---
        Cause:
        ---
        >- When the option selected is not supported."""
    def myexcepthook(dtype, value, tb):
        msg = ''.join(traceback.format_exception(dtype, value, tb))
        print(msg)
    sys.excepthook = myexcepthook

class EmptyDirectoryError(Exception):
    """EXCEPTION::EmptyDirectoryError.
        ---
        Cause:
        ---
        >- When directory does not contain any relevant files."""
    def myexcepthook(dtype, value, tb):
        msg = ''.join(traceback.format_exception(dtype, value, tb))
        print(msg)
    sys.excepthook = myexcepthook

class NotAConvLayerError(Exception):
    """EXCEPTION::NotAConvLayerError:
        ---
        Cause:
        ---
        >- When the specified layer is not a convolution layer."""
    def myexcepthook(dtype,value,tb):
        msg = ''.join(traceback.format_exception(dtype,value,tb))
        print(msg)
    sys.excepthook = myexcepthook

class LayerNotManagableError(Exception):
    """EXCEPTION::LayerNotHandleableError:
        ---
        Causes:
        ---
        >- If the layer encounter can't be handle it."""
    def myexcepthook(dtype,value,tb):
        msg = ''.join(traceback.format_exception(dtype,value,tb))
        print(msg)
    sys.excepthook = myexcepthook