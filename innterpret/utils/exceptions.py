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

class TensorNotValidException(Exception):
    """EXCEPTION::TensorNotValidException:
        ---
        Cause:
        ---
        >- When the specified layer is not a convolution layer."""
    def myexcepthook(dtype,value,tb):
        msg = ''.join(traceback.format_exception(dtype,value,tb))
        print(msg)
    sys.excepthook = myexcepthook

class InterfaceException(Exception):
    """EXCEPTION::InterfaceException:
        ---
        Causes:
        ---
        >- If the interface is not well implemented."""
    def myexcepthook(dtype,value,tb):
        msg = ''.join(traceback.format_exception(dtype,value,tb))
        print(msg)
    sys.excepthook = myexcepthook

class H5FileCorruptedError(Exception):
    """EXCEPTION::H5FileCorruptedError:
        ---
        Cause:
        ---
        >- If the h5 file containing model information is not correct."""
    def myexcepthook(dtype,value,tb):
        msg = ''.join(traceback.format_exception(dtype,value,tb))
        print(msg)
    sys.excepthook = myexcepthook