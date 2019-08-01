from __future__ import absolute_import

# -- IMPORTS -- #
import traceback
import sys

# -- MODULE EXCEPTIONS -- #
class InnterpretException(Exception):
    """EXCEPTION::InnterpretException:
        ---
        Description:
        ---
        >- Base class for all the exceptions exclusive for this module."""
    @staticmethod
    def myexcepthook(dtype,value,tb):
        print(''.join(traceback.format_exception(dtype,value,tb)))
    sys.excepthook = myexcepthook.__func__

class OptionNotSupported(InnterpretException):
    """EXCEPTION::OptionNotSupported.
        ---
        Cause:
        ---
        >- When the option selected is not supported."""
    pass

class EmptyDirectoryError(InnterpretException):
    """EXCEPTION::EmptyDirectoryError.
        ---
        Cause:
        ---
        >- When directory does not contain any relevant files."""
    pass

class TensorNotValidException(InnterpretException):
    """EXCEPTION::TensorNotValidException:
        ---
        Cause:
        ---
        >- When the specified layer is not a convolution layer."""
    pass

class InterfaceException(InnterpretException):
    """EXCEPTION::InterfaceException:
        ---
        Causes:
        ---
        >- If the interface is not well implemented."""
    pass

class H5FileCorruptedError(InnterpretException):
    """EXCEPTION::H5FileCorruptedError:
        ---
        Cause:
        ---
        >- If the h5 file containing model information is not correct."""
    pass