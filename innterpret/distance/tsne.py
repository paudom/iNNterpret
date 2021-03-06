from __future__ import absolute_import

# -- IMPORT -- #
from ..utils.interfaces import Method

class TSNEDistance(Method):
    """CLASS::TSNEDistance:
        ---
        Description:
        ---
        > Method that computes a distribution of the data
        Arguments:
        ---
        Link:
        ---
        >- https://lvdmaaten.github.io/publications/papers/JMLR_2008.pdf."""
    def __init__(self):
        raise NotImplementedError
    def interpret(self):
        """METHOD::INTERPRET:
           ---
           Arguments:
           ---
           Returns:
           --- """
        pass
    
    def __repr__(self):
		return super().__repr__()+self.__class__.__name__+'>'