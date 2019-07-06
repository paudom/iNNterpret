from __future__ import absolute_import

# -- IMPORT -- #
from colored import fg,attr

# -- VERBOSE -- #
class Verbose():
	""">> CLASS:VERBOSE: responsible for displaying messages"""
	def __init__(self,verbose=True):
		self.flag = verbose

	def print_msg(self,message):
		""">> PRINT_MSG: prints a message in a certain colour"""
		if self.flag:
			print(self.set_msg(message))

	def set_msg(self,message):
		""">> SET_MSG: returns a message in a certain colour."""
		msg = fg(75)+message+attr(0)
		return msg

	@property
	def switch_flag(self):
		""">> SWITCH_FLAG: change the verbosity level of the Toolbox."""
		if self.flag:
			self.flag = False
		else:
			self.flag = True
		
	@property
	def set_colour(self):
		""">> SET_COLOUR: sets the text colour."""
		print('%s' % fg(75))
	
	@property
	def reset_colour(self):
		""">> RESET_COLOUR: resets the text colour."""
		print('%s' % attr(0))
