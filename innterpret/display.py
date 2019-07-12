from __future__ import absolute_import

# -- IMPORT -- #
from colored import fg,attr

# -- VERBOSE -- #
class Verbose():
	"""CLASS::VERBOSE:
		Description:
		---
		> Responsible for displaying messages
		Arguments:
		---
		>- verbose {bool} -- Flag to detemrine if the messages will be displayed. (default:{True})."""
	def __init__(self,verbose=True):
		self.flag = verbose

	def print_msg(self,message):
		"""METHOD::PRINT_MSG: prints a message in a certain colour.
			---
			Arguments:
			---
			>- message {string} -- message to display.
			Returns:
			---
			>- {NONE}"""
		if self.flag:
			print(self.set_msg(message))

	def set_msg(self,message):
		"""METHOD::SET_MSG:
			---
			Arguments:
			---
			>- message {string} -- Message to format.
			Returns:
			---
			>- {string} -- A message in a certain colour"""
		msg = fg(75)+message+attr(0)
		return msg

	@property
	def switch_flag(self):
		"""METHOD::SWITCH_FLAG: change the verbosity level of the Toolbox.
			---
			Returns:
			---
			>- {NONE}."""
		if self.flag:
			self.flag = False
		else:
			self.flag = True
		
	@property
	def set_colour(self):
		"""METHOD::SET_COLOUR: sets the text colour.
			---
			Returns:
			---
			>- {NONE}."""
		print('%s' % fg(75))
	
	@property
	def reset_colour(self):
		"""METHOD::RESET_COLOUR: resets the text colour.
			---
			Returns:
			---
			>- {NONE}"""
		print('%s' % attr(0))
