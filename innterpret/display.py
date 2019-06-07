from __future__ import absolute_import

# -- IMPORT -- #
from colored import fg,attr

# >> PRINT_MSG: prints the messages of the toolbox in certain colors to make it more readable.
def print_msg(message,show=True,option='text'):
	if option == 'verbose':
		msg = fg(76)+message+attr(0)
	elif option == 'error':
		msg = fg(160)+message+attr(0)
	elif option == 'input':
		msg = fg(208)+message+attr(0)
	else:
		msg = fg(75)+message+attr(0)
	if show:
		print(msg)
	else:
		return msg

# >> SET_COLOUR: sets the text colour
def set_colour():
	print('%s------------------' % fg(75))

# >> RESET_COLOUR: resets the text colour
def reset_colour():
	print('------------------%s' % attr(0))
