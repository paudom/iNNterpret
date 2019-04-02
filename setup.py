## -- IMPORTS -- ##
from setuptools import setup, find_packages
from distutils.cmd import Command
import os

with open('README.md') as file:
	readme = file.read()
with open('LICENSE') as file:
	license = file.read()

class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info')
        
setup(
	name = 'iNNterpret',
	version = '1.0.0',
	description = 'Interpretability Toolbox',
	long_description = readme,
	author = 'Pau Domingo',
	license = license,
	url = 'https://github.com/paudom/iNNterpret.git',
	packages = find_packages(),
    cmdclass={'clean': CleanCommand}
)