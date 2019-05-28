# -- GENERIC UTILITIES -- #
	
# -- IMPORTS -- #
import platform
import os

# >> CHECK_OS: Return the current Operating System
def check_os():
	return platform.system()

# >> CREATE_RESULT_DIRECTORY: creates a directory to save the results
def create_result_directory(dirName):
	cwd = os.getcwd()
	fullPath = cwd+os.sep+dirName
	if not os.path.isdir(fullPath):
		os.mkdir(fullPath)
