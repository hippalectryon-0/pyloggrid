import os
import pathlib
import subprocess
import sys

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, _version, _build_data):
        if "READTHEDOCS_VIRTUALENV_PATH" in os.environ:
            return
        try:
            os.chdir(pathlib.Path("pyloggrid/LogGrid"))
            if sys.platform == "win32":
                subprocess.check_call(["make", "clean", "-f", "Makefile.windows"])
                subprocess.check_call(["make", "-f", "Makefile.windows"])
            else:
                subprocess.check_call(["make", "clean"])
                subprocess.check_call(["make"])

            print("Module compilation completed successfully.")
        except Exception as e:
            print(f"Error compiling modules: {str(e)}")
            raise Exception("Error compiling modules.")
