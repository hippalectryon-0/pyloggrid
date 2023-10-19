"""used to package the source and handle compilation"""

import os
import pathlib
import sys

from setuptools import Extension
from setuptools.command.build_ext import build_ext as build_ext_orig


class MakefileExtension(Extension):
    def __init__(self, name):
        # don't invoke the original build_ext for this special extension
        super().__init__(name, sources=[])


class build_ext(build_ext_orig):
    def run(self):
        for ext in self.extensions:
            self.build_makefile(ext)
        super().run()

    def build_makefile(self, ext):
        cwd = pathlib.Path().absolute()
        # this seems necesssary to keep the files installed ?
        extdir = pathlib.Path(self.build_lib) / ext.name
        makefile_dir = str(extdir.parent.absolute())
        # required otherwise whl build errors
        extdir = pathlib.Path(self.get_ext_fullpath(ext.name))
        extdir.mkdir(parents=True, exist_ok=True)

        try:
            os.chdir(makefile_dir)
            if sys.platform == "win32":
                self.spawn(["make", "-f", "Makefile.windows"])
            else:
                self.spawn(["make"])

            print("Module compilation completed successfully.")
        except Exception as e:
            print(f"Error compiling modules: {str(e)}")
        finally:
            # Change back to the original working directory
            os.chdir(str(cwd))


def build(setup_kwargs: dict):
    setup_kwargs |= dict(
        cmdclass={"build_ext": build_ext},
        ext_modules=[MakefileExtension("pyloggrid/LogGrid/convolver")],
    )
