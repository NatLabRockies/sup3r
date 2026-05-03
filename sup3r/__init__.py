# isort: skip_file
"""Super Resolving Renewable Energy Resource Data (SUP3R)"""

import os
import numpy as np
import dask
import h5netcdf
import pandas as pd
import phygnn
import rex
import sklearn
import tensorflow as tf
import xarray
import sys

from ._version import __version__

# Next import sets up CLI commands
# This line could be "import sup3r.cli" but that breaks sphinx as of 12/11/2023
from .cli import main

__author__ = """Brandon Benton"""
__email__ = 'brandon.benton@nlr.gov'

SUP3R_DIR = os.path.dirname(os.path.realpath(__file__))
CONFIG_DIR = os.path.join(SUP3R_DIR, 'configs')
TEST_DATA_DIR = os.path.join(os.path.dirname(SUP3R_DIR), 'tests', 'data')
