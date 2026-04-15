# IMPORT LIBRARIES
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import eelbrain
import mne
import re
import os
import pandas as pd

# IMPORT MODULES
from pathlib import Path
from matplotlib.patches import ConnectionPatch, Patch
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib import colors

from scipy.stats import ttest_rel, pearsonr, ttest_1samp, zscore
from enum import Enum
