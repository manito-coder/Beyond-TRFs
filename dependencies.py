# IMPORT LIBRARIES
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import eelbrain
import mne
import re
import os
import pandas as pd
import copy

# IMPORT MODULES
from pathlib import Path
from matplotlib.patches import ConnectionPatch, Patch
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.gridspec as gridspec
from matplotlib import colors
from tqdm import tqdm


from scipy.stats import ttest_rel, pearsonr, ttest_1samp, zscore
from scipy.optimize import linear_sum_assignment
from enum import Enum


