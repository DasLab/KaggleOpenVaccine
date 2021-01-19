import sys
sys.path.append('/home/tunguz/arnie')
sys.path.append('/home/tunguz')

import numpy as np
import re
from arnie.pfunc import pfunc
from arnie.free_energy import free_energy
from arnie.bpps import bpps
from arnie.mfe import mfe
import arnie.utils as utils
from decimal import Decimal
import ipynb
from xgboost import XGBRegressor
import xgboost as xgb
print(xgb.__version__)