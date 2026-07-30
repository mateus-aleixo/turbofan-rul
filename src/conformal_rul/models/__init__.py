"""Model zoo.

Output convention shared by every model, neural or boosted:
column 0 is the mean-regression head (used for point RUL and RMSE/score
benchmarks); columns 1..len(QUANTILES) are conditional quantiles in
``config.QUANTILES`` order (used by CQR). Keeping the layout identical lets
the conformal layer and the serving code stay model-agnostic.
"""

from conformal_rul.config import QUANTILES

OUTPUT_DIM = 1 + len(QUANTILES)
