"""Shared config for the current (post-revision) project. Minimal on purpose -- the
prior project's larger config.py is archived, not reused as-is.
"""

# Extended from the original "2y" after Phase 26's walk-forward backtest showed wild
# fold-to-fold instability (each market's accuracy swung 30-50 percentage points
# depending on the test window). "5y" (~1,256 rows/market) tightened that variance
# substantially but wasn't enough to properly power the 20-day-horizon rescue
# experiment (non-overlapping evaluation there left only ~3 independent test
# observations per market per fold). Extended again to "10y" (~2,511-2,512
# rows/market, verified real dense daily data, not Yahoo's sparse "max" range) to
# give the next walk-forward run a real chance at enough independent windows.
HISTORY_PERIOD = "10y"
