# Module for handling single unit activity data stored
# in NWB file format recorded with Open Ephys plugin-GUI.


class Cell(object):
    """
    Provides access to properties of an isolated single-unit, a cell.
    """

    def __init__(self, timestamps, lfp=None, posdata=None):
        # Parse input
        self._timestamps = timestamps
        self._lfp = lfp
        self._posdata = posdata
        # Prepare attributes
        self._autocorr = None

    def _compute_autocorr(self):
        pass

    @property
    def autocorr(self):
        if self._autocorr is None:
            self._compute_autocorr()
        return self._autocorr
