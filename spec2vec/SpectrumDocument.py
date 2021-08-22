from typing import Optional
from matchms.Spikes import Spikes
from .Document import Document


class SpectrumDocument(Document):
    """Create documents from spectra.

    Every peak (and loss) positions (m/z value) will be converted into a string "word".
    The entire list of all peak words forms a spectrum document. Peak words have
    the form "peak@100.32" (for n_decimals=2), and losses have the format "loss@100.32".
    Peaks with identical resulting strings will not be merged, hence same words can
    exist multiple times in a document (e.g. peaks at 100.31 and 100.29 would lead to
    two words "peak@100.3" when using n_decimals=1).

    For example:

    .. testcode::

        import numpy as np
        from matchms import Spectrum
        from spec2vec import SpectrumDocument

        spectrum = Spectrum(mz=np.array([100.0, 150.0, 200.51]),
                            intensities=np.array([0.7, 0.2, 0.1]),
                            metadata={'compound_name': 'substance1'})
        spectrum_document = SpectrumDocument(spectrum, n_decimals=1)

        print(spectrum_document.words)
        print(spectrum_document.peaks.mz)
        print(spectrum_document.get("compound_name"))

    Should output

    .. testoutput::

        ['peak@100.0', 'peak@150.0', 'peak@200.5']
        [100.   150.   200.51]
        substance1
    """
    def __init__(self, spectrum, n_decimals: int = 2,
                 mz_from: float = 0.0, mz_to: float = 1000.0):
        """

        Parameters
        ----------
        spectrum: SpectrumType
            Input spectrum.
        n_decimals
            Peak positions are converted to strings with n_decimal decimals.
            The default is 2, which would convert a peak at 100.387 into the
            word "peak@100.39".
        mz_from:
            Set lower threshold for m/z values to take into account.
            Default is 0.0.
        mz_to:
            Set upper threshold for m/z values to take into account.
            Default is 1000.0.
        """
        self.n_decimals = n_decimals
        self.mz_from = mz_from
        self.mz_to = mz_to
        self.weights = None
        super().__init__(obj=spectrum)
        self._add_weights()

    def _make_words(self):
        """Create word from peaks (and losses)."""
        mz_array_selected = self._obj.peaks.mz[(self._obj.peaks.mz >= self.mz_from) & (self._obj.peaks.mz <= self.mz_to)]
        format_string = "{}@{:." + "{}".format(self.n_decimals) + "f}"
        peak_words = [format_string.format("peak", mz) for mz in mz_array_selected]
        if self._obj.losses is not None:
            loss_words = [format_string.format("loss", mz) for mz in self._obj.losses.mz]
        else:
            loss_words = []
        self.words = peak_words + loss_words
        return self

    def _add_weights(self):
        """Add peaks (and loss) intensities as weights."""
        assert self._obj.peaks.intensities.max() <= 1, "peak intensities not normalized"

        peak_intensities = self._obj.peaks.intensities.tolist()
        if self._obj.losses is not None:
            loss_intensities = self._obj.losses.intensities.tolist()
        else:
            loss_intensities = []
        self.weights = peak_intensities + loss_intensities
        return self

    def get(self, key: str, default=None):
        """Retrieve value from Spectrum metadata dict. Shorthand for

        .. code-block:: python

            val = self._obj.metadata[key]

        """
        assert not hasattr(self, key), "Key cannot be attribute of SpectrumDocument class"
        return self._obj.get(key, default)

    @property
    def metadata(self):
        """Return metadata of original spectrum."""
        return self._obj.metadata

    @property
    def losses(self) -> Optional[Spikes]:
        """Return losses of original spectrum."""
        return self._obj.losses

    @property
    def peaks(self) -> Spikes:
        """Return peaks of original spectrum."""
        return self._obj.peaks
