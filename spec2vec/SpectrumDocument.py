from .Document import Document


class SpectrumDocument(Document):
    """Create documents from spectra.

    Every peak (and loss) positions (m/z value) will be converted into a string "word".
    The entire list of all peak words forms a spectrum document. Peak words have
    the form "peak@100.3" (for n_decimals=1), and losses have the format "loss@100.3".
    Peaks with identical resulting strings will not be merged, hence same words can
    exist multiple times in a document (e.g. peaks at 100.31 and 100.29 would lead to
    two words "peak@100.3" when using n_decimals=1).
    """
    def __init__(self, spectrum, n_decimals: int = 1):
        """

        Parameters
        ----------
        spectrum: SpectrumType
            Input spectrum.
        n_decimals
            Peak positions are converted to strings with n_decimal decimals.
            The default is 1, which would convert a peak at 100.381 into the
            word "peak@100.4".
        """
        self.n_decimals = n_decimals
        self.weights = None
        super().__init__(obj=spectrum)
        self._add_weights()

    def _make_words(self):
        """Create word from peaks (and losses)."""
        format_string = "{}@{:." + "{}".format(self.n_decimals) + "f}"
        peak_words = [format_string.format("peak", mz) for mz in self._obj.peaks.mz]
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
        return self._obj._metadata.copy().get(key, default)

    @property
    def metadata(self):
        """Return metadata of original spectrum."""
        return self._obj._metadata.copy()

    @property
    def losses(self) -> Optional[Spikes]:
        """Return losses of original spectrum."""
        return self._obj._losses.clone() if self._losses is not None else None

    @property
    def peaks(self) -> Spikes:
        """Return peaks of original spectrum."""
        return self._obj._peaks.clone()
