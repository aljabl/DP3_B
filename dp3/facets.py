"""Which similarity facets apply to which stem."""

from . import config


class FacetPolicy:
    """
    Single source of truth for the stem -> facets mapping.

    Feature extraction, the index build, search and the interface all ask this
    class, so the rules live in one place. Facets that are mapped but not yet
    implemented are left out.
    """

    def __init__(self, facets_by_stem: dict | None = None, implemented: tuple | None = None):
        self.facets_by_stem = config.FACETS_BY_STEM if facets_by_stem is None else facets_by_stem
        self.implemented = config.IMPLEMENTED_FACETS if implemented is None else implemented

    def facets_for(self, stem: str) -> tuple[str, ...]:
        """Implemented facets for this stem. Raises ValueError for an unknown stem."""

        try:
            wanted = self.facets_by_stem[stem]
        except KeyError:
            raise ValueError(f"No facets are defined for stem '{stem}'.") from None

        return tuple(facet for facet in wanted if facet in self.implemented)

    def indexed_facets(self) -> tuple[str, ...]:
        """Every facet stored in the saved index, whatever the stem."""

        return self.implemented
