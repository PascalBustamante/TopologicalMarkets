from enum import Enum

from src.ingestion.bar.base import BarBuilder
from src.ingestion.bar.tick_builder import TickBarBuilder
from src.ingestion.bar.volume_builder import VolumeBarBuilder
from src.ingestion.bar.dollar_builder import DollarBarBuilder


class SamplingMethod(Enum):
    TICK = "tick"
    VOLUME = "volume"
    DOLLAR = "dollar"


_BUILDERS: dict[SamplingMethod, type[BarBuilder]] = {
    SamplingMethod.TICK: TickBarBuilder,
    SamplingMethod.VOLUME: VolumeBarBuilder,
    SamplingMethod.DOLLAR: DollarBarBuilder,
}


def make_bar_builder(method: SamplingMethod | str, threshold: float) -> BarBuilder:
    """Construct the BarBuilder for a given sampling method.

    `method` may be a SamplingMethod or its string value (e.g. "volume"),
    so it can come straight from config/CLI input.
    """
    if isinstance(method, str):
        method = SamplingMethod(method)

    return _BUILDERS[method](threshold)
