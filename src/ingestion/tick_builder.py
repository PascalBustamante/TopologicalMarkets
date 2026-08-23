
from src.ingestion.bar import Bar

class TickBarBuilder:

    def __init__(self, ticks_per_bar: int):

        self.ticks_per_bar = ticks_per_bar

        self.reset()