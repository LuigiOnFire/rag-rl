import random
from typing import List
from .hotpot import HotpotQAStreamer
from .musique import MusiqueStreamer
# import others...

# A simple registry mapping string names to class objects
DATASET_REGISTRY = {
    "hotpot": HotpotQAStreamer,
    # "musique": MusiqueStreamer,
}

class MixedStreamer:
    def __init__(self, dataset_names: List[str], split: str = "train", limit: int = 0):
        # Initialize all requested streamers
        self.streamers = []
        self.iterators = []
        for name in dataset_names:
            if name not in DATASET_REGISTRY:
                raise ValueError(f"Unknown dataset: {name}")
            
            streamer_class = DATASET_REGISTRY[name]
            streamer_instance = streamer_class(split=split, limit=limit)
            self.streamers.append(streamer_instance)
            self.iterators.append(streamer_instance.stream())

    @property
    def total_available(self) -> int:
        """Sum of full (pre-limit) dataset sizes across all active streamers."""
        return sum(getattr(s, "total_size", len(s.dataset)) for s in self.streamers)

    @property
    def n_limit(self) -> int:
        """Total number of samples that will actually be streamed (post-limit)."""
        return sum(len(s.dataset) for s in self.streamers)

    def stream(self):
        """Randomly yields samples from the active iterators."""
        active_iters = self.iterators.copy()
        
        while active_iters:
            # Pick a random dataset iterator
            current_iter = random.choice(active_iters)
            try:
                yield next(current_iter)
            except StopIteration:
                # If this dataset runs out, remove it and continue with the rest
                active_iters.remove(current_iter)