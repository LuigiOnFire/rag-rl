import random
import logging
from typing import List, Dict, Any, Optional

from .hotpot import HotpotQAStreamer
from .musique import MusiqueStreamer
from .twowiki import TwoWikiStreamer

logger = logging.getLogger(__name__)

# A simple registry mapping string names to class objects
DATASET_REGISTRY = {
    "hotpot": HotpotQAStreamer,
    "musique": MusiqueStreamer,
    "twowiki": TwoWikiStreamer
}

class MixedStreamer:
    def __init__(self, 
        dataset_names: List[str], 
        weights: Optional[List[float]] = None,
        limit=None, 
        shuffle: bool = True,
        configs: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        self.configs = configs or {}
        self.streamers = []
        self.weights = weights
        self.shuffle = shuffle
        for name in dataset_names:
            ds_config = self.configs.get(name, {})
            
            if name not in DATASET_REGISTRY:
                logger.warning(f"Dataset '{name}' is not recognized.")
                continue
                
            streamer_class = DATASET_REGISTRY[name]
            streamer_instance = streamer_class(limit=limit, **ds_config)
            self.streamers.append(streamer_instance)

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
        active_iters = [s.stream(shuffle=self.shuffle) for s in self.streamers]
        
        if self.weights:
            active_weights = list(self.weights)
        else:
            active_weights = [1.0] * len(active_iters)
        
        while active_iters:
            # We must use random.choices which returns a list of 1 element
            current_iter = random.choices(active_iters, weights=active_weights, k=1)[0]
            try:
                yield next(current_iter)
            except StopIteration:
                idx = active_iters.index(current_iter)
                active_iters.pop(idx)
                active_weights.pop(idx)
