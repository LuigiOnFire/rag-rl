import random
import logging
from typing import List, Dict, Any, Optional

from .hotpot import HotpotQAStreamer
from .musique import MusiqueStreamer
from .twowiki import TwoWikiStreamer
from .squad import SQuADStreamer
from .nq import NQStreamer

logger = logging.getLogger(__name__)

# A simple registry mapping string names to class objects
DATASET_REGISTRY = {
    "hotpot": HotpotQAStreamer,
    "musique": MusiqueStreamer,
    "twowiki": TwoWikiStreamer,
    "squad": SQuADStreamer,
    "nq": NQStreamer
}

class MixedStreamer:
    def __init__(self, 
        dataset_names: List[str], 
        weights: Optional[List[float]] = None,
        limit=None, 
        shuffle: bool = True,
        seed: Optional[int] = None,
        configs: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        self.configs = configs or {}
        self.streamers = []
        self.dataset_names = []
        self.weights = weights
        self.shuffle = shuffle
        self.seed = seed
        self.rng = random.Random(seed) if seed is not None else random

        active_weights = []
        for position, name in enumerate(dataset_names):
            ds_config = self.configs.get(name, {})
            
            if name not in DATASET_REGISTRY:
                logger.warning(f"Dataset '{name}' is not recognized.")
                continue
                
            streamer_class = DATASET_REGISTRY[name]
            streamer_instance = streamer_class(limit=limit, **ds_config)
            self.streamers.append(streamer_instance)
            self.dataset_names.append(name)
            if weights:
                active_weights.append(weights[position] if position < len(weights) else 1.0)
        if weights:
            self.weights = active_weights

    @property
    def total_available(self) -> int:
        return sum(getattr(s, "total_size", len(s.dataset)) for s in self.streamers)

    @property
    def n_limit(self) -> int:
        return sum(len(s.dataset) for s in self.streamers)

    def stream(self):
        """Yields samples from active iterators, deterministically if seed was set."""
        # Pass both shuffle and seed down to individual streamers
        active_names = list(self.dataset_names)
        active_iters = [
            s.stream(shuffle=self.shuffle, seed=self.seed) 
            for s in self.streamers
        ]
        
        if self.weights:
            active_weights = list(self.weights)
        else:
            active_weights = [1.0] * len(active_iters)
        
        while active_iters:
            # Use self.rng to prevent polluting or being polluted by global random state
            current_iter = self.rng.choices(active_iters, weights=active_weights, k=1)[0]
            idx = active_iters.index(current_iter)
            try:
                item = next(current_iter)
                item.setdefault("dataset", active_names[idx])
                yield item
            except StopIteration:
                idx = active_iters.index(current_iter)
                active_iters.pop(idx)
                active_weights.pop(idx)
                active_names.pop(idx)
