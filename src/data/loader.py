import random
from typing import List
from .hotpot import HotpotQAStreamer
from .musique import MusiqueStreamer
from typing import List, Dict, Any, Optional
# import others...

# A simple registry mapping string names to class objects
DATASET_REGISTRY = {
    "hotpot": HotpotQAStreamer,
    # "musique": MusiqueStreamer,
}

class MixedStreamer:
    def __init__(self, 
        dataset_names: List[str], 
        limit=None, 
        shuffle: bool = True,
        configs: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        # Initialize all requested streamers
        self.configs = configs or {}
        self.streamers = []
        self.shuffle = shuffle
        for name in dataset_names:
            # Safely grab the config for this specific dataset (defaults to {} if not found)
            ds_config = self.configs.get(name, {})
            
            if name == "hotpot":
                # **ds_config unpacks the dictionary directly into the class arguments!
                self.streamers.append(HotpotQAStreamer(limit=limit, **ds_config))
                
            elif name == "musique":
                # When you eventually build MuSiQue, it instantly supports configs too
                # self.streamers.append(MusiqueStreamer(limit=limit, **ds_config))
                pass
                
            else:
                logger.warning(f"Dataset '{name}' is not recognized.")
            
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
        """Randomly yields samples from the active iterators.
        
        Creates fresh iterators on every call so that the training loop can
        safely restart by calling streamer.stream() again after exhaustion.
        Each call re-shuffles the underlying datasets when shuffle=True.
        """
        # Create new generators from the underlying datasets each time — this
        # is the key detail: copying self.iterators would just copy spent
        # generators and produce an immediately-exhausted stream on restart.
        active_iters = [s.stream(shuffle=self.shuffle) for s in self.streamers]
        
        while active_iters:
            # Pick a random dataset iterator
            current_iter = random.choice(active_iters)
            try:
                yield next(current_iter)
            except StopIteration:
                # If this dataset runs out, remove it and continue with the rest
                active_iters.remove(current_iter)