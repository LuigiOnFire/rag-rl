from .base import BaseStreamer

class MusiqueStreamer(BaseStreamer):
    def stream(self):
        # Musique has totally different JSON keys, but you map them 
        # to your standard format before yielding.
        yield {"question": "...", "answer": "...", "corpus": [...]}