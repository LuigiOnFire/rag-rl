from typing import List, Dict, Tuple
import json

# Ideally we load the actual HotpotQA dataset (e.g. from huggingface or json)
# For this task, we will simulate a streamer with a small subset or dummy data
# as we don't have internet access to download the full dataset.

class HotpotQAStreamer:
    def __init__(self, subset_size=10):
        self.subset_size = subset_size
        # Simulated Data found in prompt or similar
        self.data = [
            {
                "question": "What is the capital of France?",
                "answer": "Paris",
                "supporting_facts": ["Paris is the capital and most populous city of France."],
                "context": [
                    ["Paris", "Paris is the capital and most populous city of France."],
                    ["France", "France is a country located primarily in Western Europe."],
                    ["Lyon", "Lyon is the third-largest city and second-largest urban area of France."],
                    ["Marseille", "Marseille is the second-largest city of France."]
                ]
            },
            {
                "question": "Who wrote 'To Kill a Mockingbird'?",
                "answer": "Harper Lee",
                "supporting_facts": ["To Kill a Mockingbird is a novel by Harper Lee published in 1960."],
                "context": [
                    ["To Kill a Mockingbird", "To Kill a Mockingbird is a novel by Harper Lee published in 1960."],
                    ["Harper Lee", "Nelle Harper Lee was an American novelist best known for her 1960 novel To Kill a Mockingbird."],
                    ["Truman Capote", "Truman Capote was an American novelist, screenwriter, playwright, and actor."],
                    ["Go Set a Watchman", "Go Set a Watchman is a novel by Harper Lee published on July 14, 2015."]
                ]
            },
             {
                "question": "Which planet is known as the Red Planet?",
                "answer": "Mars",
                "supporting_facts": ["Mars is often referred to as the 'Red Planet'."],
                "context": [
                    ["Mars", "Mars is the fourth planet from the Sun and the second-smallest planet in the Solar System."],
                    ["Red Planet", "Mars is often referred to as the 'Red Planet' due to the effect of the iron oxide prevalent on Mars's surface."],
                    ["Jupiter", "Jupiter is the fifth planet from the Sun and the largest in the Solar System."],
                    ["Venus", "Venus is the second planet from the Sun."]
                ]
            },
            {
                "question": "What is the boiling point of water at sea level?",
                "answer": "100 degrees Celsius",
                "supporting_facts": ["The boiling point of water is 100 °C (212 °F) at standard pressure."],
                "context": [
                    ["Water", "Water is a chemical substance with the chemical formula H2O."],
                    ["Boiling point", "The boiling point of a substance is the temperature at which the vapor pressure of a liquid equals the pressure surrounding the liquid."],
                    ["Standard pressure", "Standard pressure is defined as 101.325 kPa."],
                    ["Water properties", "The boiling point of water is 100 °C (212 °F) at standard pressure."]
                ]
            }
        ]
        
    def stream(self) -> List[Dict]:
        """Yields samples one by one."""
        for sample in self.data:
            yield sample

    def get_corpus(self, sample) -> List[str]:
        """
        Extracts a flat list of document strings from the sample's context context.
        HotpotQA context is [[title, [sentences]], ...]. 
        Our prompt says 'context': [['Title', 'Sentence...'], ...]
        We will flatten to strings like "Title: Sentence..."
        """
        docs = []
        for item in sample['context']:
            title = item[0]
            # Verify if item[1] is list or string (Hotpot vs our dummy)
            content = item[1]
            if isinstance(content, list):
                content = " ".join(content)
            docs.append(f"{title}: {content}")
        return docs
