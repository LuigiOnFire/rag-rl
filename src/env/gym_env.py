import gymnasium as gym
import re
from typing import Any, Tuple, TypedDict, cast, Dict

# Project Imports
from src.env.state import create_initial_state, GreenState
from src.env.retriever import EphemeralRetriever
from src.env.engine import GreenEngine
from src.rl.rewards import calculate_reward
from src.agent.prompts import format_state_for_prompt
# We can import parse_action logic or define it here. 
# Since it was inline in the PPO script previously, we define it here for encapsulation.

class GreenRAGEnv(gym.Env):
    def __init__(self, streamer):
        super().__init__()
        self.streamer = streamer
        self.data_iterator = streamer.stream()
        self.current_sample = None
        self.state = None
        self.retriever = None
        self.ground_truth = None
        
        # Define spaces if needed (Text based environment, spaces are formal)
        # Using simple discrete action space is misleading as we generate text.
        # So we leave observation/action space as "Text" type (informational).
        try:
            self.observation_space = gym.spaces.Text(max_length=4096)
            self.action_space = gym.spaces.Text(max_length=512)
        except AttributeError:
             # Fallback for older gymnasium or if spaces.Text is experimental
             pass

    def _parse_action_text(self, text: str) -> Tuple[int, str]:
        # Quick parser
        act_match = re.search(r"(\d+)", text)
        action_id = int(act_match.group(1)) if act_match else -1
        
        arg_match = re.search(r"Input:\s*(.*)", text, re.DOTALL)
        raw_argument = arg_match.group(1).strip() if arg_match else ""
        return action_id, raw_argument

    def reset(self, seed=None, options=None) -> Tuple[str, dict]:
        super().reset(seed=seed)
        
        try:
            self.current_sample = next(self.data_iterator)
        except StopIteration:
            # Restart Dataset
            print("Dataset exhausted, restarting streamer...")
            self.data_iterator = self.streamer.stream()
            self.current_sample = next(self.data_iterator)

        self.state = create_initial_state(self.current_sample['question'], self.current_sample['answer'])
        self.ground_truth = self.current_sample['answer']
        self.retriever = EphemeralRetriever(self.current_sample['corpus'])
        self.engine = GreenEngine(retriever=self.retriever)

        
        obs = format_state_for_prompt(cast(Dict[str, Any], self.state))
        return obs, {}

    def step(self, action_text: str) -> Tuple[str, float, bool, bool, dict]:
        # --- ARTIFACT CLEANER ---
        # 1. Strip Token Garbage
        if self.state is None:
            raise ValueError("Environment state is None. Did you forget to reset the environment?") 
        
        if self.ground_truth is None:
            raise ValueError("Ground truth answer is None. Did you forget to reset the environment?")
        
        for bad_token in ["<|eot_id|>", "<|end_of_text|>", "</s>"]:
            action_text = action_text.replace(bad_token, "")
            
        # 2. Strip "Input" loops (The "Moths" case in Ep 6)
        # If the model starts babbling "Input 7 Input 4...", cut it.
        if "Input" in action_text and len(action_text) > 100:
             action_text = action_text.split("Input")[0]

        # 1. Parse
        action_id, raw_argument = self._parse_action_text(action_text)

        # 2. Sanitize (The "World Logic")
        # Encapsulating the logic that was previously in the training loop
        clean_argument = raw_argument
        
        # --- ROBUST INJECTION FIX ---
        # If Action 1 (Answer) and argument is empty or just whitespace/garbage
        if action_id == 1:
            # Check if it has meaningful text (at least 3 alphanumeric chars)
            has_text = len([c for c in clean_argument if c.isalnum()]) > 3
            
            if not has_text:
                # DON'T send a generic instruction.
                # DO send the original Question.
                clean_argument = self.state['question']

        # Sanitization: Remove artifacts if model starts repeating prompt structure
        if "Answer Generation" in clean_argument or "Input:" in clean_argument:
            # If the model hallucinates the prompt text "Input: ...", cut it.
            # Simple heuristic: often the model says "Input: Answer Generation..."
            # We treat that as empty or try to salvage.
            # Valid query shouldn't contain these keywords usually.
            pass # Keep logic simple for now or implement aggressive regex

        # Fallback Logic
        if len(clean_argument) < 3 and action_id in [2, 3]:
            # If Search is active but argument is empty, fallback to the last question
            if self.state['subqueries']:
                clean_argument = self.state['subqueries'][-1]['question']

        # 3. Execute
        self.state = self.engine.step(self.state, action_id, clean_argument)

        

        answer = self.state["answer"] if self.state["answer"] is not None else ""

        done = self.state['status'] in ["SOLVED", "FAILED"]
        # 4. Reward
        reward, breakdown = calculate_reward(
            self.state, 
            action_text, # Passed for length penalty calculation
            self.ground_truth,
            action_id, 
            done, 
            answer # Passed for correctness check
        )

        # 5. Return
        next_obs = format_state_for_prompt(cast(Dict[str, Any], self.state))
        
        # Important: gymnasium expects (obs, reward, terminated, truncated, info)
        terminated = done
        truncated = False
        
        return next_obs, reward, terminated, truncated, breakdown
