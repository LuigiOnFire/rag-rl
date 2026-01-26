Right now search.py is super broken
We need the actual logic in workers or actions.py
In fact I think a lot of the logic in search.py needs to be moved there

Semantic rename of instuctions is probably a good idea too

Choose the next action:
- SEARCH: Search for keywords (Entities, Names)
- ANSWER: Final answer generation
...
Format:
Action: SEARCH
Input: obama height