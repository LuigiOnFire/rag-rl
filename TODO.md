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

- NOTES FOR ME TOMORROW
  File "/home/wcrawford/rag_rl/scripts/02_generate.py", line 102, in <module>
    main()
  File "/home/wcrawford/rag_rl/scripts/02_generate.py", line 65, in main
    solution_state, debug_info = oracle_search.solve(start_state, ground_truth)
                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/wcrawford/rag_rl/src/oracle/search.py", line 134, in solve
    final_sub = current_state['subqueries'][0]
                ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^
IndexError: list index out of range

The answer is not passed correctly to search.py because the GreenState does not have an answer field. We need to figure out whether we make one of put the variable somewhere else.