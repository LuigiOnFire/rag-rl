### Defense 1: The Mathematical Legality (Why it doesn't break the model)
Your advisors' first attack will be: *"If you change the reward function, you change the optimal policy. You are no longer solving the original problem."* 

You will defend this using the most authoritative paper on reward shaping ever published.

*   **The Paper:** Ng, A. Y., Harada, D., & Russell, S. (1999). *Policy invariance under reward transformations: Theory and application to reward shaping.* **Published at ICML (International Conference on Machine Learning).**
*   **The Argument:** Andrew Ng mathematically proved that if a dense reward is formulated as a "Potential-Based Reward Function" (a transition from a lower-value state to a higher-value state), it is mathematically guaranteed *not* to alter the optimal policy of the Markov Decision Process (MDP). 
*   **How to apply it:** You argue that your sub-task reward is not an arbitrary +0.2. It is a potential-based state transition. The environment state changes from `subqueries: [PENDING]` to `subqueries: [ANSWERED]`. Because this state is strictly closer to the terminal state (the final answer), rewarding this transition obeys Ng's theorem of policy invariance. You are speeding up convergence, not changing the goal.

### Defense 2: The Credit Assignment Problem in NLP
Their second attack will be: *"Why can't the model just learn from the final 0 or 1? AlphaGo learned from sparse win/loss signals."*

You will defend this by pointing out that language generation is vastly more complex than a deterministic board game, and sparse rewards fail in stochastic NLP reasoning. 

*   **The Paper:** Uesato, J., et al. (DeepMind, 2022). *Solving Math Word Problems With Process- and Outcome-Based Feedback.* **Published in TACL (Transactions of the Association for Computational Linguistics).**
*   **The Argument:** DeepMind rigorously tested Outcome-Based Reward Models (ORMs - sparse) against Process-Based Reward Models (PRMs - dense) on multi-step reasoning tasks. They proved that sparse rewards in language models suffer from massive credit assignment failures—the model cannot distinguish a logical error in step 2 from a formatting error in step 5.
*   **How to apply it:** Point out that Thrifty RAG is a multi-step stochastic environment. If the agent executes a flawless decomposition and search, but hallucinates the final token generation, a sparse `0` penalizes the flawless search. You cite Uesato to prove that intermediate process supervision is a peer-reviewed necessity for multi-step language tasks.

### Defense 3: The Empirical State-of-the-Art
Their final attack might be: *"Okay, but does this actually scale to modern LLM architectures like GRPO?"*

Here is where you drop the preprints, but you frame them properly. In modern Deep Learning, waiting for a journal publication takes 18 months; the field moves in weeks. You use these to show empirical dominance.

*   **The Paper:** Lightman, H., et al. (OpenAI, 2023). *Let's Verify Step by Step.* **(arXiv / OpenAI Core Research).**
*   **The Argument:** This is the paper that built GPT-4's reasoning capabilities (and is the foundation for models like DeepSeek-R1). OpenAI empirically proved that process supervision (rewarding intermediate reasoning steps) vastly outperforms outcome supervision (rewarding final answers) and drastically reduces hallucination (which they term "alignment tax"). 
*   **How to apply it:** You tell your advisors: *"While ICML and TACL provide the theoretical and peer-reviewed foundation, OpenAI's 2023 empirical results dictate the current industry standard. To train a routing agent on sparse rewards in 2026 would be actively ignoring the established architecture of frontier models."*

---

### The "Critical" Self-Check
Before the meeting, you must ensure your code actually matches this defense. 
If you reward the *action* of calling `DEC_LLM`, you are committing the sin of heuristic bias (Reward Hacking). 
If you reward the *environmental state change* of a subquery being successfully resolved and added to the context, you are executing mathematically sound Process Supervision.

Does the prompt-based classifier for the Pessimistic Route (our safety net) make sense to you as the immediate next step while those sweeping jobs run on the cluster?