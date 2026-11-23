from src.agent import actions

# Mean per-action energy (Joules) from Table III of "Beyond Tokens"
OFFLINE_ACTION_JOULES = {
    "hotpot": {
        actions.ACTION_GEN_SLM: 347.9,
        actions.ACTION_GEN_LLM: 640.8,
        actions.ACTION_RET_KEY: 342.0,
        actions.ACTION_RET_VEC: 1321.3,
        actions.ACTION_RSN_SLM: 433.7,
        actions.ACTION_RSN_LLM: 1242.5,  # Proxy: DEC_LLM (32B reasoning pass)
        actions.ACTION_DEC_LLM: 1242.5,
        actions.ACTION_DEC_RSN: 2149.7,
        actions.ACTION_ABORT:   0.0,
    },
    "nq": {
        actions.ACTION_GEN_SLM: 261.7,
        actions.ACTION_GEN_LLM: 408.6,
        actions.ACTION_RET_KEY: 255.8,
        actions.ACTION_RET_VEC: 1810.4,
        actions.ACTION_RSN_SLM: 290.8,
        actions.ACTION_RSN_LLM: 697.3,
        actions.ACTION_DEC_LLM: 697.3,
        actions.ACTION_DEC_RSN: 1220.8,
        actions.ACTION_ABORT:   0.0,
    },
    "squad": {
        actions.ACTION_GEN_SLM: 295.1,
        actions.ACTION_GEN_LLM: 570.9,
        actions.ACTION_RET_KEY: 284.6,
        actions.ACTION_RET_VEC: 2379.0,
        actions.ACTION_RSN_SLM: 346.2,
        actions.ACTION_RSN_LLM: 1117.1,
        actions.ACTION_DEC_LLM: 1117.1,
        actions.ACTION_DEC_RSN: 2074.9,
        actions.ACTION_ABORT:   0.0,
    },
}

DEFAULT_ACTION_JOULES = {
    aid: sum(OFFLINE_ACTION_JOULES[ds][aid] for ds in OFFLINE_ACTION_JOULES) / 3.0
    for aid in OFFLINE_ACTION_JOULES["hotpot"]
}

def get_offline_step_joules(dataset: str, action_id: int) -> float:
    ds_key = (dataset or "hotpot").lower().strip().replace("-", "_").replace(" ", "_")
    aliases = {
        "hotpotqa": "hotpot",
        "natural_questions": "nq",
        "naturalquestion": "nq",
        "natural_question": "nq",
        "squad_v1": "squad",
        "squad_v2": "squad",
    }
    ds_key = aliases.get(ds_key, ds_key)
    table = OFFLINE_ACTION_JOULES.get(ds_key, DEFAULT_ACTION_JOULES)
    return float(table.get(action_id, 0.0))