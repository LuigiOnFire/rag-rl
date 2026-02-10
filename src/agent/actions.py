# Action IDs
ACTION_GEN_SLM = 0
ACTION_GEN_LLM = 1
ACTION_RET_KEY = 2
ACTION_RET_VEC = 3
ACTION_GRD_SLM = 4
ACTION_GRD_LLM = 5
ACTION_RWT_SLM = 6
ACTION_DEC_SLM = 7
ACTION_DEC_LLM = 8
ACTION_FAIL = 9

# All action IDs for iteration
ALL_ACTION_IDS = (
    ACTION_GEN_SLM,
    ACTION_GEN_LLM,
    ACTION_RET_KEY,
    ACTION_RET_VEC,
    ACTION_GRD_SLM,
    ACTION_GRD_LLM,
    ACTION_RWT_SLM,
    ACTION_DEC_SLM,
    ACTION_DEC_LLM,
    ACTION_FAIL,
)

def get_action_name(action_id: int) -> str:
    mapping = {
        0: "GEN_SLM",
        1: "GEN_LLM",
        2: "RET_KEY",
        3: "RET_VEC",
        4: "GRD_SLM",
        5: "GRD_LLM",
        6: "RWT_SLM",
        7: "DEC_SLM",
        8: "DEC_LLM",
        9: "FAIL"
    }
    return mapping.get(action_id, "UNKNOWN")
