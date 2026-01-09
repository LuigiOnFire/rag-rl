from src.agent import workers

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

def gen_slm(query, context):
    """0: GEN_SLM (1B Answer)"""
    return workers.worker_s_task("Answer Generation", f"Query: {query}\nContext: {context}")

def gen_llm(query, context):
    """1: GEN_LLM (8B Answer)"""
    return workers.worker_l_task("Answer Generation", f"Query: {query}\nContext: {context}")

def ret_key(query):
    """2: RET_KEY (BM25 Search)"""
    return workers.bm25_search(query)

def ret_vec(query):
    """3: RET_VEC (Dense Search)"""
    return workers.dense_search(query)

def grd_slm(query, doc):
    """4: GRD_SLM (1B Relevance Check)"""
    return workers.worker_s_task("Relevance Grading", f"Query: {query}\nDocument: {doc}")

def grd_llm(query, response):
    """5: GRD_LLM (8B Hallucination Check)"""
    return workers.worker_l_task("Hallucination Checking", f"Query: {query}\nResponse: {response}")

def rwt_slm(query):
    """6: RWT_SLM (1B Query Rewrite)"""
    return workers.worker_s_task("Query Rewriting", f"Query: {query}")

def dec_slm(query):
    """7: DEC_SLM (1B Simple Decompose)"""
    return workers.worker_s_task("Simple Decomposition", f"Query: {query}")

def dec_llm(query):
    """8: DEC_LLM (8B Deep Decompose)"""
    return workers.worker_l_task("Deep Decomposition", f"Query: {query}")

def fail():
    """9: FAIL (Abort)"""
    return "ABORT"

# Map IDs to functions
ACTION_MAP = {
    0: gen_slm,
    1: gen_llm,
    2: ret_key,
    3: ret_vec,
    4: grd_slm,
    5: grd_llm,
    6: rwt_slm,
    7: dec_slm,
    8: dec_llm,
    9: fail
}

ACTION_NAMES = {
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
