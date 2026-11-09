"""Run evaluate_reflow.py with the original reader prompt, stopping only when the selected QTZ option changes."""
import importlib.util, os, sys
W = "/data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1/"
sys.path.insert(0, W); os.chdir(W)
os.environ["CAUSALITYRAG_READER_MODE"] = "short_answer"
spec = importlib.util.spec_from_file_location("evaluate_reflow", W + "scripts/evaluate_reflow.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
from causalityrag.reader import canonicalize_reader_answer as C
from causalityrag.evaluation_metrics import valid_answer
def lab(q, x):
    if not valid_answer(x): return None
    try: return C(q, x, reader_mode="quartz")
    except Exception: return None
def same(question, left, right, reader_mode=None, **kw):
    a, b = lab(question, left), lab(question, right)
    return a is None or b is None or a == b
m.reader_answers_exact_match = same
m.main()
