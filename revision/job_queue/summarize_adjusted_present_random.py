"""summarize_adjusted_flips.py from the Table 3 code, averaging only the Random
seeds that were actually run (A1 runs Random with seed 0 only)."""
import json, runpy, sys
REPO = '/data1/yujia/CausalityRAG_release_no_frontier_repair_20260810'
sys.path.insert(0, REPO)
argv = sys.argv[1:]
factual = json.load(open(argv[argv.index('--factual') + 1]))['methods']
module = runpy.run_path(f'{REPO}/scripts/summarize_adjusted_flips.py', run_name='lib')
present = [m for m in module['RANDOM_METHODS'] if m in factual]
module['main'].__globals__['RANDOM_METHODS'] = tuple(present)
sys.argv = ['summarize_adjusted_flips.py'] + argv
module['main']()
