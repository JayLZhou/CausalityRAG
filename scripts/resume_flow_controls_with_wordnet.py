#!/usr/bin/env python3
"""Add missing control dependencies without overriding the reader environment."""
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import site
import sys

ROOT = Path('/data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1')
site.addsitedir(str(ROOT / 'runtime_deps_cfr'))
sys.path.insert(0, str(ROOT))
os.environ['NLTK_DATA'] = '/data1/yujia/nltk_data'

from causalityrag.semantic_filter import ensure_wordnet_available
from causalityrag.shared_replacement_pool import file_sha256
from scripts.run_flow_ablation_pilot import freeze_json

ensure_wordnet_available()
packages = {}
for name in ('nltk', 'regex', 'tqdm', 'joblib', 'click', 'transformers', 'torch'):
    module = importlib.import_module(name)
    packages[name] = dict(version=importlib.metadata.version(name), path=module.__file__)
target = ROOT / 'scripts/run_flow_ablation_controls_v2.py'
out = ROOT / 'out/flow_ablation_cfr_20261025_v2'
freeze_json(out / 'wordnet_resume_runtime_20261026.json', dict(
    packages=packages, dependency_resolution='append isolated dependencies after existing site-packages',
    wordnet_sha256=file_sha256('/data1/yujia/nltk_data/corpora/wordnet.zip'),
    target_sha256=file_sha256(str(target)), wrapper_sha256=file_sha256(__file__),
    argv=sys.argv[1:],
))
print('WORDNET dependency preflight passed; resuming immutable control checkpoints', flush=True)
sys.argv[0] = str(target)
runpy.run_path(str(target), run_name='__main__')
