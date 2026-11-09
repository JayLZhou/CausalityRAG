R='/data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1/'; V=R+'out/qtz_option_stop_v2_20261109/'
s=open(R+'pool_sensitivity/scripts/run_pool_sensitivity.sh').read()
o='case "$DS" in quartz) MODE=quartz ;; *) MODE=short_answer ;; esac'
assert s.count(o)==1
open(V+'run_pool_short.sh','w').write(s.replace(o,'MODE=short_answer  # same reader prompt as the original Table III run'))
s=open(R+'queue/mice_flip.py').read()
a='TAG = "qwen_a5_miceflip"'; assert s.count(a)==1
s=s.replace(a,'ROWS["quartz"] = f"{P}/qtz_optstop2/quartz/baselines_1000.jsonl"\nTAG = "qtz_optstop2_miceflip"')
b='    mode = "quartz" if ds == "quartz" else "short_answer"\n'; assert s.count(b)==1
s=s.replace(b,'    mode = "short_answer"\n')
c='    if ds == "quartz":\n        # Same lenient QuaRTz matching'; assert s.count(c)==1
s=s.replace(c,'    if False:\n        # Same lenient QuaRTz matching')
d='flip = not reader_answers_exact_match(question, answer, str(row["clean_answer"]), reader_mode=mode)'; assert s.count(d)==1
s=s.replace(d,'''from causalityrag.reader import canonicalize_reader_answer as _C
                def _lab(x):
                    try:
                        return _C(question, x, reader_mode="quartz")
                    except Exception:
                        return None
                _a, _b = _lab(answer), _lab(str(row["clean_answer"]))
                flip = _a is not None and _b is not None and _a != _b''')
open(V+'mice_flip_v2.py','w').write(s); print('v2 scripts written')
