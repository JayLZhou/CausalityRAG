# ReFlow experiment runner

`run_contribution_aware_flow_contract_attack.py` is the only experimental
runner retained in this repository. It evaluates the Breakpoint Flow-Contract
frontier and verifies candidate interventions with the frozen reader. Its
`breakpoint` mode uses analytic endpoints and exact objective-line
intersections to recover the extreme supported frontier.

`arc_jsd.py` and `run_arc_jsd_sentence_lift_attack.py` implement the retained
ARC-JSD sentence-attribution baseline and its sentence-to-token lift. They are
kept only for external comparison.
