"""Deterministic curation simulator; not the production agent or legal scorer."""
from .environment_models import Observation, ReleasedEvidence, ActionResult, StateSupport
from .experiment_manifest import digest

PUBLIC_BRIEFING = ('有一宗待核查事件。可按材料类型申请查阅，核实陈述来源和事实，'
                   '不要将陈述等同于已证实事实。查询未命中时保持未知。')
PUBLIC_KINDS = ('陈述', '证言', '辨认记录', '鉴定记录', '现场记录', '书面记录', '视听记录')


def observe(case, released, budget):
    ids = {e.id for e in case.evidence}
    if not set(released) <= ids: raise ValueError('Invalid released set')
    return Observation(case_token=digest(case.case_id)[:16], briefing=case.briefing,
        released=[ReleasedEvidence(id=e.id, text=e.text, kind=e.kind)
                  for e in case.evidence if e.id in released], remaining_actions=budget)


def act(case, released, budget, action):
    # Callers own state; do not accept state supplied by an untrusted agent.
    released = set(released)
    if budget <= 0:
        return ActionResult(status='budget_exhausted', observation=observe(case, released, 0))
    matched = [r for r in case.release_rules if (r.kind, r.person, r.time) ==
               (action.kind, action.person, action.time)]
    status = 'no_match'
    if matched:
        rule = matched[0]
        if set(rule.prerequisites) <= released:
            released.update(rule.evidence_ids); status = 'released'
        else: status = 'prerequisite_missing'
    return ActionResult(status=status, observation=observe(case, released, budget-1))


def state_support(case, released):
    if set(released) - {e.id for e in case.evidence}: raise ValueError('Unknown evidence')
    supported = []; unknown = []; disputed = []
    for c in case.claims:
        if set(c.counterevidence) & set(released): disputed.append(c.id)
        elif any(set(s) <= set(released) for s in c.support_sets): supported.append(c.id)
        else: unknown.append(c.id)
    return StateSupport(supported_claim_ids=supported, unresolved_claim_ids=unknown,
                        disputed_claim_ids=disputed)


def reachability(case):
    reached = set(case.initial_evidence_ids)
    while True:
        before = set(reached)
        for r in case.release_rules:
            if set(r.prerequisites) <= reached: reached.update(r.evidence_ids)
        if before == reached: break
    return {'reachable': sorted(reached), 'unreachable': sorted({e.id for e in case.evidence}-reached),
            'full_reachable_state': state_support(case, reached).model_dump(),
            'semantic_approval': False}
