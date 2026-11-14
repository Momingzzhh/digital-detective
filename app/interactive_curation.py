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


def apply_environment_patch(case, patch):
    """Only expose source-backed query values already available to the actor."""
    from .environment_models import PrivateCase
    if set(patch) != {'initial_evidence_ids', 'release_rules'}:
        raise ValueError('Environment patch may only change initial evidence and release rules')
    revised = PrivateCase.model_validate(case.model_dump() | patch)
    index = {e.id: e.text for e in revised.evidence}
    for rule in revised.release_rules:
        if rule.kind not in PUBLIC_KINDS: raise ValueError('Nonpublic action kind')
        available = set(revised.initial_evidence_ids) | set(rule.prerequisites)
        text = '\n'.join(index[k] for k in available)
        for value in (rule.person, rule.time):
            if value and value not in text:
                raise ValueError('Query value is not supported by prior visible evidence')
    if reachability(revised)['unreachable']:
        raise ValueError('Environment patch leaves unreachable evidence')
    return revised


def apply_case_revisions(cases, revisions):
    from .environment_models import PrivateCase
    result = list(cases); seen = set()
    for revision in revisions:
        if revision.get('author_type') != 'assistant_source_reader' or revision.get('human_approval') is not False:
            raise ValueError('Revision must be explicitly nonhuman and unapproved')
        cid = revision['case_id']
        if cid in seen: raise ValueError('Duplicate case revision')
        seen.add(cid)
        matches = [i for i,c in enumerate(result) if c.case_id == cid]
        if len(matches) != 1: raise ValueError('Revision outside fixed cohort')
        i = matches[0]; original = result[i]
        if digest(original.model_dump()) != revision['parent_case_binding']:
            raise ValueError('Revision parent changed')
        updated = PrivateCase.model_validate(original.model_dump() | {'claims': revision['claims']})
        if revision.get('withhold_evidence_ids'):
            updated = quarantine_evidence(updated, revision['withhold_evidence_ids'])
        if 'environment_patch' in revision:
            updated = apply_environment_patch(updated, revision['environment_patch'])
        result[i] = updated
    return result


def quarantine_evidence(case, evidence_ids):
    """Remove outcome-bearing material; dependent claims become unknown, not gold."""
    from .environment_models import PrivateCase
    held = set(evidence_ids)
    if not held or held - {e.id for e in case.evidence}: raise ValueError('Unknown quarantine evidence')
    value = case.model_dump()
    value['evidence'] = [e for e in value['evidence'] if e['id'] not in held]
    value['initial_evidence_ids'] = [i for i in value['initial_evidence_ids'] if i not in held]
    rules = []
    for rule in value['release_rules']:
        if held & set(rule['prerequisites']): continue  # Never silently relax a prerequisite.
        rule['evidence_ids'] = [i for i in rule['evidence_ids'] if i not in held]
        if rule['evidence_ids']: rules.append(rule)
    value['release_rules'] = rules
    for claim in value['claims']:
        if held & (set(claim['counterevidence']) | {i for support in claim['support_sets'] for i in support}):
            claim.update(support_sets=[], support_citations=[], counterevidence=[], quality='unknown',
                         unknown_reason='Supporting or counterevidence quarantined for source-boundary review.')
    return PrivateCase.model_validate(value)
