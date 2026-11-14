from app.curation_identity import identity_errors


def test_pronoun_action_sentence_cannot_establish_named_identity():
    sentences = [{'id':'S1','text':'甲某的供述。'}, {'id':'S2','text':'他随后离开。'}]
    proposal = {'claim':'甲某供述自己离开。','source_attribution':'甲某供述','identity_sentence_ids':['S2']}
    assert identity_errors([proposal],sentences,['甲某'])
    proposal['identity_sentence_ids'] = ['S1','S2']
    assert identity_errors([proposal],sentences,['甲某']) == []


def test_unknown_reference_and_attribution_are_checked():
    proposal = {'claim':'该男子离开。','source_attribution':'甲某供述','identity_sentence_ids':['missing']}
    assert identity_errors([proposal],[{'id':'S1','text':'男子离开。'}],['甲某'])
    proposal['identity_sentence_ids'] = ['S1']
    assert identity_errors([proposal],[{'id':'S1','text':'男子离开。'}],['甲某'])


def test_single_proposal_limit_survives_constrained_decoder():
    from scripts.audit_interactive_cases import CurationAudit
    from app.local_context import decoder_schema
    assert decoder_schema(CurationAudit)['properties']['atomic_event_proposals']['maxItems'] == 1
