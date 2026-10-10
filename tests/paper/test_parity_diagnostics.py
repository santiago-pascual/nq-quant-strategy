from copy import deepcopy
from src.paper.parity_diagnostics import compare_event_samples


def rows():
    return [{'event_type':'market_data','timestamp':'2026-10-08T13:00:00Z',
             'payload':{'open':1,'high':2,'low':1,'close':2,'volume':3,'market_data_contract_id':815824267}},
            {'event_type':'strategy_decision','timestamp':'2026-10-08T13:00:00Z',
             'payload':{'strategy_name':'ORB','action':'hold','signal':'flat','reason':'No breakout'}}]


def test_identical_duplicate_inputs_cannot_establish_parity():
    assert compare_event_samples(rows()+rows(), rows()+rows(), left_config={},right_config={})['state']=='UNAVAILABLE'


def test_identical_incomplete_inputs_cannot_establish_parity():
    sample=rows(); sample[0]['payload'].pop('volume')
    assert compare_event_samples(sample,sample,left_config={},right_config={})['state']=='UNAVAILABLE'


def test_equivalent_audit_timestamps_and_ids_do_not_change_decisions():
    left=rows(); right=deepcopy(left); right[0]['event_id']='different-audit-id'
    assert compare_event_samples(left,right,left_config={},right_config={})['state']=='MATCHED_SAMPLE_EQUIVALENT'


def test_unexplained_decision_and_input_differences_are_separated():
    left=rows(); right=deepcopy(left); right[1]['payload']['action']='enter'
    result=compare_event_samples(left,right,left_config={},right_config={})
    assert result['differences'][0]['category']=='UNEXPLAINED_MATCHED_INPUT_DIFFERENCE'
    right[0]['payload']['close']=1
    result=compare_event_samples(left,right,left_config={},right_config={})
    assert not result['identical_recorded_inputs']
    assert any(x['category']=='INPUT_OR_SESSION_DIFFERENCE' for x in result['differences'])
    assert result['trading_actions']==[]


def test_missing_market_data_is_not_claimed_as_parity():
    assert compare_event_samples([],rows(),left_config={},right_config={})['state']=='UNAVAILABLE'
