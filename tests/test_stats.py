import pytest
from llm_animal_experiment.stats import run_statistical_test

def test_statistical_test_empty():
    res = run_statistical_test(yes_count=0, no_count=0, invalid_count=5)
    assert res.valid_trials == 0
    assert res.invalid_trials == 5
    assert res.reject_null is False
    assert "No valid trials" in res.conclusion

def test_statistical_test_consistent_with_null():
    # 33 Yes, 67 No -> ~33%
    res = run_statistical_test(yes_count=33, no_count=67, confidence_level=0.95)
    assert res.valid_trials == 100
    assert res.reject_null is False
    assert res.binom_pvalue > 0.05
    assert res.ci_low <= (1 / 3) <= res.ci_high

def test_statistical_test_reject_null():
    # 90 Yes, 10 No -> 90% (far higher than 33%)
    res = run_statistical_test(yes_count=90, no_count=10, confidence_level=0.95)
    assert res.valid_trials == 100
    assert res.reject_null is True
    assert res.binom_pvalue < 0.001
    assert "reject the null hypothesis" in res.conclusion

def test_confidence_level_change():
    res_95 = run_statistical_test(yes_count=5, no_count=5, confidence_level=0.95)
    res_99 = run_statistical_test(yes_count=5, no_count=5, confidence_level=0.99)
    # 99% CI should be wider than 95% CI
    width_95 = res_95.ci_high - res_95.ci_low
    width_99 = res_99.ci_high - res_99.ci_low
    assert width_99 > width_95
