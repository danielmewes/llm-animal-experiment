from dataclasses import dataclass
from typing import Optional
from scipy import stats

@dataclass
class ConfidenceInterval:
    low: float
    high: float
    confidence_level: float

@dataclass
class StatisticalTestResult:
    total_trials: int
    valid_trials: int
    invalid_trials: int
    yes_count: int
    no_count: int
    observed_yes_prop: float
    expected_yes_prop: float
    confidence_level: float
    alpha: float
    ci_low: float
    ci_high: float
    binom_pvalue: float
    chi2_stat: Optional[float]
    chi2_pvalue: Optional[float]
    reject_null: bool
    conclusion: str

    def format_summary(self) -> str:
        lines = []
        lines.append("=" * 65)
        lines.append("           STATISTICAL ANALYSIS OF EXPERIMENT")
        lines.append("=" * 65)
        lines.append(f"Total Trials:    {self.total_trials}")
        lines.append(f"Valid Trials:    {self.valid_trials}")
        lines.append(f"Invalid Trials:  {self.invalid_trials} (excluded from hypothesis test)")
        lines.append("-" * 65)
        lines.append("Observed vs Expected Distribution (Valid Trials):")
        lines.append(f"  - Yes: {self.yes_count:>4} / {self.valid_trials} "
                     f"({self.observed_yes_prop * 100:>6.2f}%) "
                     f"[Expected: {self.valid_trials * self.expected_yes_prop:>6.2f} ({self.expected_yes_prop * 100:.2f}%)]")
        no_prop = (self.no_count / self.valid_trials) if self.valid_trials > 0 else 0.0
        exp_no_prop = 1.0 - self.expected_yes_prop
        lines.append(f"  - No:  {self.no_count:>4} / {self.valid_trials} "
                     f"({no_prop * 100:>6.2f}%) "
                     f"[Expected: {self.valid_trials * exp_no_prop:>6.2f} ({exp_no_prop * 100:.2f}%)]")
        lines.append("-" * 65)
        lines.append("Hypothesis Test:")
        lines.append(f"  H0: Model picks animal uniformly at random (p(Yes) = {self.expected_yes_prop:.4f})")
        lines.append(f"  H1: Model does not match uniform guess distribution (p(Yes) != {self.expected_yes_prop:.4f})")
        lines.append(f"  Confidence Level: {self.confidence_level * 100:.1f}%  (alpha = {self.alpha:.4f})")
        lines.append(f"  Exact Clopper-Pearson CI for p(Yes): [{self.ci_low:.4f}, {self.ci_high:.4f}]")
        lines.append(f"  Exact Binomial Test p-value:         {self.binom_pvalue:.6f}")
        if self.chi2_stat is not None and self.chi2_pvalue is not None:
            lines.append(f"  Chi-Square Goodness-of-Fit:          chi2 = {self.chi2_stat:.4f}, p = {self.chi2_pvalue:.6f}")
        lines.append("-" * 65)
        status = "REJECT NULL HYPOTHESIS" if self.reject_null else "FAIL TO REJECT NULL HYPOTHESIS"
        lines.append(f"Decision: {status}")
        lines.append(f"Conclusion: {self.conclusion}")
        lines.append("=" * 65)
        return "\n".join(lines)


def run_statistical_test(
    yes_count: int,
    no_count: int,
    invalid_count: int = 0,
    expected_yes_prop: float = 1.0 / 3.0,
    confidence_level: float = 0.95,
) -> StatisticalTestResult:
    """
    Performs an exact two-sided binomial test and goodness-of-fit test
    to check whether the observed frequency of 'Yes' responses is consistent
    with the expected theoretical probability (default 1/3).
    """
    valid_trials = yes_count + no_count
    total_trials = valid_trials + invalid_count
    alpha = 1.0 - confidence_level

    if valid_trials == 0:
        return StatisticalTestResult(
            total_trials=total_trials,
            valid_trials=0,
            invalid_trials=invalid_count,
            yes_count=0,
            no_count=0,
            observed_yes_prop=0.0,
            expected_yes_prop=expected_yes_prop,
            confidence_level=confidence_level,
            alpha=alpha,
            ci_low=0.0,
            ci_high=0.0,
            binom_pvalue=1.0,
            chi2_stat=None,
            chi2_pvalue=None,
            reject_null=False,
            conclusion="No valid trials recorded to perform a statistical test.",
        )

    observed_yes_prop = yes_count / valid_trials

    # Exact binomial test (two-sided)
    binom_res = stats.binomtest(
        k=yes_count,
        n=valid_trials,
        p=expected_yes_prop,
        alternative="two-sided",
    )
    binom_pvalue = float(binom_res.pvalue)

    # Clopper-Pearson exact confidence interval for proportion
    ci = binom_res.proportion_ci(confidence_level=confidence_level, method="exact")
    ci_low = float(ci.low)
    ci_high = float(ci.high)

    # Chi-square goodness of fit test (for comparison)
    chi2_stat: Optional[float] = None
    chi2_pvalue: Optional[float] = None
    expected_yes = valid_trials * expected_yes_prop
    expected_no = valid_trials * (1.0 - expected_yes_prop)
    
    # Calculate chi2 if expected frequencies are positive
    if expected_yes > 0 and expected_no > 0:
        try:
            chi_res = stats.chisquare(
                f_obs=[yes_count, no_count],
                f_exp=[expected_yes, expected_no],
            )
            chi2_stat = float(chi_res.statistic)
            chi2_pvalue = float(chi_res.pvalue)
        except Exception:
            pass

    reject_null = binom_pvalue < alpha

    if reject_null:
        conclusion = (
            f"At the {confidence_level * 100:.1f}% confidence level (p = {binom_pvalue:.4f} < {alpha:.4f}), "
            f"we reject the null hypothesis. The observed frequency of 'Yes' ({observed_yes_prop:.2%}) "
            f"deviates significantly from the expected {expected_yes_prop:.2%}."
        )
    else:
        conclusion = (
            f"At the {confidence_level * 100:.1f}% confidence level (p = {binom_pvalue:.4f} >= {alpha:.4f}), "
            f"we fail to reject the null hypothesis. The observed frequency of 'Yes' ({observed_yes_prop:.2%}) "
            f"is consistent with the expected theoretical probability of {expected_yes_prop:.2%}."
        )

    return StatisticalTestResult(
        total_trials=total_trials,
        valid_trials=valid_trials,
        invalid_trials=invalid_count,
        yes_count=yes_count,
        no_count=no_count,
        observed_yes_prop=observed_yes_prop,
        expected_yes_prop=expected_yes_prop,
        confidence_level=confidence_level,
        alpha=alpha,
        ci_low=ci_low,
        ci_high=ci_high,
        binom_pvalue=binom_pvalue,
        chi2_stat=chi2_stat,
        chi2_pvalue=chi2_pvalue,
        reject_null=reject_null,
        conclusion=conclusion,
    )
