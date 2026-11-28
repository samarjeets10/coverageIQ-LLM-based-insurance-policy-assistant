# CoverageIQ — Module 6A Feature Engineering Specification (`FEATURES.md`)

**Feature Schema Version:** `1.0.0`  
**Total Predictive Features:** `59` (`float64`, strictly finite, zero `NaN`/`Inf`)  
**Implementation:** [`features.py`](features.py)

---

## 1. Architecture & Design Decisions

### Unified Schema with Category Indicators (Step 7 Decision)
Rather than maintaining four separate incompatible feature matrices (`health`, `car`, `life`, `term`), Module 6A uses a **single unified 59-column `float64` matrix (`FEATURE_COLUMNS`)** with explicit category indicator columns (`is_health`, `is_car`, `is_life`, `is_term`, `policy_type_encoded`) and availability flags (`has_claim_settlement_ratio`, `has_annual_income`, `has_vehicle_age`):
- **Why:** LightGBM and XGBoost split naturally on category indicators while sharing affordability, coverage headroom, and claim reliability signals across categories.
- **Category Isolation:** Health-specific columns (`hospital_network_count`, `ped_*`, `co_payment_percentage`, etc.) are computed strictly when `policy_type == "health"` and set to `0.0` otherwise (preventing Car garage counts from leaking into Health hospital counts). Car columns and Life/Term columns are similarly gated by their respective categories.
- **Unified Schema & Missing PED Semantics (M5.I2):** `FEATURE_SCHEMA_VERSION` stays pinned to `1.0.0` (`59` features) for compatibility with the trained LightGBM ranker (`lightgbm_ranker.joblib`). When a `health` policy covers pre-existing conditions (`ped_covered=True`) with `ped_waiting_period_months=None`, `ped_waiting_period_days` uses a conservative `36`-month (`1080.0`-day) fallback rather than `0.0` days (which would reward missing data in weighted scoring) or `> 1260.0` days (which triggers an unintended higher leaf in LightGBM Tree 11). Uncovered health PED policies (`ped_covered=False`) and non-health categories keep `ped_waiting_period_days = 0.0`, while weighted scoring assigns `0.0` PED-wait credit whenever `ped_covered=False`.
- **Fixed Ordering:** Column order is locked to `FEATURE_COLUMNS` regardless of input policy row ordering.

### Leakage Prohibition
The following fields are **strictly excluded** from `FEATURE_COLUMNS` (`FORBIDDEN_FEATURE_COLUMNS`):
- **Targets / Post-hoc scores:** `relevance_label`, `suitability_score`
- **Hard-filter artifacts:** `is_eligible`, `failed_filters`, `passed_filters`, `elimination_reasons`
- **Unique identifiers:** `profile_id`, `policy_id`, `policy_code`

Hard filtering (`filters.py`) runs **before** feature extraction so only eligible candidates enter ranking. Direct calls to `extract_policy_features` / `build_feature_matrix` also enforce the same Stage 4 strict numeric/integer parsing, domain bounds, and unified boolean normalization (`coercion.py`), raising `InvalidNumericMetadataError` on malformed or out-of-range inputs and `MissingMandatoryFeatureInputError` on missing required inputs. Training data (`build_training_feature_dataset`) returns `(X, y, groups, metadata)` without performing random train/test splitting so Stage 6–8 evaluations use `profile_id` group-aware splits (`GroupKFold` / `GroupShuffleSplit`).

---

## 2. Feature Dictionary (59 Features)

### 2.1 Category & Availability Indicators (8)

| # | Feature Name | Formula / Definition | Source | Category | Missing Strategy |
|---|---|---|---|---|---|
| 1 | `policy_type_encoded` | `{"health": 0, "car": 1, "life": 2, "term": 3}` | `policy.policy_type` | All | Mandatory (raises error) |
| 2 | `is_health` | `1.0` if `policy_type == "health"` else `0.0` | `policy.policy_type` | All | Deterministic |
| 3 | `is_car` | `1.0` if `policy_type == "car"` else `0.0` | `policy.policy_type` | All | Deterministic |
| 4 | `is_life` | `1.0` if `policy_type == "life"` else `0.0` | `policy.policy_type` | All | Deterministic |
| 5 | `is_term` | `1.0` if `policy_type == "term"` else `0.0` | `policy.policy_type` | All | Deterministic |
| 6 | `has_claim_settlement_ratio` | `1.0` if `claim_settlement_ratio` is present else `0.0` | `policy` | All | Availability flag |
| 7 | `has_annual_income` | `1.0` if `annual_income > 0` else `0.0` | `profile` | All | Availability flag |
| 8 | `has_vehicle_age` | `1.0` if `is_car == 1` and `vehicle_age_years` present else `0.0` | `profile` | Car | Availability flag |

### 2.2 Common User Profile Features (5)

| # | Feature Name | Formula / Definition | Source | Category | Missing Strategy |
|---|---|---|---|---|---|
| 9 | `applicant_age` | `float(profile.age)` | `profile.age` | All | Mandatory |
| 10 | `annual_income` | `float(annual_income_inr)` | `profile.annual_income_inr` | All | `0.0` + `has_annual_income` |
| 11 | `monthly_budget` | `monthly_budget_inr` or `budget_max / 12.0` | `profile` | All | Derived from `budget_max` |
| 12 | `family_size` | `float(profile.family_size)` | `profile.family_size` | All | Defaults to `1.0` |
| 13 | `risk_preference_encoded` | `{"low_premium": 0, "balanced": 1, "comprehensive_coverage": 2}` | `profile.risk_preference` | All | Defaults to `1.0` (`balanced`) |

### 2.3 Common Affordability Features (5)

| # | Feature Name | Mathematical Formula | Source | Category | Missing Strategy |
|---|---|---|---|---|---|
| 14 | `annual_premium` | `0.5 * (annual_premium_min + annual_premium_max)` | `policy` | All | Mandatory (`annual_premium` fallback) |
| 15 | `monthly_premium` | `annual_premium / 12.0` | Derived | All | Mandatory |
| 16 | `premium_to_budget_ratio` | `annual_premium / max(budget_max, epsilon)` (0.0 if denom `<= 0`) | `profile`, `policy` | All | Safe zero-denom guard |
| 17 | `affordability_margin` | `budget_max - annual_premium` | `profile`, `policy` | All | Mandatory |
| 18 | `affordability_score` | `clip((budget_max - annual_premium_min) / budget_max, 0.0, 1.0)` | `profile`, `policy` | All | Safe zero-denom guard |

### 2.4 Common Coverage & Age Margin Features (6)

| # | Feature Name | Mathematical Formula | Source | Category | Missing Strategy |
|---|---|---|---|---|---|
| 19 | `sum_insured` | `float(sum_insured_max)` | `policy` | All | Mandatory (`sum_insured` fallback) |
| 20 | `requested_sum_insured` | `float(min_sum_insured)` | `profile` | All | Mandatory |
| 21 | `coverage_to_requirement_ratio` | `sum_insured / requested_sum_insured` | `profile`, `policy` | All | Safe zero-denom guard |
| 22 | `coverage_headroom` | `sum_insured - requested_sum_insured` | `profile`, `policy` | All | Mandatory |
| 23 | `age_eligibility_margin` | `max_entry_age - applicant_age` | `profile`, `policy` | All | Mandatory |
| 24 | `remaining_eligible_years` | `max(0.0, (max_renewal_age or (100 if lifelong else max_entry_age)) - applicant_age)` | `profile`, `policy` | All | Falls back to `100` / `max_entry_age` |

### 2.5 Common Benefit & Exclusion Features (4)

| # | Feature Name | Mathematical Formula | Source | Category | Missing Strategy |
|---|---|---|---|---|---|
| 25 | `total_benefit_count` | `len(covered_benefits)` | `policy.covered_benefits` | All | `0.0` if empty |
| 26 | `matched_required_benefits` | `len(required_benefits & covered_benefits)` | `profile`, `policy` | All | `0.0` if empty |
| 27 | `required_benefit_match_ratio` | `1.0` if `len(required_benefits) == 0` else `matched / len(required_benefits)` | `profile`, `policy` | All | `1.0` when none requested |
| 28 | `exclusion_conflict_count` | `len((required_benefits \| pre_existing_conditions) & exclusions)` | `profile`, `policy` | All | `0.0` if empty |

### 2.6 Common Policy Quality Features (5)

| # | Feature Name | Mathematical Formula / Definition | Source | Category | Missing Strategy |
|---|---|---|---|---|---|
| 29 | `claim_settlement_ratio` | `float(claim_settlement_ratio)` | `policy` | All | `0.0` + `has_claim_settlement_ratio` |
| 30 | `deductible_amount` | `float(deductible_amount)` | `policy` | All | `0.0` |
| 31 | `no_claim_bonus_pct` | `float(no_claim_bonus_pct)` | `policy` | All | `0.0` |
| 32 | `restoration_benefit_indicator` | `1.0` if `restoration_benefit` else `0.0` | `policy` | All | `0.0` |
| 33 | `pan_india_coverage_indicator` | `1.0` if `pan_india_coverage` else `0.0` | `policy` | All | `0.0` |

### 2.7 Health-Specific Features (8) — `0.0` when `is_health == 0`

| # | Feature Name | Formula / Definition | Source |
|---|---|---|---|
| 34 | `ped_coverage_indicator` | `1.0` if `ped_covered` else `0.0` | `policy.ped_covered` |
| 35 | `ped_waiting_period_days` | `ped_waiting_period_months * 30.0` (conservative `1080.0` [36-month] fallback when `ped_covered=True` and null; `0.0` if uncovered/non-health) | `policy.ped_waiting_period_months` |
| 36 | `initial_waiting_period_days` | `float(initial_waiting_period_days)` | `policy.initial_waiting_period_days` |
| 37 | `room_rent_restriction_indicator` | `1.0` if `room_rent_limit_type != "no_limit"` else `0.0` | `policy.room_rent_limit_type` |
| 38 | `cashless_availability_indicator` | `1.0` if `cashless_available` else `0.0` | `policy.cashless_available` |
| 39 | `hospital_network_count` | `float(network_hospitals_count)` | `policy.network_hospitals_count` |
| 40 | `co_payment_percentage` | `float(copay_pct)` | `policy.copay_pct` |
| 41 | `family_floater_indicator` | `1.0` if `coverage_mode in ("family_floater", "both")` else `0.0` | `policy.coverage_mode` |

### 2.8 Car-Specific Features (8) — `0.0` when `is_car == 0`

| # | Feature Name | Formula / Definition | Source |
|---|---|---|---|
| 42 | `vehicle_age` | `float(vehicle_age_years)` (`0.0` if omitted) | `profile.vehicle_age_years` |
| 43 | `vehicle_age_limit` | `float(meta["vehicle_age_limit_years"])` | `type_specific_metadata` |
| 44 | `vehicle_age_eligibility_margin` | `vehicle_age_limit - vehicle_age` (when both present, else `0.0`) | Derived |
| 45 | `own_damage_indicator` | `1.0` if `meta["own_damage_coverage"]` else `0.0` | `type_specific_metadata` |
| 46 | `third_party_indicator` | `1.0` if `meta["third_party_coverage"]` else `0.0` | `type_specific_metadata` |
| 47 | `zero_depreciation_indicator` | `1.0` if `meta["zero_depreciation"]` else `0.0` | `type_specific_metadata` |
| 48 | `roadside_assistance_indicator` | `1.0` if `meta["roadside_assistance"]` else `0.0` | `type_specific_metadata` |
| 49 | `engine_protection_indicator` | `1.0` if `meta["engine_protection"]` else `0.0` | `type_specific_metadata` |

### 2.9 Life & Term-Specific Features (10) — `0.0` when inapplicable

| # | Feature Name | Formula / Definition | Applicable Categories |
|---|---|---|---|
| 50 | `life_cover` | `float(meta.get("life_cover_max") or sum_insured)` | `life`, `term` |
| 51 | `requested_life_cover` | `float(min_sum_insured)` | `life`, `term` |
| 52 | `life_cover_ratio` | `life_cover / requested_life_cover` | `life`, `term` |
| 53 | `policy_term_years` | `float(meta.get("policy_term_years_max") or max_tenure_years)` | `life`, `term` |
| 54 | `premium_payment_term_years` | `float(meta.get("premium_payment_term_years") or 0.0)` | `life`, `term` |
| 55 | `maturity_benefit_indicator` | `1.0` if `meta.get("maturity_benefit")` else `0.0` | `life` |
| 56 | `surrender_value_indicator` | `1.0` if `meta.get("surrender_value_available")` else `0.0` | `life` |
| 57 | `smoking_eligibility_compatibility` | `1.0` (`all` / non-smoker), `0.5` (`smoker_surcharge` for smoker), `0.0` (ineligible) | `term` |
| 58 | `critical_illness_rider_indicator` | `1.0` if `meta.get("critical_illness_rider")` else `0.0` | `term` |
| 59 | `accidental_death_rider_indicator` | `1.0` if `meta.get("accidental_death_rider")` else `0.0` | `term` |

---

## 3. Worked Example (`PROF-HEA-001` × `POL-HLT-002`)

**Inputs:**
- **Profile (`PROF-HEA-001`):** `policy_type="health"`, `age=55`, `annual_income_inr=1638000`, `monthly_budget_inr=3016.67`, `budget_max=36200`, `family_size=1`, `min_sum_insured=1500000`, `required_benefits=["cashless_hospitalization", "daycare_procedures"]`, `risk_preference="comprehensive_coverage"`
- **Policy (`POL-HLT-002`):** `annual_premium_min=9400`, `annual_premium_max=31600`, `sum_insured_max=5000000`, `max_entry_age=60`, `lifelong_renewal=True`, `copay_pct=10.0`, `initial_waiting_period_days=30`, `ped_covered=True`, `ped_waiting_period_months=48`, `network_hospitals_count=8139`, `claim_settlement_ratio=95.81`, `no_claim_bonus_pct=10.0`, `covered_benefits` includes both requested benefits.

**Computed Feature Highlights:**
- `policy_type_encoded = 0.0`, `is_health = 1.0`, `is_car = 0.0`, `is_life = 0.0`, `is_term = 0.0`
- `annual_premium = (9400 + 31600) / 2 = 20500.0`, `monthly_premium = 1708.3333`
- `premium_to_budget_ratio = 20500 / 36200 = 0.5663`, `affordability_margin = 15700.0`, `affordability_score = (36200 - 9400) / 36200 = 0.7403`
- `coverage_to_requirement_ratio = 5000000 / 1500000 = 3.3333`, `coverage_headroom = 3500000.0`
- `age_eligibility_margin = 60 - 55 = 5.0`, `remaining_eligible_years = 100 - 55 = 45.0`
- `total_benefit_count = 5.0`, `matched_required_benefits = 2.0`, `required_benefit_match_ratio = 1.0`, `exclusion_conflict_count = 0.0`
- `ped_waiting_period_days = 48 * 30 = 1440.0`, `hospital_network_count = 8139.0`, `co_payment_percentage = 10.0`
- All Car (`vehicle_*`, `*_indicator`) and Life/Term (`life_cover*`, `smoking_*`) columns = `0.0`.
