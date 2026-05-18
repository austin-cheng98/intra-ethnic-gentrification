import os

import time

import logging

import requests

import numpy as np

import pandas as pd

from scipy import stats

from scipy.optimize import minimize

import warnings

 

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(message)s")

log = logging.getLogger(__name__)

 

CENSUS_API_KEY = os.environ.get("CENSUS_API_KEY")

 

ACS_VINTAGES = [2009, 2014, 2019, 2023]

 

N_BOOTSTRAP = 5000

RANDOM_SEED = 42

rng = np.random.default_rng(RANDOM_SEED)

 

MATCHING_VARS = [

    "median_hh_income",

    "pct_renter",

]

 

OUTCOME_VARS = [

    "median_gross_rent",

    "pct_renter",

    "pct_noncitizen",

]

 

STUDY_AREAS = {

    "flushing": {

        "state": "36",

        "county": "081",

        "treatment_tracts": [

            "084500", "084900", "085300", "085500",

        ],

        "control_tracts": [

            "085700", "085900", "086100", "086300", "086500",

        ],

        "treatment_year": 2018,

        "ethnic_group": "Asian",

    },

    "washington_heights": {

        "state": "36",

        "county": "061",

        "treatment_tracts": [

            "023900", "024100", "024301", "024500",

            "024700", "024900", "025100", "025300",

            "025500", "026100", "026300", "026500",

        ],

        "control_tracts": [

            "026700", "027300", "027500", "028100",

            "028300", "028700", "028900", "029500",

        ],

        "treatment_year": 2018,

        "ethnic_group": "Hispanic",

    },

    "sunset_park": {

        "state": "36",

        "county": "047",

        "treatment_tracts": [

            "007000",  # 8th Ave corridor, 42nd-46th St core

            "007200",  # 8th Ave corridor, 46th-52nd St core

            "007400",  # 8th Ave corridor, 52nd-58th St core

            "007600",  # 8th Ave corridor, 58th-65th St core

            "001800",  # Brooklyn Army Terminal / NYCEDC waterfront

            "002000",  # MADE Bush Terminal / NYCEDC waterfront

            "002200",  # South Brooklyn Marine Terminal zone

        ],

        "control_tracts": [

            "012200",  # East Sunset Park residential (east of 6th Ave)

            "012400",  # East Sunset Park residential

            "012600",  # East Sunset Park residential

            "012800",  # East Sunset Park / Borough Park fringe

            "013000",  # Borough Park fringe, outside intervention zone

            "013200",  # Borough Park fringe, outside intervention zone

        ],

        "treatment_year": 2018,

        "ethnic_group": "Asian",

    },

    "jackson_heights": {

        "state": "36",

        "county": "081",

        "treatment_tracts": [

            "027300",  # Roosevelt Ave core, 74th-82nd St (Little Colombia)

            "027500",  # Roosevelt Ave / 37th Ave, 82nd-90th St corridor

            "027700",  # 37th Ave commercial core, historic district zone

            "027900",  # Roosevelt Ave, 74th St subway hub / Little India

            "028100",  # 37th Ave south -- Colombian/Ecuadorian residential

            "028300",  # Junction Blvd -- South American residential core

        ],

        "control_tracts": [

            "028500",  # Northern Jackson Heights / Elmhurst fringe

            "028700",  # Elmhurst -- outside BID and Open Streets footprint

            "028900",  # Elmhurst residential, outside intervention zone

            "029100",  # Corona fringe, outside primary treatment corridor

            "029300",  # Corona residential, outside Roosevelt Ave BID

        ],

        "treatment_year": 2019,

        "ethnic_group": "Hispanic",

    },

}

 

ACS_VARS_BASE = {

    "B19013_001E": "median_hh_income",

    "B15002_015E": "male_bachelors",

    "B15002_032E": "female_bachelors",

    "B15002_001E": "edu_universe",

    "B25003_003E": "renter_occ",

    "B25003_001E": "total_occ",

    "B25064_001E": "median_gross_rent",

    "B25070_007E": "rent_burden_30_34",

    "B25070_008E": "rent_burden_35_39",

    "B25070_009E": "rent_burden_40_49",

    "B25070_010E": "rent_burden_50plus",

    "B25070_001E": "rent_burden_universe",

    "B05001_006E": "noncitizen_count",

    "B05001_001E": "nativity_universe",

}

 

ACS_VARS_RACE = {

    "B02001_005E": "asian_alone",

    "B03003_003E": "hispanic_any_race",

    "B01003_001E": "total_pop",

}

 

def census_get(url: str, params: dict, retries: int = 3) -> list:

    params = params.copy()

    if not CENSUS_API_KEY or CENSUS_API_KEY == "YOUR_KEY_HERE":

        params.pop("key", None)

 

    for attempt in range(retries):

        try:

            r = requests.get(url, params=params, timeout=30)

            r.raise_for_status()

            try:

                return r.json()

            except ValueError as e:

                preview = r.text[:250].replace("\n", " ")

                raise ValueError(

                    f"Census API returned non-JSON response from {r.url}: {preview}"

                ) from e

        except Exception:

            if attempt == retries - 1:

                raise

            time.sleep(2 ** attempt)

 

def fetch_acs5(year: int, variables: dict, state: str, county: str) -> pd.DataFrame:

    base = f"https://api.census.gov/data/{year}/acs/acs5"

    params = {

        "get": "NAME," + ",".join(variables.keys()),

        "for": "tract:*",

        "in": f"state:{state} county:{county}",

        "key": CENSUS_API_KEY,

    }

    raw = census_get(base, params)

    df = pd.DataFrame(raw[1:], columns=raw[0])

    df["GEOID"] = df["state"] + df["county"] + df["tract"]

    df = df.set_index("GEOID").rename(columns=variables)

    for col in variables.values():

        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["year"] = year

    return df[list(variables.values()) + ["year"]]

 

def fetch_all_acs(state: str, county: str) -> pd.DataFrame:

    frames = []

    for yr in ACS_VINTAGES:

        log.info("  Fetching ACS %d for county %s...", yr, county)

        df_base = fetch_acs5(yr, ACS_VARS_BASE, state, county)

        df_race = fetch_acs5(yr, ACS_VARS_RACE, state, county)

        df = df_base.join(df_race, rsuffix="_r")

        df["year"] = yr

        frames.append(df)

    return pd.concat(frames)

 

def derive_variables(df: pd.DataFrame, ethnic_group: str) -> pd.DataFrame:

    df = df.copy()

 

    df["bachelors_count"] = df["male_bachelors"] + df["female_bachelors"]

    df["pct_bachelors"] = (

        df["bachelors_count"] / df["edu_universe"].replace(0, np.nan) * 100

    )

    df["pct_renter"] = (

        df["renter_occ"] / df["total_occ"].replace(0, np.nan) * 100

    )

 

    burden_cols = [

        "rent_burden_30_34", "rent_burden_35_39",

        "rent_burden_40_49", "rent_burden_50plus",

    ]

    df["rent_burden_count"] = df[burden_cols].sum(axis=1)

    df["pct_rent_burden"] = (

        df["rent_burden_count"]

        / df["rent_burden_universe"].replace(0, np.nan) * 100

    )

    df["pct_noncitizen"] = (

        df["noncitizen_count"] / df["nativity_universe"].replace(0, np.nan) * 100

    )

 

    if ethnic_group == "Asian":

        df["coethnic_count"] = df["asian_alone"]

    elif ethnic_group == "Hispanic":

        df["coethnic_count"] = df["hispanic_any_race"]

    else:

        df["coethnic_count"] = np.nan

 

    df["pct_coethnic"] = (

        df["coethnic_count"] / df["total_pop"].replace(0, np.nan) * 100

    )

    return df

 

def weighted_smd_table(

    df: pd.DataFrame,

    match_vars: list,

    treatment_col: str = "treated",

    weights: pd.Series | None = None,

) -> pd.DataFrame:

    rows = []

    treated = df[df[treatment_col] == 1]

    controls = df[df[treatment_col] == 0]

    for var in match_vars:

        t_vals = treated[var].dropna()

        c_vals = controls[var].dropna()

        if t_vals.empty or c_vals.empty:

            rows.append({"variable": var, "SMD": np.nan})

            continue

 

        if weights is not None:

            c_w = weights.reindex(c_vals.index).fillna(0.0)

            c_mean = np.average(c_vals, weights=c_w) if c_w.sum() else np.nan

        else:

            c_mean = c_vals.mean()

 

        pooled_sd = np.sqrt((t_vals.var(ddof=0) + c_vals.var(ddof=0)) / 2)

        smd = abs(t_vals.mean() - c_mean) / pooled_sd if pooled_sd else np.nan

        rows.append({

            "variable": var,

            "treated_mean": round(t_vals.mean(), 3),

            "control_mean": round(c_mean, 3) if not np.isnan(c_mean) else np.nan,

            "SMD": round(smd, 3) if not np.isnan(smd) else np.nan,

        })

    return pd.DataFrame(rows)

 

def entropy_balance_match(

    df_pre: pd.DataFrame,

    treatment_ids: list,

    control_ids: list,

    match_vars: list,

) -> tuple[pd.DataFrame, list, pd.Series, pd.DataFrame]:

    all_ids = treatment_ids + control_ids

    df = df_pre.loc[df_pre.index.isin(all_ids), match_vars].copy()

    df["treated"] = df.index.isin(treatment_ids).astype(int)

    df = df.dropna(subset=match_vars)

 

    treated = df[df["treated"] == 1]

    controls = df[df["treated"] == 0]

    if treated.empty or controls.empty:

        return df.iloc[0:0].copy(), [], pd.Series(dtype=float), pd.DataFrame()

 

    x_control = controls[match_vars].to_numpy(dtype=float)

    target = treated[match_vars].mean().to_numpy(dtype=float)

    scale = df[match_vars].std(ddof=0).replace(0, 1).to_numpy(dtype=float)

    n_control = len(controls)

    initial = np.ones(n_control) / n_control

 

    def objective(w):

        weighted_mean = w @ x_control

        imbalance = ((weighted_mean - target) / scale) ** 2

        regularization = 0.001 * np.sum((w - initial) ** 2)

        return float(np.sum(imbalance) + regularization)

 

    result = minimize(

        objective,

        initial,

        method="SLSQP",

        bounds=[(0, 1)] * n_control,

        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1},

        options={"maxiter": 1000, "ftol": 1e-10},

    )

    if not result.success:

        log.warning("  Entropy balancing failed; falling back to unweighted sample.")

        weights = pd.Series(1.0, index=df.index, dtype=float)

    else:

        weights = pd.Series(1.0, index=df.index, dtype=float)

        weights.loc[controls.index] = result.x * len(treated)

 

    balance = weighted_smd_table(df, match_vars, weights=weights)

    log.info("\n  Entropy-balanced Balance (weighted SMD):\n%s",

             balance.to_string(index=False))

    poorly_balanced = balance[balance["SMD"] >= 0.25]

    if not poorly_balanced.empty:

        log.warning("  WARNING: SMD >= 0.25 for: %s",

                    poorly_balanced["variable"].tolist())

 

    return df, df.index.tolist(), weights, balance

 

def did_att(

    df_long: pd.DataFrame,

    matched_ids: list,

    treatment_ids: list,

    pre_year: int,

    post_year: int,

    outcome: str,

    weights: pd.Series | None = None,

) -> dict:

    control_ids = [i for i in matched_ids if i not in treatment_ids]

    if not treatment_ids or not control_ids:

        return {

            "outcome": outcome,

            "pre_year": pre_year,

            "post_year": post_year,

            "ATT": np.nan,

            "SE": np.nan,

            "CI_lo": np.nan,

            "CI_hi": np.nan,

            "p_value": np.nan,

            "sig": "",

            "note": "Insufficient matched treatment/control tracts",

        }

 

    df_pre = df_long[df_long["year"] == pre_year]

    df_post = df_long[df_long["year"] == post_year]

    common_ids = [

        tract_id for tract_id in matched_ids

        if tract_id in df_pre.index and tract_id in df_post.index

    ]

    common_treat = [tract_id for tract_id in common_ids if tract_id in treatment_ids]

    common_ctrl = [tract_id for tract_id in common_ids if tract_id not in treatment_ids]

 

    if not common_treat or not common_ctrl:

        return {

            "outcome": outcome,

            "pre_year": pre_year,

            "post_year": post_year,

            "ATT": np.nan,

            "SE": np.nan,

            "CI_lo": np.nan,

            "CI_hi": np.nan,

            "p_value": np.nan,

            "sig": "",

            "note": "No common pre/post matched treatment/control tracts",

        }

 

    pre_vals = df_pre.loc[common_ids, outcome]

    post_vals = df_post.loc[common_ids, outcome]

    delta = (post_vals - pre_vals).dropna()

    common_treat = [tract_id for tract_id in common_treat if tract_id in delta.index]

    common_ctrl = [tract_id for tract_id in common_ctrl if tract_id in delta.index]

 

    if not common_treat or not common_ctrl:

        return {

            "outcome": outcome,

            "pre_year": pre_year,

            "post_year": post_year,

            "ATT": np.nan,

            "SE": np.nan,

            "CI_lo": np.nan,

            "CI_hi": np.nan,

            "p_value": np.nan,

            "sig": "",

            "note": "Insufficient common outcome data",

        }

 

    if weights is None:

        weights = pd.Series(1.0, index=common_ids, dtype=float)

 

    def _control_mean(ids):

        vals = delta.loc[ids].dropna()

        w = weights.reindex(vals.index).fillna(1.0)

        return np.average(vals, weights=w) if len(vals) and w.sum() else np.nan

 

    def _att(tids, cids):

        return delta.loc[tids].mean() - _control_mean(cids)

 

    point_est = _att(common_treat, common_ctrl)

    if np.isnan(point_est):

        return {

            "outcome": outcome,

            "pre_year": pre_year,

            "post_year": post_year,

            "ATT": np.nan,

            "SE": np.nan,

            "CI_lo": np.nan,

            "CI_hi": np.nan,

            "p_value": np.nan,

            "sig": "",

            "note": "Insufficient common outcome data",

        }

 

    boot_ests = [

        _att(

            rng.choice(common_treat, size=len(common_treat), replace=True).tolist(),

            rng.choice(common_ctrl, size=len(common_ctrl), replace=True).tolist(),

        )

        for _ in range(N_BOOTSTRAP)

    ]

 

    boot_arr = np.array(boot_ests)

    se = np.nanstd(boot_arr)

    ci_lo, ci_hi = np.nanpercentile(boot_arr, [2.5, 97.5])

    z = point_est / (se + 1e-9)

    p_val = 2 * (1 - stats.norm.cdf(abs(z)))

 

    return {

        "outcome": outcome,

        "pre_year": pre_year,

        "post_year": post_year,

        "ATT": round(point_est, 3),

        "SE": round(se, 3),

        "CI_lo": round(ci_lo, 3),

        "CI_hi": round(ci_hi, 3),

        "p_value": round(p_val, 4),

        "sig": (

            "***" if p_val < 0.01 else

            "**"  if p_val < 0.05 else

            "*"   if p_val < 0.10 else ""

        ),

    }

 

def parallel_trends_test(

    df_long: pd.DataFrame,

    matched_ids: list,

    treatment_ids: list,

    pre_years: list,

    outcome: str,

) -> dict:

    from statsmodels.formula.api import ols

 

    df_pre = df_long[

        df_long.index.isin(matched_ids) & df_long["year"].isin(pre_years)

    ][[outcome, "year"]].copy()

    df_pre["treated"] = df_pre.index.isin(treatment_ids).astype(int)

    df_pre["post_pre"] = (df_pre["year"] > min(pre_years)).astype(int)

    df_pre = df_pre.dropna(subset=[outcome])

 

    if len(df_pre) < 10:

        return {

            "outcome": outcome,

            "interaction_coef": np.nan,

            "p_value": np.nan,

            "conclusion": "Insufficient data",

        }

 

    try:

        model = ols(f"{outcome} ~ treated * post_pre", data=df_pre).fit()

        coef = model.params.get("treated:post_pre", np.nan)

        pval = model.pvalues.get("treated:post_pre", np.nan)

        conclusion = (

            "Parallel trends supported (p >= 0.10)"

            if pval >= 0.10

            else "WARNING: Parallel trends may be violated (p < 0.10)"

        )

    except Exception as e:

        coef, pval, conclusion = np.nan, np.nan, f"Error: {e}"

 

    return {

        "outcome": outcome,

        "interaction_coef": round(coef, 4) if not np.isnan(coef) else np.nan,

        "p_value": round(pval, 4) if not np.isnan(pval) else np.nan,

        "conclusion": conclusion,

    }

 

def run_area(area_name: str, cfg: dict) -> dict:

    log.info("\n%s\nSTUDY AREA: %s\n%s", "=" * 60, area_name.upper(), "=" * 60)

 

    state, county = cfg["state"], cfg["county"]

    treatment_ids = [state + county + t for t in cfg["treatment_tracts"]]

    control_ids   = [state + county + c for c in cfg["control_tracts"]]

    ethnic_group  = cfg["ethnic_group"]

    tx_year       = cfg["treatment_year"]

 

    log.info("\n[1/5] Fetching ACS data...")

    df_raw = fetch_all_acs(state, county)

 

    log.info("[2/5] Deriving variables...")

    df = derive_variables(df_raw, ethnic_group)

 

    all_study_ids = treatment_ids + control_ids

    df_study = df[df.index.isin(all_study_ids)].copy()

 

    if df_study.empty:

        log.warning(

            "  WARNING: No data found for specified tracts. "

            "Check tract IDs against current Census boundary vintages."

        )

        return {}

 

    log.info("\n  Tract coverage by ACS vintage:")

    for yr in ACS_VINTAGES:

        have = set(df_study.loc[df_study["year"] == yr].index)

        missing = [tract_id for tract_id in all_study_ids if tract_id not in have]

        log.info("    %d: %d/%d tracts found%s",

                 yr, len(have), len(all_study_ids),

                 f" ({len(missing)} missing)" if missing else "")

 

    log.info("\n[3/5] Running entropy-balanced matching...")

    pre_years_all = [y for y in ACS_VINTAGES if y < tx_year]

    match_year = max(pre_years_all)

    df_pre = df_study[df_study["year"] == match_year]

 

    matched_df, matched_ids, match_weights, balance_df = entropy_balance_match(

        df_pre, treatment_ids, control_ids, MATCHING_VARS

    )

 

    matched_treat = [i for i in matched_ids if i in treatment_ids]

    matched_ctrl  = [i for i in matched_ids if i in control_ids]

    log.info(

        "  Matched: %d treatment, %d control tracts",

        len(matched_treat), len(matched_ctrl),

    )

 

    log.info("\n[4/5] Estimating DiD ATT...")

    post_years = [y for y in ACS_VINTAGES if y >= tx_year]

 

    did_results = []

    for post_yr in post_years:

        for outcome in OUTCOME_VARS:

            result = did_att(

                df_study, matched_ids, matched_treat,

                match_year, post_yr, outcome, weights=match_weights,

            )

            result["area"] = area_name

            result["match_method"] = "entropy_balancing"

            did_results.append(result)

 

    did_df = pd.DataFrame(did_results)

    log.info(

        "\n  DiD Results (pre=%d):\n%s", match_year,

        did_df[["outcome", "post_year", "ATT", "SE", "CI_lo", "CI_hi",

                "p_value", "sig"]].to_string(index=False),

    )

 

    log.info("\n[5/5] Testing parallel pre-trends...")

    trends_results = []

    for outcome in OUTCOME_VARS:

        res = parallel_trends_test(

            df_study, matched_ids, matched_treat, pre_years_all, outcome,

        )

        res["area"] = area_name

        trends_results.append(res)

        log.info(

            "  %s: coef=%.4f, p=%.4f -> %s",

            outcome,

            res["interaction_coef"] if not pd.isna(res["interaction_coef"]) else float("nan"),

            res["p_value"] if not pd.isna(res["p_value"]) else float("nan"),

            res["conclusion"],

        )

 

    return {

        "did": did_df,

        "trends": pd.DataFrame(trends_results),

        "matched_ids": matched_ids,

        "matched_treat": matched_treat,

        "matched_ctrl": matched_ctrl,

        "balance": balance_df.assign(area=area_name, match_method="entropy_balancing"),

        "weights": (

            match_weights.rename("weight")

            .reset_index()

            .rename(columns={"index": "GEOID"})

            .assign(area=area_name, treated=lambda d: d["GEOID"].isin(matched_treat).astype(int))

        ),

    }

 

def main():

    all_did = []

    all_trends = []

    all_balance = []

    all_weights = []

 

    for area_name, cfg in STUDY_AREAS.items():

        results = run_area(area_name, cfg)

        if results:

            all_did.append(results["did"])

            all_trends.append(results["trends"])

            all_balance.append(results["balance"])

            all_weights.append(results["weights"])

 

    if not all_did:

        log.error("No results produced. Check CENSUS_API_KEY and tract IDs.")

        return

 

    did_combined    = pd.concat(all_did, ignore_index=True)

    trends_combined = pd.concat(all_trends, ignore_index=True)

    balance_combined = pd.concat(all_balance, ignore_index=True)

    weights_combined = pd.concat(all_weights, ignore_index=True)

 

    out_dir = "output"

    os.makedirs(out_dir, exist_ok=True)

    did_path    = os.path.join(out_dir, "did_results.csv")

    trends_path = os.path.join(out_dir, "parallel_trends.csv")

    balance_path = os.path.join(out_dir, "matching_balance_revised.csv")

    weights_path = os.path.join(out_dir, "matching_weights_revised.csv")

    did_combined.to_csv(did_path, index=False)

    trends_combined.to_csv(trends_path, index=False)

    balance_combined.to_csv(balance_path, index=False)

    weights_combined.to_csv(weights_path, index=False)

 

    log.info(

        "\n%s\nOUTPUTS SAVED\n  DiD results     -> %s\n  Parallel trends -> %s\n  Match balance   -> %s\n  Match weights   -> %s\n%s",

        "=" * 60, did_path, trends_path, balance_path, weights_path, "=" * 60,

    )

 

if __name__ == "__main__":

    main()
