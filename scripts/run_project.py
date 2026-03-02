#!/usr/bin/env python3
"""
End-to-end NHANES analysis project:
Food insecurity and depression risk among U.S. youth.
"""

from __future__ import annotations

import io
import json
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm


BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{module}_{suffix}.xpt"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
REPORT_DIR = PROJECT_ROOT / "report"


@dataclass(frozen=True)
class Cycle:
    label: str
    year: int
    suffix: str


CYCLES: list[Cycle] = [
    Cycle("2005-2006", 2005, "D"),
    Cycle("2007-2008", 2007, "E"),
    Cycle("2009-2010", 2009, "F"),
    Cycle("2011-2012", 2011, "G"),
    Cycle("2013-2014", 2013, "H"),
    Cycle("2015-2016", 2015, "I"),
    Cycle("2017-2018", 2017, "J"),
    Cycle("2021-2022", 2021, "L"),
]

MODULES = ("DEMO", "DPQ", "FSQ", "BMX")
PHQ_ITEMS = [f"DPQ0{i}0" for i in range(1, 10)]


def ensure_dirs() -> None:
    for path in (RAW_DIR, PROCESSED_DIR, FIG_DIR, TABLE_DIR, REPORT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def download_file(cycle: Cycle, module: str) -> Path:
    cycle_dir = RAW_DIR / cycle.label
    cycle_dir.mkdir(parents=True, exist_ok=True)
    out_path = cycle_dir / f"{module}_{cycle.suffix}.xpt"
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    url = BASE_URL.format(year=cycle.year, module=module, suffix=cycle.suffix)
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    if not response.content.startswith(b"HEADER RECORD"):
        raise ValueError(f"Unexpected file format from URL: {url}")
    out_path.write_bytes(response.content)
    return out_path


def decode_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.decode() if isinstance(c, bytes) else c for c in df.columns]
    return df


def read_xpt(path: Path) -> pd.DataFrame:
    df = pd.read_sas(path, format="xport")
    return decode_columns(df)


def require_columns(df: pd.DataFrame, columns: Iterable[str], module: str, cycle: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"{module} {cycle} missing required columns: {missing}")


def load_cycle(cycle: Cycle) -> pd.DataFrame:
    frames: dict[str, pd.DataFrame] = {}
    for module in MODULES:
        local_path = download_file(cycle, module)
        frames[module] = read_xpt(local_path)

    demo = frames["DEMO"]
    dpq = frames["DPQ"]
    fsq = frames["FSQ"]
    bmx = frames["BMX"]

    require_columns(
        demo,
        ["SEQN", "RIDAGEYR", "RIAGENDR", "INDFMPIR", "WTMEC2YR", "SDMVSTRA", "SDMVPSU"],
        "DEMO",
        cycle.label,
    )
    require_columns(dpq, ["SEQN", *PHQ_ITEMS], "DPQ", cycle.label)
    require_columns(bmx, ["SEQN", "BMXBMI"], "BMX", cycle.label)

    if "FSDHH" in fsq.columns:
        food_status_col = "FSDHH"
    elif "FSDAD" in fsq.columns:
        food_status_col = "FSDAD"
    else:
        raise KeyError(f"FSQ {cycle.label} is missing both FSDHH and FSDAD")

    demo_sub = demo[
        ["SEQN", "RIDAGEYR", "RIAGENDR", "INDFMPIR", "WTMEC2YR", "SDMVSTRA", "SDMVPSU"]
    ].rename(
        columns={
            "RIDAGEYR": "age",
            "RIAGENDR": "gender_code",
            "INDFMPIR": "income_pir",
            "WTMEC2YR": "wtmec2yr",
            "SDMVSTRA": "strata",
            "SDMVPSU": "psu",
        }
    )
    dpq_sub = dpq[["SEQN", *PHQ_ITEMS]].copy()
    fsq_sub = fsq[["SEQN", food_status_col]].rename(columns={food_status_col: "food_security_raw"})
    bmx_sub = bmx[["SEQN", "BMXBMI"]].rename(columns={"BMXBMI": "bmi"})

    merged = demo_sub.merge(dpq_sub, on="SEQN", how="left")
    merged = merged.merge(fsq_sub, on="SEQN", how="left")
    merged = merged.merge(bmx_sub, on="SEQN", how="left")
    merged["cycle"] = cycle.label
    merged["cycle_year"] = cycle.year
    return merged


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric_cols = ["age", "gender_code", "income_pir", "wtmec2yr", "strata", "psu", "food_security_raw", "bmi", *PHQ_ITEMS]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        # Some XPT-decoded zero values appear as tiny scientific-notation floats.
        df.loc[df[col].abs() < 1e-12, col] = 0.0

    youth = df[(df["age"] >= 12) & (df["age"] <= 19)].copy()

    for col in PHQ_ITEMS:
        youth[col] = np.round(youth[col], 0)
        youth.loc[~youth[col].isin([0, 1, 2, 3]), col] = np.nan

    youth["phq9_total"] = youth[PHQ_ITEMS].sum(axis=1, min_count=len(PHQ_ITEMS))
    youth["depression"] = np.where(
        youth["phq9_total"].notna(),
        (youth["phq9_total"] >= 10).astype(int),
        np.nan,
    )

    food_map = {1.0: "Full food security", 2.0: "Marginal food security", 3.0: "Low food security", 4.0: "Very low food security"}
    youth["food_security_category"] = youth["food_security_raw"].map(food_map)
    youth["food_insecure"] = np.where(
        youth["food_security_raw"].isin([1, 2, 3, 4]),
        youth["food_security_raw"].isin([3, 4]).astype(int),
        np.nan,
    )

    youth["gender"] = youth["gender_code"].map({1.0: "Male", 2.0: "Female"})
    youth["male"] = np.where(youth["gender_code"].isin([1, 2]), (youth["gender_code"] == 1).astype(int), np.nan)
    youth["weight_adj"] = youth["wtmec2yr"] / len(CYCLES)

    analytical_cols = [
        "SEQN",
        "cycle",
        "cycle_year",
        "age",
        "gender",
        "male",
        "income_pir",
        "bmi",
        "food_security_raw",
        "food_security_category",
        "food_insecure",
        "phq9_total",
        "depression",
        "wtmec2yr",
        "weight_adj",
        "strata",
        "psu",
    ]
    analytic = youth[analytical_cols].copy()

    model_df = analytic.dropna(
        subset=["depression", "food_insecure", "age", "male", "income_pir", "bmi", "weight_adj"]
    ).copy()
    model_df = model_df[model_df["weight_adj"] > 0].copy()
    model_df["depression"] = model_df["depression"].astype(int)
    model_df["food_insecure"] = model_df["food_insecure"].astype(int)
    model_df["male"] = model_df["male"].astype(int)
    return analytic, model_df


def fit_weighted_logit(
    df: pd.DataFrame,
    predictors: list[str],
    outcome: str = "depression",
    weight_col: str = "weight_adj",
):
    X = df[predictors].astype(float)
    X = sm.add_constant(X, has_constant="add")
    y = df[outcome].astype(float)
    weights = df[weight_col].astype(float)
    # Normalize weights so their sum equals analytic N.
    # This keeps weighted point estimates while preventing artificially tiny SEs.
    weights = weights * (len(weights) / weights.sum())
    model = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=weights)
    result = model.fit()
    return result


def format_or_table(result, term_map: dict[str, str]) -> pd.DataFrame:
    ci = result.conf_int()
    rows = []
    for term, coef in result.params.items():
        if term == "const":
            continue
        lower = ci.loc[term, 0]
        upper = ci.loc[term, 1]
        rows.append(
            {
                "term": term,
                "label": term_map.get(term, term),
                "odds_ratio": float(np.exp(coef)),
                "ci_lower": float(np.exp(lower)),
                "ci_upper": float(np.exp(upper)),
                "p_value": float(result.pvalues[term]),
            }
        )
    return pd.DataFrame(rows)


def weighted_mean(x: pd.Series, w: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    w = pd.to_numeric(w, errors="coerce")
    mask = x.notna() & w.notna() & (w > 0)
    if mask.sum() == 0:
        return math.nan
    return float(np.average(x.loc[mask], weights=w.loc[mask]))


def build_prevalence_tables(model_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_food = []
    for group_value, group_name in [(0, "Food secure"), (1, "Food insecure")]:
        g = model_df[model_df["food_insecure"] == group_value]
        by_food.append(
            {
                "group": group_name,
                "n_unweighted": int(len(g)),
                "depression_prevalence_weighted": weighted_mean(g["depression"], g["weight_adj"]),
            }
        )
    by_food_df = pd.DataFrame(by_food)

    by_gender_food = []
    for gender in ["Female", "Male"]:
        for group_value, group_name in [(0, "Food secure"), (1, "Food insecure")]:
            g = model_df[(model_df["gender"] == gender) & (model_df["food_insecure"] == group_value)]
            by_gender_food.append(
                {
                    "gender": gender,
                    "group": group_name,
                    "n_unweighted": int(len(g)),
                    "depression_prevalence_weighted": weighted_mean(g["depression"], g["weight_adj"]),
                }
            )
    by_gender_food_df = pd.DataFrame(by_gender_food)
    return by_food_df, by_gender_food_df


def fit_gender_stratified_models(model_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gender in ["Female", "Male"]:
        g = model_df[model_df["gender"] == gender].copy()
        predictors = ["food_insecure", "age", "income_pir", "bmi"]
        result = fit_weighted_logit(g, predictors)
        term = "food_insecure"
        ci = result.conf_int()
        rows.append(
            {
                "gender": gender,
                "n_unweighted": int(len(g)),
                "odds_ratio": float(np.exp(result.params[term])),
                "ci_lower": float(np.exp(ci.loc[term, 0])),
                "ci_upper": float(np.exp(ci.loc[term, 1])),
                "p_value": float(result.pvalues[term]),
            }
        )
    return pd.DataFrame(rows)


def save_figures(
    prevalence_df: pd.DataFrame,
    or_df: pd.DataFrame,
    gender_prev_df: pd.DataFrame,
    strat_or_df: pd.DataFrame,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    # Figure 1: weighted prevalence by food insecurity.
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    x = prevalence_df["group"]
    y = prevalence_df["depression_prevalence_weighted"] * 100
    bars = ax1.bar(x, y, color=["#4C72B0", "#DD8452"])
    ax1.set_ylabel("Weighted depression prevalence (%)")
    ax1.set_title("Depression prevalence by household food insecurity (NHANES 2005-2018)")
    for bar, n in zip(bars, prevalence_df["n_unweighted"]):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"{bar.get_height():.1f}%\n(n={n:,})",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig1.tight_layout()
    fig1.savefig(FIG_DIR / "food_insecurity_vs_depression_prevalence.png", dpi=300)
    plt.close(fig1)

    # Figure 2: forest plot for adjusted ORs.
    forest_df = or_df.copy().sort_values("odds_ratio", ascending=True)
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(forest_df))
    ax2.errorbar(
        forest_df["odds_ratio"],
        y_pos,
        xerr=[
            forest_df["odds_ratio"] - forest_df["ci_lower"],
            forest_df["ci_upper"] - forest_df["odds_ratio"],
        ],
        fmt="o",
        color="#2A9D8F",
        ecolor="#264653",
        capsize=4,
    )
    ax2.axvline(1.0, color="red", linestyle="--", linewidth=1)
    ax2.set_xscale("log")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(forest_df["label"])
    ax2.set_xlabel("Adjusted odds ratio (log scale)")
    ax2.set_title("Forest plot: adjusted ORs for depressive symptoms")
    fig2.tight_layout()
    fig2.savefig(FIG_DIR / "adjusted_or_forest_plot.png", dpi=300)
    plt.close(fig2)

    # Figure 3: gender-stratified prevalence.
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    genders = ["Female", "Male"]
    secure_vals = []
    insecure_vals = []
    for gender in genders:
        secure_vals.append(
            float(
                gender_prev_df[
                    (gender_prev_df["gender"] == gender) & (gender_prev_df["group"] == "Food secure")
                ]["depression_prevalence_weighted"].iloc[0]
            )
            * 100
        )
        insecure_vals.append(
            float(
                gender_prev_df[
                    (gender_prev_df["gender"] == gender) & (gender_prev_df["group"] == "Food insecure")
                ]["depression_prevalence_weighted"].iloc[0]
            )
            * 100
        )
    x_idx = np.arange(len(genders))
    width = 0.35
    b1 = ax3.bar(x_idx - width / 2, secure_vals, width, label="Food secure", color="#4C72B0")
    b2 = ax3.bar(x_idx + width / 2, insecure_vals, width, label="Food insecure", color="#DD8452")
    ax3.set_xticks(x_idx)
    ax3.set_xticklabels(genders)
    ax3.set_ylabel("Weighted depression prevalence (%)")
    ax3.set_title("Depression prevalence stratified by gender")
    ax3.legend()
    for bars in [b1, b2]:
        for bar in bars:
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                f"{bar.get_height():.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig3.tight_layout()
    fig3.savefig(FIG_DIR / "gender_stratified_prevalence.png", dpi=300)
    plt.close(fig3)

    # Figure 4: gender-stratified OR for food insecurity effect.
    fig4, ax4 = plt.subplots(figsize=(7, 4.5))
    strat = strat_or_df.sort_values("odds_ratio", ascending=True)
    y_pos = np.arange(len(strat))
    ax4.errorbar(
        strat["odds_ratio"],
        y_pos,
        xerr=[strat["odds_ratio"] - strat["ci_lower"], strat["ci_upper"] - strat["odds_ratio"]],
        fmt="o",
        color="#7A5195",
        ecolor="#003F5C",
        capsize=4,
    )
    ax4.axvline(1.0, color="red", linestyle="--", linewidth=1)
    ax4.set_xscale("log")
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(strat["gender"])
    ax4.set_xlabel("Odds ratio for food insecurity (log scale)")
    ax4.set_title("Gender-stratified association: food insecurity -> depression")
    fig4.tight_layout()
    fig4.savefig(FIG_DIR / "gender_stratified_food_insecurity_or.png", dpi=300)
    plt.close(fig4)


def df_to_fixed_width_table(df: pd.DataFrame, float_fmt: str = "{:.3f}") -> str:
    if df.empty:
        return "No data."

    formatted = df.copy()
    for col in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[col]):
            formatted[col] = formatted[col].map(lambda x: "" if pd.isna(x) else float_fmt.format(x))
    col_widths = [max(len(str(col)), *(len(str(v)) for v in formatted[col])) for col in formatted.columns]
    header = " | ".join(str(col).ljust(width) for col, width in zip(formatted.columns, col_widths))
    sep = "-+-".join("-" * width for width in col_widths)
    rows = [
        " | ".join(str(val).ljust(width) for val, width in zip(row, col_widths))
        for row in formatted.itertuples(index=False, name=None)
    ]
    return "\n".join([header, sep, *rows])


def add_text_page(pdf: PdfPages, title: str, sections: list[tuple[str, str]]) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.965, title, ha="center", va="top", fontsize=17, fontweight="bold")
    y = 0.92
    for heading, content in sections:
        fig.text(0.07, y, heading, ha="left", va="top", fontsize=12.5, fontweight="bold")
        y -= 0.024
        wrapped = textwrap.fill(content, width=105)
        fig.text(0.07, y, wrapped, ha="left", va="top", fontsize=10.5, linespacing=1.35)
        line_count = wrapped.count("\n") + 1
        y -= 0.018 * line_count + 0.03
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_image_page(pdf: PdfPages, title: str, image_path: Path, caption: str) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.965, title, ha="center", va="top", fontsize=16, fontweight="bold")
    ax = fig.add_axes([0.08, 0.22, 0.84, 0.66])
    img = plt.imread(image_path)
    ax.imshow(img)
    ax.axis("off")
    fig.text(0.08, 0.14, textwrap.fill(caption, width=115), ha="left", va="top", fontsize=10.5)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_table_page(pdf: PdfPages, title: str, subtitle: str, table_text: str) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.965, title, ha="center", va="top", fontsize=16, fontweight="bold")
    fig.text(0.08, 0.93, textwrap.fill(subtitle, width=118), ha="left", va="top", fontsize=10.5)
    fig.text(0.08, 0.88, table_text, family="monospace", fontsize=8.9, ha="left", va="top")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def generate_pdf_report(
    summary: dict,
    or_df: pd.DataFrame,
    strat_or_df: pd.DataFrame,
    prevalence_df: pd.DataFrame,
) -> Path:
    report_path = REPORT_DIR / "Food_Insecurity_Youth_Depression_Risk_Ao_Xu_2005_2022.pdf"
    github_link = "https://github.com/bobaoxu2001/Food-insecurity"

    food_row = or_df.loc[or_df["term"] == "food_insecure"].iloc[0]
    food_or = food_row["odds_ratio"]
    food_ci = (food_row["ci_lower"], food_row["ci_upper"])
    food_p = food_row["p_value"]
    food_p_text = "<0.0001" if food_p < 1e-4 else f"{food_p:.4f}"

    prevalence_secure = float(prevalence_df.loc[prevalence_df["group"] == "Food secure", "depression_prevalence_weighted"].iloc[0]) * 100
    prevalence_insecure = float(prevalence_df.loc[prevalence_df["group"] == "Food insecure", "depression_prevalence_weighted"].iloc[0]) * 100

    with PdfPages(report_path) as pdf:
        add_text_page(
            pdf,
            "Food Insecurity and Youth Depression Risk in the United States",
            [
                ("Author", "Ao Xu"),
                ("Project repository", github_link),
                (
                    "Background",
                    "Food insecurity is a core social determinant of health. Adolescents exposed to household food insecurity may experience chronic stress, poorer diet quality, and psychosocial strain that contribute to depression risk.",
                ),
                (
                    "Research question",
                    "Does household food insecurity increase the risk of depressive symptoms among adolescents and youth in the U.S.?",
                ),
                (
                    "Data source",
                    f"National Health and Nutrition Examination Survey (NHANES), pooled cycles {summary['cycle_start']} to {summary['cycle_end']} ({summary['n_cycles']} cycles), using DEMO, DPQ (PHQ-9), FSQ, and BMX modules.",
                ),
            ],
        )

        add_text_page(
            pdf,
            "Literature Review and Methods",
            [
                (
                    "Literature review",
                    "Prior epidemiologic studies consistently report that food insecurity is associated with worse mental health outcomes in both adults and youth. Mechanisms include material deprivation, family stress, and reduced ability to maintain stable healthy routines.",
                ),
                (
                    "Study design",
                    "Cross-sectional observational analysis using nationally representative NHANES survey data. Primary exposure was food insecurity status (low/very low vs full/marginal), harmonized from FSQ summary variables (FSDHH and, when needed, FSDAD). Primary outcome was moderate-to-severe depressive symptoms (PHQ-9 >= 10).",
                ),
                (
                    "Population",
                    f"Participants aged 12-19 years. Initial youth sample size: {summary['n_youth']:,}; complete-case analytic sample: {summary['n_analytic']:,}.",
                ),
                (
                    "Covariates",
                    "Age (years), gender (male vs female), family income-to-poverty ratio (INDFMPIR), and BMI (kg/m^2).",
                ),
                (
                    "Model",
                    "Weighted logistic regression: Depression ~ Food_Insecurity + Age + Gender + Income + BMI. MEC weights were rescaled by the number of pooled cycles.",
                ),
            ],
        )

        add_text_page(
            pdf,
            "Data Processing and Statistical Analysis",
            [
                (
                    "Data processing",
                    "Raw NHANES XPT files were downloaded directly from CDC public endpoints and merged by SEQN. PHQ items were cleaned to valid scores (0-3), and records with special missing codes were excluded from complete-case modeling. Food security status used FSDHH when available, with FSDAD harmonized for cycles where FSDHH was not released.",
                ),
                (
                    "Missing-value handling",
                    "Complete-case analysis was used for outcome, exposure, and covariates in the main model. This approach preserves model interpretability but may introduce selection bias if missingness is not random.",
                ),
                (
                    "Main finding (adjusted association)",
                    (
                        f"Food insecurity was associated with higher odds of depressive symptoms: "
                        f"OR={food_or:.2f}, 95% CI [{food_ci[0]:.2f}, {food_ci[1]:.2f}], p={food_p_text}."
                    ),
                ),
                (
                    "Prevalence contrast",
                    (
                        f"Weighted depression prevalence was {prevalence_secure:.1f}% in food-secure youth versus "
                        f"{prevalence_insecure:.1f}% in food-insecure youth."
                    ),
                ),
            ],
        )

        add_image_page(
            pdf,
            "Results Figure 1",
            FIG_DIR / "food_insecurity_vs_depression_prevalence.png",
            "Figure 1. Weighted prevalence of depressive symptoms by household food insecurity status.",
        )
        add_image_page(
            pdf,
            "Results Figure 2",
            FIG_DIR / "adjusted_or_forest_plot.png",
            "Figure 2. Forest plot of adjusted odds ratios from the multivariable weighted logistic regression.",
        )
        add_image_page(
            pdf,
            "Results Figure 3",
            FIG_DIR / "gender_stratified_prevalence.png",
            "Figure 3. Weighted depression prevalence by food security status, stratified by gender.",
        )
        add_image_page(
            pdf,
            "Results Figure 4",
            FIG_DIR / "gender_stratified_food_insecurity_or.png",
            "Figure 4. Gender-stratified adjusted odds ratios for the association between food insecurity and depression.",
        )

        main_table = or_df[["label", "odds_ratio", "ci_lower", "ci_upper", "p_value"]].copy()
        main_table.columns = ["Predictor", "OR", "CI_Lower", "CI_Upper", "p_value"]
        add_table_page(
            pdf,
            "Key Regression Estimates",
            "Adjusted odds ratios from the weighted multivariable logistic model.",
            df_to_fixed_width_table(main_table, float_fmt="{:.4f}"),
        )

        strat_table = strat_or_df[["gender", "n_unweighted", "odds_ratio", "ci_lower", "ci_upper", "p_value"]].copy()
        strat_table.columns = ["Gender", "N", "OR_FoodInsecure", "CI_Lower", "CI_Upper", "p_value"]
        add_table_page(
            pdf,
            "Gender-Stratified Model",
            "Association between food insecurity and depression within each gender stratum.",
            df_to_fixed_width_table(strat_table, float_fmt="{:.4f}"),
        )

        add_text_page(
            pdf,
            "Discussion, Implications, and Future Research",
            [
                (
                    "Discussion",
                    "The analysis indicates a robust positive association between household food insecurity and depressive symptoms among U.S. youth, even after adjustment for demographics and socioeconomic indicators.",
                ),
                (
                    "Public health implications",
                    "Policies that reduce household food insecurity (SNAP optimization, school meal access, and targeted community food support) may provide mental health benefits in addition to nutritional gains.",
                ),
                (
                    "Limitations",
                    "Cross-sectional NHANES design cannot establish causality. Complete-case analysis may introduce bias. Residual confounding and measurement error remain possible.",
                ),
                (
                    "Future research",
                    "Future work should test longitudinal pathways, mediation through diet quality and family stress, and effect modification by race/ethnicity, geography, and social support.",
                ),
                (
                    "Reproducibility",
                    f"All code, scripts, and outputs are reproducible from the project repository: {github_link}",
                ),
            ],
        )
    return report_path


def main() -> None:
    ensure_dirs()

    print("Loading NHANES cycles...")
    all_cycles = [load_cycle(cycle) for cycle in CYCLES]
    combined = pd.concat(all_cycles, ignore_index=True)
    combined.to_csv(PROCESSED_DIR / "nhanes_combined_raw_selected_columns.csv", index=False)

    analytic, model_df = preprocess(combined)
    analytic.to_csv(PROCESSED_DIR / "nhanes_youth_analysis_dataset.csv", index=False)
    model_df.to_csv(PROCESSED_DIR / "nhanes_youth_complete_cases_for_model.csv", index=False)

    predictors = ["food_insecure", "age", "male", "income_pir", "bmi"]
    model = fit_weighted_logit(model_df, predictors)
    term_map = {
        "food_insecure": "Food insecurity (vs secure)",
        "age": "Age (years)",
        "male": "Male (vs female)",
        "income_pir": "Income-to-poverty ratio",
        "bmi": "BMI (kg/m^2)",
    }
    or_df = format_or_table(model, term_map).sort_values("label")
    or_df.to_csv(TABLE_DIR / "adjusted_logistic_or_table.csv", index=False)

    prevalence_df, gender_prev_df = build_prevalence_tables(model_df)
    prevalence_df.to_csv(TABLE_DIR / "depression_prevalence_by_food_security.csv", index=False)
    gender_prev_df.to_csv(TABLE_DIR / "depression_prevalence_by_food_security_and_gender.csv", index=False)

    strat_or_df = fit_gender_stratified_models(model_df)
    strat_or_df.to_csv(TABLE_DIR / "gender_stratified_food_insecurity_or.csv", index=False)

    cycle_counts = analytic.groupby("cycle", as_index=False).size().rename(columns={"size": "n_youth"})
    cycle_counts.to_csv(TABLE_DIR / "youth_sample_size_by_cycle.csv", index=False)

    summary = {
        "n_cycles": len(CYCLES),
        "cycle_start": CYCLES[0].label,
        "cycle_end": CYCLES[-1].label,
        "n_combined_all_ages": int(len(combined)),
        "n_youth": int(len(analytic)),
        "n_analytic": int(len(model_df)),
        "weighted_depression_prevalence_overall": weighted_mean(model_df["depression"], model_df["weight_adj"]),
        "weighted_food_insecurity_prevalence": weighted_mean(model_df["food_insecure"], model_df["weight_adj"]),
    }
    with open(TABLE_DIR / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    save_figures(prevalence_df, or_df, gender_prev_df, strat_or_df)
    report_path = generate_pdf_report(summary, or_df, strat_or_df, prevalence_df)

    print("Analysis complete.")
    print(f"Combined sample (all ages): {summary['n_combined_all_ages']:,}")
    print(f"Youth sample (12-19): {summary['n_youth']:,}")
    print(f"Analytic complete-case sample: {summary['n_analytic']:,}")
    print(f"Report generated: {report_path}")


if __name__ == "__main__":
    main()
