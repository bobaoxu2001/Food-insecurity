#!/usr/bin/env python3
"""
End-to-end NHANES analysis project:
Food insecurity and depression risk among U.S. youth.
"""

from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import BaseDocTemplate, Frame, Image, PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle
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


def format_p_value(p_value: float) -> str:
    return "<0.0001" if p_value < 1e-4 else f"{p_value:.4f}"


def build_report_styles() -> dict[str, ParagraphStyle]:
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#16324F"),
            spaceAfter=16,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=12,
            leading=16,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#334155"),
            spaceAfter=10,
        ),
        "section": ParagraphStyle(
            "SectionHeading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#1E3A5F"),
            spaceBefore=8,
            spaceAfter=8,
        ),
        "subsection": ParagraphStyle(
            "SubsectionHeading",
            parent=styles["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=11.2,
            leading=14,
            textColor=colors.HexColor("#334155"),
            spaceBefore=5,
            spaceAfter=3,
        ),
        "body": ParagraphStyle(
            "BodyTextCustom",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10.6,
            leading=15,
            alignment=TA_JUSTIFY,
            textColor=colors.HexColor("#111827"),
            spaceAfter=7,
        ),
        "bullet": ParagraphStyle(
            "BulletTextCustom",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10.6,
            leading=14,
            leftIndent=12,
            bulletIndent=0,
            textColor=colors.HexColor("#111827"),
            spaceAfter=4,
        ),
        "caption": ParagraphStyle(
            "FigureCaption",
            parent=styles["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=9.4,
            leading=12,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#374151"),
            spaceAfter=10,
        ),
        "small": ParagraphStyle(
            "SmallMeta",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9.4,
            leading=12,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#4B5563"),
            spaceAfter=4,
        ),
    }


def draw_report_page(canvas, doc) -> None:
    page = canvas.getPageNumber()
    width, height = LETTER
    canvas.saveState()

    # Header (starting from page 2 to keep the title page clean)
    if page > 1:
        header_y = height - 0.68 * inch
        canvas.setStrokeColor(colors.HexColor("#D1D5DB"))
        canvas.setLineWidth(0.8)
        canvas.line(doc.leftMargin, header_y, width - doc.rightMargin, header_y)
        canvas.setFillColor(colors.HexColor("#374151"))
        canvas.setFont("Helvetica", 9)
        canvas.drawString(
            doc.leftMargin,
            height - 0.53 * inch,
            "Food Insecurity and Youth Depression Risk in the United States",
        )

    # Footer with page number.
    canvas.setFillColor(colors.HexColor("#6B7280"))
    canvas.setFont("Helvetica", 9)
    canvas.drawRightString(width - doc.rightMargin, 0.5 * inch, f"Page {page}")
    canvas.restoreState()


def styled_table(data: list[list[str]], col_widths: list[float]) -> Table:
    table = Table(data, colWidths=col_widths, hAlign="LEFT", repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3A5F")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 1), (-1, -1), 9.3),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def add_figure(story: list, image_path: Path, caption: str, styles: dict[str, ParagraphStyle]) -> None:
    image = Image(str(image_path))
    max_width = 6.35 * inch
    max_height = 4.35 * inch
    scale = min(max_width / image.imageWidth, max_height / image.imageHeight)
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    image.hAlign = "CENTER"

    story.append(image)
    story.append(Spacer(1, 0.08 * inch))
    story.append(Paragraph(caption, styles["caption"]))
    story.append(Spacer(1, 0.15 * inch))


def generate_pdf_report(
    summary: dict,
    or_df: pd.DataFrame,
    strat_or_df: pd.DataFrame,
    prevalence_df: pd.DataFrame,
) -> Path:
    report_path = REPORT_DIR / "Food_Insecurity_Youth_Depression_Risk_Ao_Xu_2005_2022.pdf"
    github_link = "https://github.com/bobaoxu2001/Food-insecurity"
    styles = build_report_styles()

    food_row = or_df.loc[or_df["term"] == "food_insecure"].iloc[0]
    food_or = food_row["odds_ratio"]
    food_ci = (food_row["ci_lower"], food_row["ci_upper"])
    food_p = food_row["p_value"]
    food_p_text = format_p_value(food_p)

    prevalence_secure = (
        float(prevalence_df.loc[prevalence_df["group"] == "Food secure", "depression_prevalence_weighted"].iloc[0]) * 100
    )
    prevalence_insecure = (
        float(prevalence_df.loc[prevalence_df["group"] == "Food insecure", "depression_prevalence_weighted"].iloc[0]) * 100
    )

    doc = BaseDocTemplate(
        str(report_path),
        pagesize=LETTER,
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
        topMargin=0.95 * inch,
        bottomMargin=0.75 * inch,
        title="Food Insecurity and Youth Depression Risk in the United States",
        author="Ao Xu",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main_frame")
    doc.addPageTemplates([PageTemplate(id="report", frames=[frame], onPage=draw_report_page)])

    story: list = []

    # Cover section.
    story.append(Spacer(1, 0.8 * inch))
    story.append(Paragraph("Food Insecurity and Youth Depression Risk in the United States", styles["title"]))
    story.append(Paragraph("A National Epidemiologic Analysis Using NHANES 2005-2022", styles["subtitle"]))
    story.append(Paragraph("Author: <b>Ao Xu</b>", styles["small"]))
    story.append(
        Paragraph(
            "Repository: <a href='https://github.com/bobaoxu2001/Food-insecurity'>https://github.com/bobaoxu2001/Food-insecurity</a>",
            styles["small"],
        )
    )
    story.append(Spacer(1, 0.35 * inch))

    executive_rows = [
        ["Indicator", "Value"],
        ["Combined sample (all ages)", f"{summary['n_combined_all_ages']:,}"],
        ["Youth sample (12-19 years)", f"{summary['n_youth']:,}"],
        ["Complete-case analytic sample", f"{summary['n_analytic']:,}"],
        ["Depression prevalence in food-secure youth", f"{prevalence_secure:.1f}%"],
        ["Depression prevalence in food-insecure youth", f"{prevalence_insecure:.1f}%"],
        [
            "Adjusted OR for food insecurity",
            f"{food_or:.2f} (95% CI {food_ci[0]:.2f}-{food_ci[1]:.2f}), p {food_p_text}",
        ],
    ]
    story.append(styled_table(executive_rows, col_widths=[2.8 * inch, 3.6 * inch]))
    story.append(Spacer(1, 0.2 * inch))
    story.append(
        Paragraph(
            "Research question: Does household food insecurity increase the risk of depressive symptoms among adolescents and youth in the United States?",
            styles["body"],
        )
    )
    story.append(PageBreak())

    # Background and methods.
    story.append(Paragraph("1. Background and Literature Context", styles["section"]))
    story.append(
        Paragraph(
            "Food insecurity is a major social determinant of health linked to both physical and psychological burden. "
            "Adolescents exposed to unstable household food access may face chronic stress, social strain, and reduced diet quality, "
            "which can collectively elevate depression risk.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "Prior epidemiologic evidence has reported consistent associations between food insecurity and poor mental health outcomes. "
            "This report evaluates whether those patterns persist in a pooled, nationally representative U.S. youth sample.",
            styles["body"],
        )
    )

    story.append(Paragraph("2. Data and Study Design", styles["section"]))
    story.append(
        Paragraph(
            f"Data were drawn from the National Health and Nutrition Examination Survey (NHANES), pooled from "
            f"{summary['cycle_start']} through {summary['cycle_end']} ({summary['n_cycles']} cycles).",
            styles["body"],
        )
    )
    for bullet in [
        "Modules merged: DEMO, DPQ (PHQ-9), FSQ, and BMX.",
        "Target population: youth aged 12-19 years.",
        "Primary outcome: depressive symptoms (PHQ-9 >= 10).",
        "Primary exposure: food insecurity (low/very low vs full/marginal).",
        "Food-security variable harmonization: FSDHH when available, FSDAD otherwise.",
        "Covariates: age, sex, income-to-poverty ratio, and BMI.",
    ]:
        story.append(Paragraph(bullet, styles["bullet"], bulletText="•"))

    story.append(Paragraph("3. Statistical Analysis", styles["section"]))
    story.append(
        Paragraph(
            "Weighted logistic regression was fit as: Depression ~ Food Insecurity + Age + Sex + Income + BMI. "
            "MEC weights were pooled across cycles and normalized to preserve weighted point estimates while keeping inference stable.",
            styles["body"],
        )
    )
    story.append(PageBreak())

    # Results narrative and figures.
    story.append(Paragraph("4. Results", styles["section"]))
    story.append(
        Paragraph(
            f"Food insecurity showed a strong positive association with depressive symptoms "
            f"(OR {food_or:.2f}, 95% CI {food_ci[0]:.2f}-{food_ci[1]:.2f}, p {food_p_text}). "
            f"Weighted prevalence was {prevalence_secure:.1f}% among food-secure youth and {prevalence_insecure:.1f}% among food-insecure youth.",
            styles["body"],
        )
    )
    add_figure(
        story,
        FIG_DIR / "food_insecurity_vs_depression_prevalence.png",
        "Figure 1. Weighted prevalence of depressive symptoms by household food insecurity status.",
        styles,
    )
    add_figure(
        story,
        FIG_DIR / "adjusted_or_forest_plot.png",
        "Figure 2. Forest plot of adjusted odds ratios from the multivariable weighted logistic model.",
        styles,
    )
    story.append(PageBreak())
    add_figure(
        story,
        FIG_DIR / "gender_stratified_prevalence.png",
        "Figure 3. Weighted depression prevalence by food-security status, stratified by gender.",
        styles,
    )
    add_figure(
        story,
        FIG_DIR / "gender_stratified_food_insecurity_or.png",
        "Figure 4. Gender-stratified adjusted odds ratios for the food insecurity and depression association.",
        styles,
    )

    # Regression tables.
    story.append(PageBreak())
    story.append(Paragraph("5. Regression Results Tables", styles["section"]))
    story.append(
        Paragraph(
            "Table 1 presents adjusted associations from the pooled weighted logistic model.",
            styles["body"],
        )
    )
    table1 = [["Predictor", "OR", "95% CI", "p-value"]]
    for row in or_df.itertuples(index=False):
        table1.append(
            [
                row.label,
                f"{row.odds_ratio:.2f}",
                f"{row.ci_lower:.2f} to {row.ci_upper:.2f}",
                format_p_value(row.p_value),
            ]
        )
    story.append(styled_table(table1, col_widths=[2.7 * inch, 1.0 * inch, 1.7 * inch, 0.9 * inch]))
    story.append(Spacer(1, 0.2 * inch))

    story.append(
        Paragraph(
            "Table 2 presents gender-stratified adjusted odds ratios for the food insecurity effect.",
            styles["body"],
        )
    )
    table2 = [["Gender", "N", "OR (Food insecurity)", "95% CI", "p-value"]]
    for row in strat_or_df.itertuples(index=False):
        table2.append(
            [
                row.gender,
                f"{int(row.n_unweighted):,}",
                f"{row.odds_ratio:.2f}",
                f"{row.ci_lower:.2f} to {row.ci_upper:.2f}",
                format_p_value(row.p_value),
            ]
        )
    story.append(styled_table(table2, col_widths=[1.2 * inch, 0.9 * inch, 1.4 * inch, 1.8 * inch, 1.0 * inch]))

    # Discussion and implications.
    story.append(PageBreak())
    story.append(Paragraph("6. Discussion and Public Health Implications", styles["section"]))
    story.append(
        Paragraph(
            "This analysis indicates a robust association between household food insecurity and depressive symptoms in U.S. youth. "
            "The relationship remains substantial after adjustment for age, sex, economic status, and BMI.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "From a public health perspective, food access interventions may deliver dual benefits across nutritional and mental health domains. "
            "Potential policy channels include school meal optimization, targeted food assistance, and family-centered prevention strategies.",
            styles["body"],
        )
    )

    story.append(Paragraph("7. Limitations and Future Research", styles["section"]))
    for bullet in [
        "Cross-sectional NHANES design limits causal interpretation.",
        "Complete-case analysis can introduce bias under non-random missingness.",
        "Residual confounding and measurement error remain possible.",
        "Future work should test mediation pathways and longitudinal relationships.",
    ]:
        story.append(Paragraph(bullet, styles["bullet"], bulletText="•"))

    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("Reproducibility", styles["subsection"]))
    story.append(
        Paragraph(
            f"All code, data-processing scripts, tables, and figures are available in the project repository: "
            f"<a href='{github_link}'>{github_link}</a>.",
            styles["body"],
        )
    )

    doc.build(story)
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
