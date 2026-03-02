# Food Insecurity and Youth Depression Risk in the U.S.

Author: **Ao Xu**  
Repository: https://github.com/bobaoxu2001/Food-insecurity

## Project Goal / 研究目标

**Research question:**  
Does household food insecurity increase the risk of depressive symptoms among adolescents and youth in the United States?

**中文问题：**  
美国家庭食物不安全是否会显著增加青少年抑郁症状风险？

---

## Data Source / 数据来源

Primary source: **NHANES (CDC)**  
https://www.cdc.gov/nchs/nhanes/

Pooled cycles used in this project:
- 2005-2006
- 2007-2008
- 2009-2010
- 2011-2012
- 2013-2014
- 2015-2016
- 2017-2018
- 2021-2022

Modules merged:
- DEMO (demographics + income + weights)
- DPQ (PHQ-9 depression screening)
- FSQ (food security)
- BMX (BMI)

> Note: FSQ food-security status uses `FSDHH` when available, and `FSDAD` for cycles where `FSDHH` is not released.

---

## Methods / 方法

1. **Data cleaning**
   - Merge by `SEQN`
   - Restrict to youth aged 12-19
   - Construct depression outcome: `PHQ-9 >= 10`
   - Construct food insecurity binary: low/very low vs full/marginal
   - Handle missing values via complete-case modeling
   - Apply pooled-cycle MEC weights (advanced weighting)

2. **Statistical model**
   - Weighted logistic regression:
     - `Depression ~ Food_Insecurity + Age + Gender + Income + BMI`
   - Output:
     - Odds Ratio (OR)
     - 95% CI
     - p-value

3. **Visualization**
   - Food insecurity vs depression prevalence
   - Adjusted OR forest plot
   - Gender-stratified prevalence and OR

---

## Reproducible Run / 一键运行

### 1) Install dependencies

```bash
python3 -m pip install --upgrade pip pandas statsmodels matplotlib seaborn scipy requests
```

### 2) Run full pipeline

```bash
python3 scripts/run_project.py
```

This script will automatically:
- Download NHANES XPT files
- Clean and merge data
- Fit models
- Export tables and figures
- Generate final PDF report

---

## Key Outputs / 主要产物

- Final report PDF  
  `report/Food_Insecurity_Youth_Depression_Risk_Ao_Xu_2005_2022.pdf`

- Processed analytic datasets  
  `data/processed/`

- Regression & prevalence tables  
  `outputs/tables/`

- Figures (prevalence, forest plot, stratified plots)  
  `outputs/figures/`

- Optional SQL merge template  
  `sql/merge_nhanes_template.sql`

---

## Current Run Snapshot / 当前运行结果摘要

From the latest full run:
- Combined sample (all ages): **82,123**
- Youth sample (12-19): **11,659**
- Complete-case analytic sample: **2,081**
- Weighted depression prevalence:
  - Food secure: **6.98%**
  - Food insecure: **17.27%**
- Main adjusted association (food insecurity -> depression):
  - **OR = 3.04**
  - **95% CI: 2.13-4.34**
  - **p < 0.001**

---

## Writing Sample Structure / 报告结构

The generated PDF follows:
- Background
- Literature Review
- Methods
- Data Processing
- Statistical Analysis
- Results
- Discussion
- Public Health Implications
- Limitations
- Future Research