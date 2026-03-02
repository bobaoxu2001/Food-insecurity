-- NHANES merge template (DuckDB / PostgreSQL-style SQL)
-- Purpose: merge DEMO + DPQ + FSQ + BMX by SEQN and build analysis variables.
-- Replace table names with your own loaded tables.

WITH merged AS (
    SELECT
        d.SEQN,
        d.cycle,
        d.RIDAGEYR AS age,
        d.RIAGENDR AS gender_code,
        d.INDFMPIR AS income_pir,
        d.WTMEC2YR AS wtmec2yr,
        d.SDMVSTRA AS strata,
        d.SDMVPSU AS psu,
        COALESCE(f.FSDHH, f.FSDAD) AS food_security_raw,
        b.BMXBMI AS bmi,
        p.DPQ010, p.DPQ020, p.DPQ030, p.DPQ040, p.DPQ050,
        p.DPQ060, p.DPQ070, p.DPQ080, p.DPQ090
    FROM demo d
    LEFT JOIN dpq p ON d.SEQN = p.SEQN
    LEFT JOIN fsq f ON d.SEQN = f.SEQN
    LEFT JOIN bmx b ON d.SEQN = b.SEQN
),
cleaned AS (
    SELECT
        *,
        CASE WHEN age BETWEEN 12 AND 19 THEN 1 ELSE 0 END AS in_youth_range,
        CASE
            WHEN food_security_raw IN (3, 4) THEN 1
            WHEN food_security_raw IN (1, 2) THEN 0
            ELSE NULL
        END AS food_insecure,
        (
            CASE WHEN DPQ010 BETWEEN 0 AND 3 THEN DPQ010 ELSE NULL END +
            CASE WHEN DPQ020 BETWEEN 0 AND 3 THEN DPQ020 ELSE NULL END +
            CASE WHEN DPQ030 BETWEEN 0 AND 3 THEN DPQ030 ELSE NULL END +
            CASE WHEN DPQ040 BETWEEN 0 AND 3 THEN DPQ040 ELSE NULL END +
            CASE WHEN DPQ050 BETWEEN 0 AND 3 THEN DPQ050 ELSE NULL END +
            CASE WHEN DPQ060 BETWEEN 0 AND 3 THEN DPQ060 ELSE NULL END +
            CASE WHEN DPQ070 BETWEEN 0 AND 3 THEN DPQ070 ELSE NULL END +
            CASE WHEN DPQ080 BETWEEN 0 AND 3 THEN DPQ080 ELSE NULL END +
            CASE WHEN DPQ090 BETWEEN 0 AND 3 THEN DPQ090 ELSE NULL END
        ) AS phq9_total
    FROM merged
)
SELECT
    *,
    CASE
        WHEN phq9_total >= 10 THEN 1
        WHEN phq9_total IS NOT NULL THEN 0
        ELSE NULL
    END AS depression
FROM cleaned
WHERE in_youth_range = 1;
