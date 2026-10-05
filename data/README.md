# Migration data

`migration_un2024.npz` contains `R` (float64, destination rows and origin columns), `b = R.sum(axis=1)`, and the sorted integer location `codes` used on both axes.

The source is the public Kaggle mirror of the UN DESA International Migrant Stock 1990–2024 data:
https://www.kaggle.com/datasets/muneebhassanf/un-international-migrant-stock-1990-2024

Download endpoint:
https://www.kaggle.com/api/v1/datasets/download/muneebhassanf/un-international-migrant-stock-1990-2024

The original UN workbook was unavailable to this reproduction (the earlier download attempt returned HTTP 403). The included data are explicitly from the mirror; byte-level equivalence with the original workbook is not claimed. The extracted CSV is retained; SHA-256 hashes of it and the originally downloaded archive are recorded in `migration_provenance.json`.

Construction follows the uploaded script: select Year = 2024, destination code < 900, origin code < 900; use the `Total` column (both sexes), remove spaces from number strings, and treat missing/nonpositive entries as zero. The sorted origin codes form both axes. The 233 codes include the `Others` origin category; destination rows absent from the source are zero. There are no duplicate destination-origin pairs after filtering.

Checks: shape 233 × 233; 9,042 strictly positive entries; sum 281,956,199. The SHA-256 of the C-order little-endian float64 matrix is `dca70176e1729919bc64b498a9f120bce2ba5364de8bca5f1acd164ee65d4a72`.

The fixed split uses `np.argwhere(R > 0)`, `np.random.default_rng(2024).permutation`, and `round(0.10 * 9042)`: 904 test and 8,138 training entries. The full matrix is the evaluation target; an algorithm initialization must not access held-out values. Row totals are supplied as side information, as specified in the manuscript.
