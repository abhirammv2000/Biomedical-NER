# EDA Summary: BC5CDR

- Documents: **1500** (500 train / 500 dev / 500 test)
- Chemical mentions: **15935**, Disease mentions: **12850**
- Unique chemical concepts (MeSH): **1269**, unique disease concepts: **1081**
- Gold CID relations: **3116**

## Biostatistics: chemicals enriched as disease inducers
Fisher's exact test (one-sided) with Benjamini-Hochberg FDR correction. **49** chemicals are significantly enriched (q < 0.05) for chemical-induced-disease relations vs. the rest of the corpus.

Top 10 by significance:

| chemical                  | mesh    |   candidate_pairs |   cid_positive |   cid_rate |   odds_ratio |           p |        q_bh |
|:--------------------------|:--------|------------------:|---------------:|-----------:|-------------:|------------:|------------:|
| pilocarpine               | D010862 |                97 |             47 |      0.485 |         3.96 | 8.94545e-11 | 6.27076e-08 |
| Suxamethonium             | D013390 |                46 |             27 |      0.587 |         5.96 | 4.32561e-09 | 1.51613e-06 |
| puromycin aminonucleoside | D011692 |                61 |             32 |      0.525 |         4.63 | 7.48736e-09 | 1.74955e-06 |
| haloperidol               | D006220 |                76 |             36 |      0.474 |         3.78 | 2.95912e-08 | 5.18586e-06 |
| adriamycin                | D004317 |               251 |             84 |      0.335 |         2.13 | 7.29749e-08 | 1.02311e-05 |
| 13-cis-retinoic acid      | D015474 |                23 |             15 |      0.652 |         7.84 | 1.95251e-06 | 0.000228118 |
| cocaine                   | D003042 |               169 |             58 |      0.343 |         2.2  | 2.9943e-06  | 0.000299857 |
| Indomethacin              | D007213 |                30 |             17 |      0.567 |         5.47 | 6.45594e-06 | 0.000502846 |
| isoprenaline              | D007545 |                67 |             29 |      0.433 |         3.2  | 6.24856e-06 | 0.000502846 |
| indocyanine green         | D007208 |                 7 |              7 |      1     |       inf    | 1.01255e-05 | 0.000709797 |

## Figures
- `figures/mentions_by_type.png`
- `figures/mention_length.png`
- `figures/top_chemicals.png`, `figures/top_diseases.png`
- `figures/cid_degree.png`