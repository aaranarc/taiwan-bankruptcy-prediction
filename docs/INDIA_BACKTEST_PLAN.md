# Plan: backtesting the model on Indian corporate defaults

**Status: plan only.** Nothing here has been built, downloaded or scraped. Every firm-specific fact (listing status, dates, events) is marked **verify**: it comes from general knowledge, not from documents checked for this plan, and must be confirmed against the primary source named in each row before use.

## 1. Question

Would this model, trained on Taiwanese listed companies (1999–2009), have flagged Indian listed companies that later defaulted, and **how many years before** the default? A second question is how many healthy peers it would have flagged by mistake.

What this plan can and can't show:
- It **can** show whether the model *ranks* Indian defaulters above comparable healthy firms, and whether that happened early enough to matter.
- It **cannot** give calibrated Indian PDs. The base rate, horizon and accounting differ, so treat outputs as rankings unless they're recalibrated (§6.3).

## 2. Labels: what counts as "distress"

A firm-year is a **defaulter** if a qualifying event happens within the **24 months after the financial statements were published**. Qualifying events:

| Event | Primary source | Notes |
|-------|----------------|-------|
| Admission to insolvency (CIRP) under the IBC by the NCLT | NCLT order (nclt.gov.in); IBBI CIRP data (ibbi.gov.in) | Use the **admission** date, not the filing date. |
| Rating downgraded to **D** (default) by CRISIL / ICRA / CARE / India Ratings / Brickwork | Rating rationale press release on the agency's site | Agencies publish default histories; use the first D on any long-term instrument. |
| A publicly disclosed missed payment on bonds or loans | Stock-exchange disclosure (BSE/NSE corporate announcements) under SEBI LODR | Usually coincides with the D rating. |
| RBI supersession of the board (financial firms) | RBI press release | For NBFCs/HFCs, which go through a special IBC route. |

**Event date** = the earliest of the qualifying events. **Why "after publication":** a FY2018 annual report (year ending 31 March 2018) may only appear in August–September 2018. The model can only use what an analyst could have read, so all timing uses the **filing date**, never the fiscal year end (this avoids look-ahead bias).

Gap to flag: the Taiwan label ("bankruptcy under Taiwan Stock Exchange business regulations", UCI dataset 572) has **no documented horizon**. The 24-month window is a choice, not something inherited from the training data. Report results for 12 and 24 months.

## 3. Universe

### 3.1 Candidate defaulters

Start with the firms named in the brief, then add non-financial IBC cases to get a usable sample. **All rows need verification**, and the listed entity must be the one whose financials are used.

| Firm | Sector | Listed entity to use (verify) | Event (verify against source) | Financial firm? | Issues to flag |
|------|--------|-------------------------------|-------------------------------|-----------------|----------------|
| IL&FS group | Infrastructure finance | The parent (IL&FS Ltd) was **not listed**, as far as I know. Candidates: IL&FS Transportation Networks, IL&FS Engineering & Construction | Payment defaults and D ratings in 2018; government superseded the board in 2018 | Parent: yes; subsidiaries: partly | Group cross-guarantees mean a subsidiary's own ratios may not show group risk. |
| DHFL (Dewan Housing Finance) | Housing finance | DHFL | Missed payments in 2019; RBI superseded the board and referred it under the IBC (FSP route) in 2019 | **Yes** | Later forensic findings alleged misstatement, so reported numbers may *understate* risk (label noise). |
| Jet Airways | Airline | Jet Airways (India) | Grounded in 2019; IBC admission in 2019 | No | Results were delayed near the end, so the last pre-default year may be missing. |
| Future Retail | Retail | Future Retail | Loan defaults 2021–22; IBC admission in 2022 | No | COVID-period distortion in FY2021. |
| Reliance Capital | Diversified financial | Reliance Capital | RBI superseded the board in 2021, IBC admission in 2021 | **Yes** | Holding-company balance sheet. |
| Bhushan Steel | Steel | Bhushan Steel | One of RBI's 2017 IBC referrals | No | |
| Jaypee Infratech | Real estate / infra | Jaypee Infratech | 2017 IBC referral | No | Real-estate accounting (inventory-heavy). |
| Alok Industries | Textiles | Alok Industries | 2017 IBC referral | No | |
| Lanco Infratech | Power / EPC | Lanco Infratech | 2017 IBC referral | No | |
| Amtek Auto | Auto components | Amtek Auto | 2017 IBC referral | No | |
| Electrosteel Steels | Steel | Electrosteel Steels | 2017 IBC referral | No | |
| Videocon Industries | Consumer durables / oil & gas | Videocon Industries | IBC admission in 2018 | No | Group consolidation issues. |
| Reliance Communications | Telecom | Reliance Communications | IBC in 2019 | No | |
| Sintex Industries | Textiles / plastics | Sintex Industries | IBC in 2021 | No | |
| Srei Infrastructure Finance | Infrastructure finance | Srei Infrastructure Finance | RBI supersession and IBC in 2021 | **Yes** | |

Target: about 40–60 non-financial defaulters. Build the list from IBBI's CIRP data, filtered to companies with a BSE/NSE listing, instead of relying on memory.

**Financial firms (DHFL, Reliance Capital, Srei, IL&FS):** analyse them **separately or exclude them**. A lender with 85% liabilities to assets is normal, while for a manufacturer it's alarming. The Taiwan model's leverage features would mislabel every NBFC as distressed. Report them as a case study, not in the headline AUC.

### 3.2 Matched healthy sample

For each defaulter-year, pick **3–5 controls** that match on:
- the same sector (NSE/BSE industry classification),
- the same fiscal year,
- total assets within ±50%,
- **no** qualifying event in the following 36 months (checked against the same sources).

Illustrative peers to check, not a final list:
- Jet Airways → InterGlobe Aviation. SpiceJet is **not** a clean control, because it had its own distress episodes.
- Future Retail → Trent, Avenue Supermarts, V-Mart, Shoppers Stop.
- Steel defaulters → Tata Steel, JSW Steel, JSPL (check each year).
- Infra/EPC defaulters → IRB Infrastructure, Ashoka Buildcon, KNR Constructions.

Matching removes the obvious size and industry effects. **It also inflates AUC** compared with a random sample of all firms, so say this next to every number. Optionally, add a second, unmatched control sample of all listed non-financial firms to estimate the false-alarm rate.

## 4. Data sources

| Source | What it gives | Cost / access | Use in this plan |
|--------|---------------|---------------|------------------|
| **Screener.in** | Standardised annual P&L, balance sheet and cash flow, typically 10+ years; Excel export for logged-in users | Free tier with export; paid tier for more | Main free source. **Manual export, one company at a time. Check the terms of use; no automated scraping.** Line items are standardised (e.g. "Reserves", not "Retained earnings"). |
| **Annual reports** (BSE/NSE filings, company websites) | Full statements, notes, auditor qualifications, **filing dates** | Free | Fill gaps (retained earnings, interest expense detail); confirm filing dates. |
| **BSE/NSE XBRL results** | Machine-readable financial results | Free | Cross-checks; mostly quarterly. |
| **MCA21** (Ministry of Corporate Affairs) | Filed financial statements, including some unlisted subsidiaries | Pay per document | Only for gaps (e.g. IL&FS entities). |
| **CMIE Prowess** (Prowess IQ / dx) | Standardised financials for listed and unlisted firms, long history, **including delisted firms**, used in most Indian default research | **Paid**: institutional/academic licence | The best option if your college library has access. It removes most survivorship and standardisation problems. |
| **Capitaline, ACE Equity** | Similar standardised databases | Paid | Alternatives to Prowess. |
| **IBBI** (ibbi.gov.in), **NCLT** orders | CIRP admissions and dates | Free | Labels. |
| **Rating agency websites** | Rating rationales, D downgrades, default histories | Free | Labels and event dates. |
| **RBI press releases** | Board supersessions of NBFCs/HFCs | Free | Labels for financial firms. |

## 5. Mapping Indian ratios to the model

**The core problem:** the Taiwan features were **pre-normalised to 0–1 by the data provider, and the exact transformation isn't documented**. For example, the "Current Ratio" column has a median of about 0.011, which is clearly not a raw current ratio. So a raw Indian current ratio of 1.3 cannot be fed into the model as it is. Don't hide this; it's the biggest limitation of the backtest.

**Option A: quantile (rank) mapping into the existing model.** For each Indian ratio:
1. compute its percentile among *all Indian firms in that fiscal year*;
2. replace it with the Taiwan training value at the same percentile, using the existing `app/models/feature_quantiles.json`.

This assumes that being at the 80th percentile of leverage means the same in both markets. The monotone constraints help: the model's direction for each ratio is fixed and economically sensible. Only about 15–20 of the 95 features can be computed from standard Indian statements. The rest would have to be set to the Taiwan median, which weakens the model in an unknown way.

**Option B (recommended): retrain a reduced model, then apply Option A.** Retrain the same pipeline (monotone XGBoost + calibration, same CV) on **only the Taiwan features that can be computed in India**. Report its Taiwan CV metrics next to the full model's, so the cost of dropping features is measured. Then score Indian firms with quantile-mapped inputs. Candidate features, each with a definition to fix in a mapping table:

| Model feature (Taiwan column) | Indian definition (from standardised statements) |
|-------------------------------|---------------------------------------------------|
| Debt ratio % | Total liabilities / total assets |
| Net worth/Assets | 1 − Debt ratio (identity in the training data) |
| Current Ratio | Current assets / current liabilities |
| Quick Ratio | (Current assets − inventories) / current liabilities |
| Cash/Total Assets | Cash and equivalents / total assets |
| Working Capital to Total Assets | (CA − CL) / total assets |
| Current Liability to Assets | Current liabilities / total assets |
| Retained Earnings to Total Assets | Reserves and surplus (retained earnings component, verify) / total assets |
| ROA(A) before interest and % after tax | (PAT + interest × (1 − tax rate)) / total assets |
| Net Income to Total Assets | PAT / total assets |
| Total debt/Total net worth | Borrowings / net worth |
| Total Asset Turnover | Revenue / total assets |
| CFO to Assets | Cash flow from operations / total assets |
| Interest Coverage Ratio | EBIT / interest. **Excluded from the monotone constraints because the Taiwan column name is ambiguous; keep it out of the reduced model too.** |

**Option C: train an Indian model.** With roughly 40–60 defaulters this would overfit, and any evaluation would be too noisy. Not recommended, except as a later comparison with nested CV.

## 6. Evaluation

### 6.1 Discrimination
ROC-AUC and PR-AUC on the matched sample, with **bootstrap 95% confidence intervals** (resample firms, not firm-years, because years of the same firm are correlated). With about 50 defaulters, expect the intervals to be wide. Report them anyway.

### 6.2 Lead time (the main output)
For each defaulter, score every fiscal year from t−5 to t−1 before the event, using only statements filed before the event date.
- **First-flag lead time** = event date − filing date of the earliest year the firm was flagged (PD ≥ the API threshold, or grade D/E). Report both definitions.
- **Persistent-flag lead time** = the same, but counting only flags that stay on in every later year. This is more conservative.
- Report the distribution (median, interquartile range), the share of defaulters flagged at least 1 year ahead, and the share never flagged.
- **Pair it with the cost:** the false-alarm rate on controls (flagged control firm-years / all control firm-years). A model that flags everyone has a great lead time.

### 6.3 Calibration
Expect PDs to be off in level. The base rate in the matched sample is set by the design (about 1 in 4–6), not by the Indian market. Option: recalibrate only the intercept on a held-out half to compare levels. Otherwise **present only ranks and flags**, and say so.

## 7. Data gaps and biases to state openly

1. **Undocumented Taiwan normalisation.** The mapping relies on the rank-equivalence assumption (§5).
2. **Survivorship.** Free sources may drop delisted firms, which are exactly the defaulters. Check each one is still downloadable. Prowess fixes this.
3. **The last year is missing.** Distressed firms often file late or not at all in the final year (Jet Airways is a candidate example, to be verified). Count and report missing firm-years; don't impute them.
4. **Misstated financials.** Fraud cases (allegations at DHFL, for example) make the reported numbers look healthier than reality. Treat them as label noise and list them.
5. **Accounting change.** India moved from Indian GAAP to Ind AS around FY2016–17 (phased by company size), which creates a break in the series. Flag the transition year per firm.
6. **Fiscal year.** India uses a March year-end; the Taiwan data is assumed to use calendar years. Lead times use filing dates, so this mostly washes out.
7. **Financial firms** don't fit the feature set (§3.1).
8. **Small sample:** wide intervals. No fine-grained claims ("model X beats model Y") without overlapping-interval caveats.
9. **Matching inflates AUC** compared with the real population (§3.2).
10. **Different era and market:** a 1999–2009 Taiwanese sample against 2015–2023 Indian defaults. A weak result is a plausible, honest outcome.

## 8. Work plan (when this is approved)

1. **Labels (no code):** assemble the defaulter list from IBBI CIRP data, keep companies with a BSE/NSE listing, and record event dates with source links in `data/india/labels.csv`. About 1–2 days.
2. **Controls:** choose 3–5 matched peers per defaulter-year and record the matching fields. About 1 day.
3. **Financials:** manual Screener exports (or Prowess if your library has it), plus the annual-report filing dates. **No automated scraping.** About 2–4 days.
4. **Mapping table:** fix the definitions in §5 in a reviewed config file (like `config/monotone_constraints.json`).
5. **Reduced model:** retrain on the mappable features and report the Taiwan CV cost (same scripts, a new feature list).
6. **Backtest script:** quantile-map, score, compute AUC with bootstrap CIs, lead times and the false-alarm rate. Write `reports/india_backtest.json` with the commit hash, following the same rules as the rest of the project.
7. **Write-up:** results, every gap from §7 that was hit, and what *not* to conclude.

## 9. What you can claim afterwards

Only what `reports/india_backtest.json` shows. For example: "*N of M* non-financial Indian defaulters were flagged at least one year before default, at a false-alarm rate of *X%* on matched peers (AUC *a* [CI *b*–*c*])". If the result is weak, the honest framing is still a good interview story: *the model didn't transfer, and here is why: undocumented normalisation, a different market and era, small sample*.
