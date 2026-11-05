# Learning notes

Plain-English notes on what changed in each phase and why, with interview questions you should be able to answer.
All numbers here come from `reports/metrics.json`. If you rerun `python -m training.cv_evaluate`, re-read that file instead of trusting this page.

---

## Phase 0: fixing what was wrong

### 1. The README said "SMOTE", but the code doesn't use it

`training/train_model.py` handles class imbalance with XGBoost's `scale_pos_weight`, not SMOTE.

- **SMOTE** creates synthetic bankrupt firms by interpolating between real ones.
- **`scale_pos_weight`** keeps the data as it is. It makes each bankrupt firm count about 30× more in the loss (30 = healthy firms ÷ bankrupt firms in the training fold).

The README now says `scale_pos_weight`, and `imbalanced-learn` (the SMOTE library) has been removed from `requirements.txt` because nothing imports it. The "RF + SMOTE" row was dropped for the same reason.

### 2. Altman Z-score: sign, and why the zones were removed

Altman's Z = 1.2·X1 + 1.4·X2 + 3.3·X3 + 0.6·X4 + 1.0·X5. A **lower** Z means **more** risk. ROC-AUC assumes a **higher** score means more risk, so you must pass **−Z**. Passing +Z gives you 1 − AUC, which is how the original README ended up with 0.09.

The 1.81 / 2.99 zone cut-offs were fitted on raw ratios of 1960s US manufacturers. In this dataset every ratio is already squashed into 0–1, so a Z value of, say, 1.2 has no meaning on Altman's scale.

**Choice made:** the zones (and the raw Z number) were removed from the API and UI. Altman stays only as a **benchmark ROC-AUC/PR-AUC**, because AUC depends only on the *ranking* of firms, which survives the rescaling. Keeping a Z number in the UI would invite people to read it against cut-offs that don't apply.

### 3. Cross-validation instead of one 80/20 split

The old test set had **44** bankrupt firms. One extra miss moved recall by 2.3 points, and there was no way to tell luck from skill.

Now `training/cv_evaluate.py` runs **5×5 repeated stratified K-fold**:
- **Stratified:** every fold keeps the 3.2% bankrupt rate.
- **5 folds:** each firm is tested exactly once per repeat.
- **Repeated 5×:** reshuffled 5 times. That gives 25 scores, so we report mean ± standard deviation.
- **No leakage:** winsorisation bounds and the scaler are re-fitted on each training fold only, never on the test fold.

### 4. A bug found along the way: RobustScaler and logistic regression

The old logistic regression baseline scored ~0.5 ROC-AUC (a coin flip). The cause: `RobustScaler` divides by the inter-quartile range (IQR), and 22 columns have an IQR near zero, so their scaled values reach about 10¹³. Tree models (Random Forest, XGBoost) don't care, because a split like "x > 5" works the same after any order-preserving transform. A linear model, though, gets weights that blow up. LR now uses `StandardScaler` on the winsorised data and scores about 0.90. The deployed XGBoost pipeline is unchanged.

### 5. One source of truth for numbers

`reports/metrics.json` records every metric, the git commit, library versions and caveats. `training/render_readme.py` writes the README table from it, and the Streamlit overview reads it at runtime. Nothing is typed by hand.

### Known caveats (also stored in metrics.json)

- The XGBoost hyperparameters in `app/models/best_params.pkl` were tuned on the old 80% train split, which overlaps the CV test folds, so the XGBoost row is slightly optimistic. The fix would be nested CV.
- Those parameters are **not** in `train_model.py`'s grid, so the deployed model was probably tuned in the notebook. Phase 1 retrains from the script.
- Recall at 0.50 isn't comparable across models: Random Forest's probabilities are compressed, so few firms cross 0.50 even when the ranking is good. Compare models on AUC. Phase 1 replaces 0.50 with a cost-based threshold.

### Interview questions

**Q: Why ROC-AUC *and* PR-AUC?**
ROC-AUC asks: "Pick a random bankrupt firm and a random healthy one. How often is the bankrupt one scored higher?" With 97% healthy firms, that can look great while most alerts are false. PR-AUC focuses on the positive class, so it's the stricter number here. A random model's PR-AUC is about the base rate (0.032).

**Q: Why is the Altman AUC so high if the zones don't work?**
AUC only uses the ranking. The rescaling preserved the ranking, so −Z still separates firms well. It just can't be read on Altman's absolute scale.

**Q: What is data leakage, and how did you avoid it?**
Leakage means using information from the test data during training. Here the winsorisation percentiles and the scaler's median/IQR are computed from the training fold only, then applied to the test fold.

**Q: Why `scale_pos_weight` instead of SMOTE?**
It's simpler, it creates no synthetic firms, and it can't leak synthetic points across folds. SMOTE must be applied inside each training fold only, which is a common mistake.

**Q: Why repeated CV rather than a single split?**
With 220 bankrupt firms in total, a single test set holds only 44 of them. Repeating gives a spread, so you can say "0.94 ± 0.01" and know whether a change beats the noise.

**Q: Is standard deviation across CV folds a confidence interval?**
Not exactly. Folds share training data, so they aren't independent and the std understates the true uncertainty a little. It's still a fair, honest indication of spread.


---

## Phase 1: making the model credit-risk grade

*Numbers below are quoted from `reports/metrics.json` (commit `f211501`). If you rerun the pipeline, re-read that file.*

Pipeline: `python -m training.cv_evaluate` (honest evaluation, picks the calibration method and the threshold) → `python -m training.train_model` (refits on all data, writes `app/models/`) → `python -m training.render_readme`.

### 1. Monotone constraints

A tree model can learn odd local wiggles, for example "debt ratio 0.31 is safer than 0.30". A credit officer can't defend that. XGBoost's `monotone_constraints` forbid it: for a feature marked +1, raising it may never lower the predicted risk. The list is in `config/monotone_constraints.json`. Each of the 19 features has a written reason, and the excluded ones have a reason too. For example, the "Interest Coverage Ratio" column is *excluded*: its name says "interest expense to EBIT", which is the inverse of coverage, so the direction can't be verified.

**Cost:** paired by fold, ROC-AUC changed by +0.0003 and PR-AUC by -0.0030. Both are far smaller than the fold-to-fold noise (0.025 for PR-AUC). The constraints are essentially free here. The test `tests/test_model.py::test_pd_is_monotone_in_constrained_feature` sweeps every constrained feature on the final model and checks it.

Why the direction survives preprocessing: winsorising (clipping) and RobustScaler ((x − median) / IQR) are both order-preserving, so "higher raw value" still means "higher model input".

### 2. Calibration: turning a score into a PD

`scale_pos_weight` makes the model treat each bankrupt firm as about 30 firms, so its raw "probabilities" are far too high (see the orange line in `reports/calibration.png`). Calibration fits a mapping from raw score to real frequency:

- **Platt / sigmoid:** fits a logistic curve. It's strictly increasing, so the ranking (AUC) is unchanged.
- **Isotonic:** fits any non-decreasing step function. It's more flexible, but it creates ties and needs more data.

**No leakage:** inside each outer training fold, `CalibratedClassifierCV(cv=inner 5-fold, ensemble=False)` gets out-of-fold scores *from that training fold*, fits the calibrator on them, then refits the model on the whole training fold. The outer test fold is never touched.

**Choice:** isotonic had a Brier score lower by only 0.00015 (std of the paired gain 0.00060), but it lost PR-AUC (0.399 vs 0.436). The first run picked isotonic by pure "lowest Brier". I changed the rule (it's written in the JSON) to "isotonic only if its gain beats the noise", so **sigmoid** is used. Say this openly in interviews: it's a decision made after seeing results, and it's justified because the gain was smaller than the noise.

Brier score = mean of (PD − outcome)². Lower is better. Calibrated: 0.0236; uncalibrated monotone model: 0.0746. For reference, always predicting the base rate would give about 0.0312.

The reliability plot uses a log scale, so decile bins with *zero* observed defaults can't be drawn. That's why fewer than 10 points appear.

### 3. Cost-based threshold

0.50 is only right if a missed default and a false alarm cost the same. With a 10:1 cost ratio, we pick the PD threshold that minimises `10·(missed defaults) + 1·(false alarms)` on the pooled out-of-fold PDs: **0.15**. Theory says that for perfectly calibrated PDs the answer is 1/(1+10) ≈ 0.091, so the empirical value is in the right neighbourhood.

At 0.15: precision 30.8%, recall 70.9% (non-nested run). The threshold is in every `/predict` response (`threshold`, `flagged`). The nested-CV section below re-chooses it inside every training fold and compares it with the uncalibrated model at 0.50.

### 4. Risk grades A–E

Grades are PD bands set in `config/risk_policy.json` *before* looking at results: A < 1%, B < 3%, C < 10%, D < 30%, E the rest. The check that matters is that the observed default rate rises with the grade: A 0.3% → B 3.0% → C 5.2% → D 13.4% → E 41.9%. It does, so the grades are **rank-ordered**. That isn't a calibration claim: within a band the predicted and observed rates can differ (the README and Model card show them side by side), so say "rank-ordered", not "calibrated grades".

### 5. Counterfactual "path to safety" (`POST /predict/counterfactual`)

Question: "What is the smallest realistic change that would get this firm to grade C?" `app/services/scenarios.py` does a bounded grid search:
- **Only levers management controls:** debt ratio, short-term liabilities, borrowing dependency, current ratio, cash, retained earnings (`adjustable_features` in the policy). Every lever is monotone-constrained, so we know which way is "safer".
- **Bounded:** candidate values move only in the safer direction and stay between the 1st and 99th percentile of the training data. The model has never seen anything beyond that.
- **Smallest:** fewest ratios first (1, then 2, then 3), then the smallest total move in percentile points. Percentiles make "0.01 of current ratio" and "0.01 of cash/assets" comparable.
- **Linked ratios:** `Net worth/Assets = 1 − Debt ratio %` exactly in this data, so changing one updates the other.

Limits to admit: the other ratios are held fixed, even though in reality paying down debt also changes cash. That's why it's a "path", not a plan.

### 6. Stress test (`POST /predict/stress`)

The shocks and the exact columns each one touches are listed in `config/risk_policy.json` and returned by `GET /meta`:
- revenue −x% → total asset turnover and revenue per share scale by (1 − x)
- interest cost +x% → interest-bearing debt interest rate scales by (1 + x)
- debt +x% → debt ratio, debt/net worth and liability/equity scale up; net worth/assets = 1 − debt ratio
- profitability → ROA and net-income ratios move down by N *percentile points*. These columns are offset-normalised (about 0.5 = break-even), so "−20%" would be meaningless for them.

Results are clipped to the training range. The honest caveat: these are approximations on pre-normalised ratios, not an accounting model.

### 7. Engineering changes worth knowing

- **Exact version pins** (`requirements-api.txt`): a pickled scikit-learn model must be loaded by the version that saved it.
- **The quantile table is saved as JSON**, not a pickle. pandas 3 pickled the column labels as pyarrow objects, and the API image (correctly) has no pyarrow. Only *running* the Docker image caught this, so CI now runs the image as well as building it.
- **`xgboost-cpu` on Linux:** the same library without the ~300 MB CUDA dependency.
- **Redis URL is never logged:** managed Redis URLs contain the password. The old code logged it on every start.

### Interview questions

**Q: What does a monotone constraint cost you, and how did you measure it?**
The same 25 folds with and without constraints, compared fold by fold. The PR-AUC difference was -0.0030 with a standard deviation of 0.025, so there was no measurable cost.

**Q: Why calibrate if AUC doesn't change?**
AUC only measures ranking. Pricing, provisioning and a cost-based threshold need the *level* of the probability to be right.

**Q: How do you calibrate without leaking?**
The calibrator is fitted on out-of-fold predictions *within* the training fold, and the outer test fold is only used for scoring.

**Q: Why isn't your threshold 0.5?**
Because a missed default is assumed to cost 10× a false alarm. With calibrated PDs, the cost-minimising cut-off is near 1/(1+10); the actual value is chosen from out-of-fold predictions inside each training fold.

**Q: How do you know the grades work?**
They're rank-ordered: observed default rates rise strictly from A to E on nested out-of-fold predictions. I show mean predicted PD next to the observed rate for each grade rather than claiming the bands are calibrated.

**Q: Isn't a counterfactual just gaming the model?**
It's bounded to ratios management can change, within ranges the model has seen, in the direction the constraints guarantee is safer. It's a conversation starter for an analyst, not advice.

**Q: What is SHAP showing in the waterfall?**
Each feature's contribution to the XGBoost log-odds *before* calibration. Calibration is a single increasing curve applied afterwards, so the order and sign of contributions still apply to the PD.


---

## Phase 2: moving hosting to Vercel (Python only)

Full details: [DEPLOYMENT.md](DEPLOYMENT.md).

**Shape:** one Vercel project with two *services*. `web` is Streamlit, run as a Python ASGI app (`server.py` wraps `streamlit.App`). `api` is the existing FastAPI app, built from the root `Dockerfile` and run on Vercel's *Container Runtime*. Only `web` has a public route. `api` is reachable only through a **service binding**: Vercel injects its private URL into `web` as `API_INTERNAL_URL`.

**Why the API is a container, not a Python function:** the UI's dependencies alone measured about 424 MB against the 500 MB Python function limit. XGBoost, SHAP, scikit-learn and SciPy wouldn't fit beside them. A container image can be up to 15 GB; ours is 1.13GB locally.

**Why the URL and secrets can't reach the browser:** all Streamlit Python runs on the server. The browser only receives rendered widgets over a WebSocket. The one place that could leak the URL is an error message (a `requests` exception contains the host), so `ui/api.py` replaces it with a generic message, and a test checks this.

**Redis is optional by design:** with no `REDIS_URL`, or a Redis that's down, `get` returns a miss and `set` returns False. The prediction still works, just uncached. Tests cover "not configured", "down at start-up" and "fails mid-request".

**Cold start:** measured locally, not assumed: 1.58 s from `docker run` to the first answer on Darwin arm64. Vercel adds VM start and image fetch on top. Containers scale to zero after 5 idle minutes in production, so the first user after a quiet spell waits. DEPLOYMENT.md has the steps to measure it on Vercel.

**Session state is temporary:** a refresh or a new function instance means a new Streamlit session, so you sign in again. Anything that must survive (users, history) is in Supabase.

### Interview questions

**Q: Why not put everything in one container?**
You could, but Vercel serves Streamlit natively as a Python function (cheap, scales with traffic), while the heavy model needs a container. Separate services also mean a UI change doesn't rebuild the model image.

**Q: What does a service binding give you that an env var with a URL doesn't?**
It's private (no public route), deployment-aware (a preview UI talks to the same preview's API), and Vercel generates it, so there's no hard-coded hostname. It doesn't authenticate, which is why the API still checks JWTs.

**Q: How do you know the app works without Redis?**
The cache wrapper swallows connection errors and returns a miss, and there are tests for "not configured", "down at start-up" and "fails mid-request".

**Q: What would you check before removing Render?**
That a preview *and* production deployment both return a real prediction that's logged to Supabase. Then suspend Render first, so the step can be reversed, before deleting it.

---

## Phase 3: making the Streamlit UI look like a product

**Theme first, CSS second.** Colours, fonts, borders, radius and chart palettes are set in `.streamlit/config.toml`: a deep navy/slate base with a single blue accent. `ui/styles.css` only does what the theme can't: it hides Streamlit's menu, toolbar, decoration bar and footer, and styles our own `cr-*` cards and badges. It targets Streamlit's `data-testid` attributes, which are stable because Streamlit's own tests depend on them, unlike the auto-generated CSS class names. Green, amber and red are used **only** for risk (grades, SHAP direction, flags), so colour always means something.

**Structure.** `streamlit_app.py` only sets up the page and `st.navigation`. Each page is a small script in `ui/pages/`. Shared pieces (cards, badge, gauge, waterfall, path card) are in `ui/components.py`, and every Plotly chart uses the one template registered in `ui/theme.py`.

**Design decisions you can explain:**
- **The PD gauge gives each grade an equal-width band.** PD bands are very uneven (A is under 1%, E is 30–100%). On a linear 0–100% dial, grades A–C would share the first tenth. The needle is interpolated inside its band, and a white tick marks the decision threshold.
- **The SHAP waterfall adds up exactly.** The API returns the base value, the top 10 contributions and "all other features" summed, so the bars land on the model's score. A test checks this to 0.001.
- **The path-to-safety card reports percentiles** ("P91 → P16 of firms") as well as raw values, because the inputs are pre-normalised numbers that mean little on their own.
- **Empty, loading and error states on every page**, plus "API responded in X s", which doubles as the cold-start measurement on Vercel.
- **Phone width:** Streamlit stacks columns automatically; the CSS switches the card grid to two columns and tightens padding. Checked with real screenshots at 390 px.

**Bugs the screenshots caught** (that's why you look at the real thing, not just tests):
1. Chart labels were clipped on narrow screens.
2. A stress test pushed a distressed firm's ROA to the single most extreme value in the data (the 0th percentile). Percentile shocks are now kept within P1–P99, and a regression test covers it.

**Testing.** `tests/frontend/test_pages.py` uses Streamlit's `AppTest` to run each page headlessly. It clicks real buttons and checks session state, errors and the number of charts. The API client is replaced by an in-process fake that calls the *real* model, so the pages render real outputs without a server.

### Interview questions

**Q: Why Streamlit and not React?**
The whole project is Python, and I can defend every line. Streamlit's theme config and a small CSS file are enough for a clean analyst tool, and the UI logic is still testable with AppTest.

**Q: How do you test a Streamlit app?**
`streamlit.testing.v1.AppTest` runs a script without a browser. You can set session state, click buttons and inspect the elements it produced. External calls are mocked at the module boundary (`ui.api`).

**Q: Is the API URL visible to users?**
No. Streamlit code runs on the server, and only rendered elements reach the browser. The one leak path is an error message, and those are sanitised and tested.

---

## Phase 4: planning an Indian default backtest (plan only)

See [INDIA_BACKTEST_PLAN.md](INDIA_BACKTEST_PLAN.md). The key ideas:

- **Labels come from public events** (IBC admission, a D rating, a missed payment, an RBI supersession), dated by when they happened. Financial statements are dated by when they were **filed**, so the backtest never uses information an analyst didn't have (no look-ahead).
- **The biggest obstacle is the inputs, not the model:** the Taiwan ratios were normalised by the data provider in an undocumented way. The plan maps Indian ratios by *rank* (percentile) and retrains a reduced model on only the ratios India can supply, measuring what that costs on Taiwan first.
- **Lead time** is the headline result (how many years before default the model first flagged the firm), always reported next to the false-alarm rate on matched healthy peers.
- **Lenders (DHFL, Reliance Capital, IL&FS) are handled separately:** high leverage is normal for them, so a model trained on industrial firms would flag all of them.

### Interview questions

**Q: Can you apply a model trained in one country to another?**
Only after checking that the inputs mean the same thing. Here they don't (undocumented normalisation), so the plan uses rank mapping and reports the result as a test of transfer, not as a validated Indian PD.

**Q: What's look-ahead bias in default prediction?**
Using statements as of the fiscal year end when they were only published months later, sometimes after the default. The plan dates everything by filing date.

**Q: Why match healthy firms by sector and size?**
So the model isn't rewarded for simply spotting "small steel company", but matching also inflates AUC compared with the real population. The plan says so next to every number.
