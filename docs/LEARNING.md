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
