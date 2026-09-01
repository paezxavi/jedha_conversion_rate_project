# Who subscribes to the newsletter — and what can be done about it?

A conversion model built for **www.datascienceweekly.org**, who wanted to read its parameters and
find a lever on a **3.2%** conversion rate.

Jedha *Full Stack Data Scientist* — **Block 3, Machine Learning Project** (Conversion rate
challenge). scikit-learn.

The full analysis, with every decision and its measured cost, is in
[`conversion_rate_project.ipynb`](conversion_rate_project.ipynb).

## The problem

The newsletter's data scientists opened a competition: predict, from five columns about a visit,
whether the visitor subscribes. The score is the **f1-score of the positive class** — 3.2% of
visits convert, so accuracy is meaningless (predicting "nobody subscribes" is right 96.8% of the
time). Part of the brief is the model itself; the other part is reading its parameters for
something the team can act on.

The model was built as asked. It reaches **f1 = 0.77 on unseen visitors**, and the answer to the
question behind the brief is uncomfortable:

> The score is almost entirely carried by **how many pages the visitor read** — a number that only
> exists once the visit is over. Remove it and the f1 falls from **0.77 to 0.21**.
>
> The one column that can be acted on before the visit is the **country**, and it holds the whole
> opportunity: **China is 24% of the traffic and 1% of the sign-ups.**

## The dataset

`conversion_data_train.csv` — 284 580 labelled visits, `conversion_data_test.csv` — 31 620
unlabelled ones to predict. Five columns: `country`, `age`, `new_user`, `source`,
`total_pages_visited`.

Nothing is missing and nothing is mistyped. Two things look like they need a cleaning rule, and
both rules would be wrong:

- **268 769 rows are exact duplicates — and none of them is a duplicate.** Five columns describe
  only 13 942 distinct profiles, and 1 869 of those profiles appear with *both* outcomes: same
  country, same age, same page count, one subscribed and one did not. `drop_duplicates()` deletes
  94% of the file and turns a 3.2% conversion rate into 25.7%. Priced in the notebook: **−8.6
  points of f1**.
- **Two visitors are aged 111 and 123.** Data-entry errors, 0.003% of the file, and the file to
  predict contains nobody above 69. Removing them moves the f1 by **+0.0003**, so they stay and the
  notebook says why instead of silently trimming a distribution.

## What we found

### The page counter is the model

![Conversion rate against pages read](images/2_pages_curve.png)

Not a segment — a switch. Under 5 pages the conversion rate is below 0.2%; at 15 pages it is 74%;
from 21 pages on, **every visitor in the file subscribed**. The correlation with the target is
**0.53**, five times anything else.

Rebuilding the model on one group of columns at a time settles what it is using:

| Columns given to the model | f1 |
|---|---|
| every column | **0.774** |
| `total_pages_visited` alone | 0.714 |
| pages + country | 0.729 |
| pages + country + new_user | 0.766 |
| every column **except** `source` | 0.772 |
| every column **except** `total_pages_visited` | **0.208** |
| `new_user` alone | 0.131 |
| `age` alone | 0.103 |
| `country` alone | 0.089 |
| `source` alone | 0.065 |

One column carries 92% of the score, and everything the marketing team can act on before a visit,
taken together, scores 0.21.

### China is 24% of the traffic and 1% of the sign-ups

![Conversion rate at equal reading depth](images/3_china_gap.png)

Chinese visitors browse like everyone else — **4.55 pages on average against 4.98** — and then do
not subscribe. At equal reading depth the gap holds: at 13-15 pages a visitor converts at **61%**
from the US, the UK or Germany and at **6%** from China; at 6-12 pages the gap is a factor of 30.

A gap that survives at equal reading depth is not a gap in interest. Something at the **last step**
— the sign-up form — behaves differently for these visitors.

### The other segments, and the one that does nothing

![Conversion rate by segment](images/1_conversion_by_segment.png)

Returning visitors convert **5 times better** than new ones (7.2% against 1.4%), age falls
monotonically from 5.5% (17-24) to 0.55% (50+), and `source` — the column that maps onto a budget
line — is flat: Ads 3.5%, Seo 3.3%, Direct 2.8%, worth **0.002 of f1**.

## The model

A logistic regression on the five columns: standardised numerics, one-hot categoricals, stratified
80/20 split. **f1 0.762 on train, 0.768 on test** at the default threshold — eight features on
227 664 rows leaves nothing to memorise.

Two things then improved it, and one did not:

- **The decision threshold** — 0.5 minimises the error count, which is not what f1 measures.
  Cross-validated on the training set alone it lands on **0.399**, worth **+0.006 of f1** (0.7678 →
  0.7743) and stable at ±0.005 over ten splits. Precision 0.81, recall 0.74.
- **Nothing else beat the straight line.** Gradient boosting (0.771), a pruned random forest
  (0.767) and a tuned penalty (0.774) all land inside the ±0.006 that separates two random splits.
  The unpruned forest is the only clear result — 0.81 on train against 0.74 on test.
- **The ceiling is 0.806.** An oracle that knows the conversion rate of all 13 942 profiles, on the
  file itself, scores 0.806: **3 358 visits are described by a profile that appears with both
  outcomes** and no model reading these five columns can separate them. Stopping at 0.774 is a
  measurement, not a shrug.

![Model coefficients](images/5_coefficients.png)

Read as odds ratios: being outside China multiplies the odds of subscribing by **21 to 37**, every
3.3 extra pages by **12.5**, being a new visitor divides them by **5.5**, and eight years of age by
**1.8**. `Seo` against `Ads` is ×0.97 — nothing.

The same model, read as the rule it applies: **read about twelve pages and you are a likely
subscriber — unless you are in China, where it takes sixteen.**

## What the newsletter team should do

1. **Use it as a session scorer, not a campaign planner.** `total_pages_visited` only exists once
   the visit is over, so the model cannot rank an audience beforehand. Live, it can: at the twelfth
   page it is right four times out of five, which is a sign-up prompt triggered on a score rather
   than on every page.
2. **Audit the Chinese sign-up path before buying any traffic.** 24% of visits, 0.13% conversion
   against 3.8% for the US, and the gap holds at equal reading depth — a form that does not submit,
   an e-mail that does not deliver, a missing translation. **At the US rate those same visits would
   produce 2 620 sign-ups instead of 89: +28% total conversions, from traffic already paid for.**
3. **Do not reallocate the acquisition budget on `source`.** The three channels are within half a
   point of each other and the column is worth 0.002 of f1. A campaign-level breakdown is the thing
   to log next.
4. **Log three more columns**: which article was read (not just how many), the device, and the time
   on page. The ceiling says these five columns are exhausted; the next points have to come from
   new ones, and two of those three are known *before* the sign-up decision.

## The submission

`conversion_data_test_predictions_XAVIER-PAEZ-logreg.csv` — 31 620 rows, one column named
`converted`, no index. The model is refit on all 284 580 labelled visits and applied with the
threshold of 0.399. It calls **2.9%** of the file subscribers against a 3.2% base rate.

Expected leaderboard score: **0.768 ± 0.006**, measured by re-running the whole procedure over ten
splits — not the 0.774 of a single one.

## Reproducing

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Then open `conversion_rate_project.ipynb` and select the `.venv` kernel. The notebook reads only
the two CSV files — no API call, no credential and no network access anywhere in this project.

Charts render as static images so the notebook stays readable on GitHub; the same PNGs are written
to `images/`. If `kaleido` cannot find a browser, run `.venv/bin/plotly_get_chrome`.

Everything is seeded on `RANDOM_STATE = 42`, so a re-run reproduces every number above. The whole
notebook executes in about a minute.

## Layout

```
conversion_rate_project.ipynb    the analysis — sections 1 to 8
conversion_data_train.csv        284 580 labelled visits
conversion_data_test.csv         31 620 visits to predict
conversion_data_test_predictions_XAVIER-PAEZ-logreg.csv   the submission
images/                          the five charts, also embedded in the notebook
requirements.txt                 pinned versions
```
