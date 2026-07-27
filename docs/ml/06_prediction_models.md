Prediction Models

Overview

Prediction models consume:

Approved feature sets

Embeddings

User-independent football context

They never consume raw data directly.

Models

1. Expected Minutes

Target:Minutes played next fixture.

Inputs:

Injury status

Starts

Rotation

Congestion

Manager tendencies

Team context

Player embedding

Output:

Expected minutes

Start probability

2. Expected FPL Points

Target:Next GW points and 3/6 GW aggregates.

Model candidates:

XGBoost

LightGBM

CatBoost

Features:

Selected Feature Scientist features

Embeddings

Expected minutes

Outputs:

Mean prediction

Confidence

Feature contributions

3. Price Movement

Targets:

Rise

Fall

No change

Features:

Transfer momentum

Official predictor

Ownership

Market velocity

Availability

Form

Outputs:

Probability rise

Probability fall

4. Fair Value

Estimate intrinsic player value independent of current FPL price.

Outputs:

Fair price

Overvalued score

Undervalued score

Training

Walk-forward only.

Never randomly split data.

Champion/challenger workflow:

Current Model↓Candidate↓Evaluation↓Promote if better

Metrics

Minutes:

MAE

Start accuracy

Points:

MAE

Rank correlation

Top-k precision

Price:

Accuracy

Log loss

Brier score

Calibration

Recommendations are the final business metric.
