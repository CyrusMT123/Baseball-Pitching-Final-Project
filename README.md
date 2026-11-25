# Fast and Furious: Analyzing MLB Pitch Movement and Performance

**Lincoln Hamid, Cyrus Kiely, Dashua Ough** (QSS Final Project, Dartmouth College)

This project studies pitch movement in Major League Baseball to better visualize and understand why certain pitches are more effective than others. With Hawk-Eye tracking cameras now installed in every MLB stadium, pitch traits are measured on every pitch with a high level of precision. The data and the nuance of pitch dynamics raise the question this project focuses on: **which measurable traits actually make a pitch effective, and how much do they matter?**

Using Baseball Savant data from 2024 to 2026, merged at the pitcher-season-pitch-type level, we:

- measure how **unique** each pitch is with k-nearest neighbors,
- estimate which pitch traits are associated with **whiff rate** for each pitch type with weighted least squares regressions,
- find the most and least **similar pitchers** by arsenal and velocity, and
- train a supplementary **neural network that classifies pitch type** from a pitch's characteristics.

---

## Repository Structure

```
Baseball-Pitching-Final-Project/
├── data/                       # raw Baseball Savant CSV exports (2024, 2025, 2026)
│   ├── pitch_arsenals{24,25,26}.csv
│   ├── pitch_movement{24,25,26}.csv
│   ├── pitch-arsenal-stats1{24,25,26}.csv
│   ├── pitch-arsenal-stats300{24,25,26}.csv
│   └── pitcher_arm_angles{24,25,26}.csv
├── derived/
│   ├── data24-26.csv           # combined, cleaned dataset (one row per pitcher-season-pitch type)
│   ├── pitch_type_summary.csv  # means of key metrics by pitch type (Table 1 in the report)
│   └── coef_summary.csv        # whiff-rate regression coefficients by pitch type (Table 2)
├── visualizations/
│   ├── <Pitcher> <Year>.png    # short-form movement plots
│   └── coefficients/           # coefficient plots per pitch type + whiff vs. xwOBA
├── CombiningData.ipynb
├── Visualizations.ipynb
├── pitcher_similarity_plus_pitch_prediction.ipynb
├── pitcher_uniqueness.ipynb
├── Regressions.ipynb
└── README.md
```

| Folder | Purpose |
|---|---|
| `data/` | Raw CSVs downloaded from Baseball Savant, kept unchanged so the merge can be reproduced |
| `derived/` | The cleaned, merged dataset and the summary tables used in the report |
| `visualizations/` | Saved figures: movement profiles and regression coefficient charts |

---

## Data

All data comes from Baseball Savant, which gets data on pitch movement from the Hawk-Eye tracking cameras installed in every MLB stadium. We downloaded the following leaderboards as CSVs for the 2024, 2025, and 2026 seasons:

| Leaderboard | Contents |
|---|---|
| [Pitch arsenal](https://baseballsavant.mlb.com/leaderboard/pitch-arsenals) | The rate at which each pitcher throws each pitch type, and each pitch type's average velocity |
| [Pitch movement](https://baseballsavant.mlb.com/leaderboard/pitch-movement) | The horizontal and vertical movement induced by spin and seam-shifted wake effects |
| [Pitch arsenal stats](https://baseballsavant.mlb.com/leaderboard/pitch-arsenal-stats) | Batter performance against each pitcher and pitch type (whiff %, K %, xwOBA, hard-hit %, run value, …) |
| [Release traits](https://baseballsavant.mlb.com/leaderboard/pitcher-arm-angles) | The angle of the pitcher's arm relative to the body, and the x, y, and z position of the ball at release |

The data was merged on `player_id`, `pitch_type`, and `year`. The release-trait data, which is at the pitcher-season level, was merged in afterward on `player_id` and `year`. Each row of the final dataset is one pitcher's pitch type in one season.

- **Qualified pitch types:** A pitcher's pitch type that was thrown fewer than 300 times or that ended fewer than 50 plate appearances is tagged as not qualified. "Qualified" is the baseball term for meeting the minimum participation requirement for a leaderboard or award.
- **Horizontal break:** We adjusted horizontal break so that, for both left- and right-handed pitchers, movement toward the pitcher's arm side is positive and movement toward the glove side is negative.
- **Coverage:** Some pitchers, and some pitchers' pitch types, appear only in certain years because players don't appear in every season. This isn't a missing-data problem, but it can make it harder to compare some players from year to year.

The final dataset (`derived/data24-26.csv`) has **3,677 rows and 53 columns**, covering **489 pitchers**.

---

## Methodology

### 1. Build the Combined Dataset
**Notebook:** `CombiningData.ipynb`

- Merges the five Savant exports for each season, stacks the 2024–2026 seasons, and adds the `qualified` flag
- Checks that there are no duplicate pitcher-year-pitch rows and that the row counts match

**Output:** `derived/data24-26.csv`

### 2. Movement Profiles
**Notebook:** `Visualizations.ipynb`

Creates a short-form movement profile for each pitcher-season, plotting each pitch's horizontal break against its vertical break, with each circle sized in proportion to usage rate. Also compares arm angle with release height and release side.

**Outputs:** `visualizations/<Pitcher> <Year>.png`

### 3. Pitch Uniqueness (k-Nearest Neighbors)
**Notebook:** `pitcher_uniqueness.ipynb`

1. Standardize four traits so that each one carries the same weight: horizontal movement, vertical movement, arm angle at release, and velocity. Without this step, velocity would greatly outweigh the others.
2. Compute the Euclidean distance between pitches' standardized vectors, measured in standard deviations.
3. For each pitch, find the five closest pitches **thrown by other pitchers**. Excluding the pitcher's own pitches keeps a pitch from simply matching itself in another season. A pitch's uniqueness score is its average distance to those five neighbors. We average five neighbors rather than using only the nearest one, so that a single unlikely look-alike doesn't make an unusual pitch look ordinary.
4. Repeat the calculation within each pitch type, to see how differently the same pitch type can move depending on the pitcher.

### 4. Pitch Traits and Performance (Regression)
**Notebook:** `Regressions.ipynb`

We measure effectiveness with **whiff rate**: swings and misses divided by total swings. We chose whiff rate because it was less noisy than other performance metrics. Across the 2,044 qualified pitch-type seasons, it was strongly correlated with xwOBA allowed (r = −0.72).

For each of eight pitch types (FF, SI, FC, FS, CH, SL, ST, CU), we estimate:

```
Whiff%_i = β0 + β1·z_Velo + β2·z_ArmAngle + β3·z_RelHeight + β4·z_ArmsideMvmt + β5·z_VerticalMvmt + u_i
```

- Every feature is standardized, so each coefficient is the change in whiff rate for a 1 SD change in that feature, with the other features held constant.
- The model is weighted least squares, with each pitch weighted by its plate appearances. This gives more weight to pitchers who threw more of that pitch type.
- We excluded candidate features that were highly collinear with these five.
- Significance is tested with 95% confidence intervals and t-tests at the 5% level.

**Outputs:** `derived/coef_summary.csv`, `derived/pitch_type_summary.csv`, `visualizations/coefficients/*.png`

### 5. Pitcher Similarity
**Notebook:** `pitcher_similarity_plus_pitch_prediction.ipynb`

Each pitcher gets a vector made of two parts:

- **Usage:** the raw percentage of the time they throw each pitch type, with 0% for pitch types they don't throw.
- **Velocity:** for each pitch type, how many standard deviations the pitcher's average velocity is from the league average for that pitch. Velocity is standardized this way so it doesn't dwarf pitch usage.

We then compare every pair of pitchers using Euclidean distance. The comparison is run three times: for all pitchers, for pitchers with at least 3 pitches thrown at least 5% of the time, and for pitchers with at least 4 such pitches. In the restricted runs, the velocity averages used for standardization include only the pitchers in that pool, not the full league.

### 6. Pitch-Type Prediction Model (Supplementary)
**Notebook:** `pitcher_similarity_plus_pitch_prediction.ipynb`

This is a multilayer back-propagation perceptron trained to identify the type of pitch thrown (11 classes).

- **Inputs:** usage, velocity, horizontal and vertical break, run value, whiff %, K %, put-away %, xBA, xSLG, xwOBA, hard-hit %
- **Model 1:** Dense 20 → 5 → 11 (softmax), trained for 100 epochs
- **Model 2:** Dense 20 → 12 → 29 → 11 (softmax), trained for 1,000 epochs

---

## Research Questions

1. How unique are individual pitchers' pitches, and what makes a pitch stand apart from the rest of the league?
2. For each pitch type, which traits (velocity, vertical and horizontal movement, release height, and arm angle) are associated with swings and misses?
3. Across pitchers, which pitchers are the most and least similar in their arsenals and pitch velocities?

---

## Key Findings

- **Uniqueness:** The league's most unusual pitches belong almost entirely to a few low-arm-angle relievers, led by Tyler Rogers. Rogers, Tim Hill, and Hoby Milner account for 8 of the 12 most unique pitches in MLB. Rogers' arm angle is 44° lower than the second-lowest, which belongs to Hill.
- **Velocity is king, and not only for fastballs:** Velocity has a positive coefficient on whiff rate for seven of the eight pitch types, and the coefficient is significant for five of them. Curveballs, sliders, and sweepers are among those five. The smallest velocity coefficients are for changeups and splitters, which are thrown slower on purpose to disrupt hitters' timing.
- **Four-seam fastballs:** All five traits are significant. A lower arm angle (−2.44 percentage points of whiff rate per 1 SD) and higher velocity (+2.43) have the largest effects, followed by more vertical break (+1.81), a lower release height (−0.90), and less arm-side movement (−0.59).
- **Less of the "typical" movement:** Sinkers, splitters, and four-seamers that move less toward the arm side, and sweepers that move less toward the glove side, are associated with more whiffs.
- **Modest explanatory power:** These traits explain only a modest share of the variation in whiff rate (R² from 0.09 to 0.34), so no single combination of traits makes a pitch effective.
- **Most similar arsenals (pitchers with 4+ pitches):** Jorge Alcala & Grant Taylor, Erick Fedde & Justin Topa, Jacob deGrom & Jared Jones, Cole Ragans & Grayson Rodriguez, and Justin Verlander & Landon Knack.
- **Pitch-type classifier:** Training accuracy was about 82% for Model 1 and 89% for Model 2. Random guessing would score about 9%. Most errors come from pitches whose movement overlaps, such as Dylan Cease's 2026 fastball, sinker, and changeup, which are close on a movement plot.

---

## Running the Project

Run the notebooks from the repository root in this order:

| Order | Notebook | Description |
|---|---|---|
| 1 | `CombiningData.ipynb` | Merges the raw Savant CSVs into `derived/data24-26.csv` |
| 2 | `Visualizations.ipynb` | Movement profiles and release-trait plots |
| 3 | `pitcher_uniqueness.ipynb` | k-NN pitch uniqueness |
| 4 | `Regressions.ipynb` | Whiff-rate regressions and coefficient plots |
| 5 | `pitcher_similarity_plus_pitch_prediction.ipynb` | Pitcher similarity and the pitch-type neural network |

---

## Limitations

- The study treats pitches as if they happened in a vacuum, without the context in which they were thrown. Pitch location and sequencing are key to performance but couldn't be evaluated with this dataset, which helps explain the low R² values.
- MLB data has survivorship bias: pitchers who throw low-quality pitches are sent down to the minor leagues and don't appear consistently in the data. We are only seeing how these traits work at the highest level.
- Whiff rate gave the highest R², but it doesn't account for pitches meant to induce weak contact rather than whiffs, like sinkers.
- The data has season averages for each pitcher and pitch type, not individual pitches.

## Future Work

- Evaluate pitches with a metric other than whiff rate while keeping a comparable level of correlation, and add features such as extension (how close to the plate the pitcher releases the ball).
- Examine how repeated exposure to the same arm angle affects hitters, and whether higher velocity on one pitch raises the whiff rate of another.
- Train the predictive model on individual pitchers' velocity and arsenal data, then give it single-game data to build a tool that identifies who pitched a game.
