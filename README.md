# Teen Mental Health Analysis

Exploratory analysis and classification model for depression in teens, based on social media use and lifestyle factors. The project covers the full data science workflow: exploring a dataset, preparing it for modeling, training multiple classifiers, handling class imbalance, and interpreting model coefficients against independent EDA evidence.

## Key Findings

In a dataset of 1,200 teens (31 labeled as depressed):

- **Sleep, social media use, stress, and anxiety** are the four variables that meaningfully differ between depressed and non-depressed groups.
- Median sleep among depressed teens: **4.6 hours**, vs **6.5 hours** for non-depressed.
- Median daily social media use among depressed teens: **7.0 hours**, vs **4.4 hours** for non-depressed.
- A logistic regression classifier with class weighting and a tuned decision threshold (0.4) caught **all 6 depressed teens** in the held-out test set, with 6 false positives among 234 non-depressed teens. (100% recall, 50% precision.)
- A Random Forest classifier underperformed despite being more flexible, illustrating that model complexity should match data size.
- The model assigned large coefficients to several features (gender, platform, academic performance) where the underlying group differences were absent or minimal — flagged as overfitting artifacts rather than findings.

## Project Structure

```
teen-mental-health-analysis/
├── teen-mental-health-analysis.ipynb   # main analysis notebook
├── data/
│   ├── Teen_Mental_Health_Dataset.csv  # source dataset
│   ├── LICENSE                         # Apache 2.0 license for the dataset
│   └── NOTICE                          # attribution to original source
└── README.md
```

## Reproducing the Analysis

### Requirements

- Python 3.10+
- pandas, numpy, matplotlib, seaborn, scikit-learn

### Steps

1. Clone the repository:

```
git clone https://github.com/TanmoyDas6697/teen-mental-health-analysis.git
cd teen-mental-health-analysis
```

2. Install dependencies:

```
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

3. Launch Jupyter and open the notebook:

```
jupyter notebook teen-mental-health-analysis.ipynb
```

4. Run the cells from top to bottom.

The notebook expects the dataset at `data/Teen_Mental_Health_Dataset.csv` (relative path from the repo root).

## Dataset

The dataset was obtained from Kaggle: [Teenager Mental Health Dataset by Algozee](https://www.kaggle.com/datasets/algozee/teenager-menthal-healy).

Licensed under the Apache License, Version 2.0. The full license text and attribution notice are in the `data/` folder.

## Limitations

- Only 31 depressed cases in the dataset (25 in training, 6 in test). Minority-class performance metrics are noisy and not generalizable.
- The dataset appears synthetic or heavily preprocessed. Patterns may be cleaner than they would be in real-world data.
- No regularization or cross-validation was applied. Coefficients on weak features (gender, platform usage) are likely overfit and are not reported as findings.
- This is a correlational analysis. No causal claims can be made about whether social media use, sleep, or stress *cause* depression versus being associated with it.

## License

The dataset is licensed under Apache 2.0 (see `data/LICENSE`). The notebook and analysis code are released without a specific license; default copyright applies.
