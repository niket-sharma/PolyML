# PolyML

This repository contains example notebooks and datasets from
**Applications of Machine Learning to Optimizing Polyolefin Manufacturing**
(Chapter 10 of *Integrated Process Modeling, Advanced Control and Data Analytics
for Optimizing Polyolefin Manufacturing*, Wiley–VCH, 2023).

The original scripts distributed with the chapter were created for interactive
use and are preserved here for reference. Two cleaned up examples are provided
for quick experimentation:

- `random_forest_pipeline.py` &ndash; trains a `RandomForestRegressor` on
  `HDPE_LG_Plant_Data.csv` and reports RMSE.
- `gradient_boosting_pipeline.py` &ndash; demonstrates a gradient boosting model
  on the same dataset.

## Setup

Install the required packages and run one of the example pipelines:

```bash
pip install -r requirements.txt
python random_forest_pipeline.py
```

The data files `HDPE_LG_Plant_Data.csv` and `Polymer_Tg_SMILES.csv` are included
for convenience.

## Citation

Liu, Y. A., & Sharma, N. (2023). Applications of Machine Learning to Optimizing
Polyolefin Manufacturing. In *Integrated Process Modeling, Advanced Control and
Data Analytics for Optimizing Polyolefin Manufacturing* (Chapter 10). Wiley-VCH
GmbH. <https://doi.org/10.1002/9783527843831.ch10>
