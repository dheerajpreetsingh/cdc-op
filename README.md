# House Price Prediction with Structured + Satellite Image Features

This project builds a house-price prediction system using a combination of structured property data, satellite imagery downloaded via the Mapbox API, and machine-learning models enhanced with PCA and interpretability tools. The workflow is split into modular components so that each stage remains clear, reproducible, and easy to understand.

---

## Project Structure

.
├── data_fetcher.py
├── preprocessing.ipynb
├── model_training.ipynb
├── data.zip

### Contents of `data.zip`

Extracting the archive produces:
data/
├── bad_test_image_ids.csv
├── bad_train_image_ids.csv
├── test_image_features.csv
├── train_image_features.csv
├── test.csv
├── train.csv

These files are required for the notebooks and scripts to run. Extract `data.zip` into the project root before running anything.

---

## 1. Satellite Image Fetching

`data_fetcher.py` downloads satellite imagery for each property using latitude and longitude coordinates and saves them locally. Images are fetched from the Mapbox Static API using an access token provided inside the script.

### Key features

- Skips images that have already been successfully downloaded  
- Re-downloads only images marked as “bad”  
- Handles missing coordinate records gracefully  
- Organizes image outputs into individual folders

Images are stored in:
- `train_img/`
- `test_img/`

### Running the script

The script expects `train.csv` and `test.csv` to be present. If corrupted images are found later, their IDs can be added to:
- `bad_train_image_ids.csv`
- `bad_test_image_ids.csv`

and the script will overwrite them on the next run.

---

## 2. Exploratory Data Analysis and Preprocessing

Notebook: `preprocessing.ipynb`

This notebook focuses on understanding the dataset rather than training models. It primarily contains visual exploration and discussion.

### Topics covered

- Initial dataset inspection
- Missing data exploration
- Distribution plots for key numerical features
- Correlation matrices
- Pair plots examining feature relationships
- Geographic maps and spatial insights
- Outlier identification and justification for retaining them

**Note on Outliers:** Outliers are intentionally not removed because many represent real luxury homes and carry meaningful information about the market rather than being noise.

---

## 3. Modeling, Feature Extraction and PCA

Notebook: `model_training.ipynb`

This notebook implements the end-to-end modeling workflow.

### Steps performed

1. Merge structured tabular data with extracted image features  
2. Standardize numerical features where appropriate  
3. Apply Principal Component Analysis (PCA) to image feature matrices  
4. Train XGBoost regression models  
5. Tune hyperparameters using RandomizedSearchCV  
6. Evaluate performance on validation data  
7. Use interpretability methods including feature importance plots, SHAP visualizations, and Grad-CAM overlays

### Why PCA was used

Image-derived feature sets tend to be extremely high dimensional and often contain redundant information. PCA is applied to:

- Reduce dimensionality  
- Minimize multicollinearity  
- Improve model generalization  
- Lower computational cost  
- Reduce risk of overfitting  
- Preserve most of the signal while removing noise

Only components explaining most of the variance are retained during training.

---

## Requirements

Install the required Python libraries before running the notebooks:

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn requests
```

Note: Additional image or deep-learning libraries may be required depending on how Grad-CAM is configured.

---

## How to Use

1. **Extract the dataset archive:**
   `unzip data.zip`

2. **Download satellite images for train and test rows:**
   `python data_fetcher.py`

3. **Open and run the exploratory notebook:**
   `preprocessing.ipynb`

4. **Train and analyze models in:**
   `model_training.ipynb`

---

## Duplicate IDs

Some IDs appear multiple times with different dates. The same image is used for all occurrences belonging to that ID. This simplifies processing while keeping meaningful location information intact.

---

## Modeling Notes

- Outliers remain in the dataset intentionally.
- Geographic information has a large influence on price.
- PCA is applied to stabilize and compress image features.
- RandomizedSearchCV helps reduce hyperparameter tuning time.
- SHAP and Grad-CAM support transparency and explainability.

---

## Troubleshooting

**Force plots not rendering in SHAP**
```python
import shap
shap.initjs()
```

Trust the notebook if prompted.

**Missing or failed images**

Check the bad image ID files and rerun:
`python data_fetcher.py`


**Memory usage issues**

- Reduce PCA components  
- Limit training sample size temporarily  
- Adjust model depth and subsampling parameters  

---

## Reproducibility

Random seeds are set where possible, but slight variation may occur because randomized search and XGBoost introduce nondeterministic behavior.

---

## Future Improvements

- Integration with experiment tracking tools  
- Additional geospatial feature engineering  
- Benchmarking against neural network approaches  
- API or web-based deployment pipeline  
- Automated image feature extraction workflows

---

## Summary

This project demonstrates how structured features, geospatial imagery, dimensionality reduction, and modern machine-learning models can be combined to predict property values efficiently and transparently. Each file is structured to separate exploration, preprocessing, and modeling so that results remain easy to follow and reproduce.