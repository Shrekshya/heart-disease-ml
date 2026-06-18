# Heart Disease Prediction - Supervised Learning Project

A supervised machine learning project comparing three classification models 
for heart disease prediction using the UCI Heart Disease dataset.

## 📁 Project Structure
- `Task1.ipynb` — Full data analysis, feature engineering and model building
- `app.py` — Interactive Streamlit web application
- `heart.csv` — Heart disease dataset (1025 patients, 13 features)
- `model_comparison_results.csv` — Final model comparison results
- `requirements.txt` — Required libraries

## 📊 Dataset
- 1025 patients
- 13 features (age, cholesterol, blood pressure etc.)
- Target: 1 = Has Heart Disease, 0 = No Heart Disease
- Balanced dataset: 525 positive, 500 negative cases

## 🤖 Models Compared
| Model | Accuracy | Has Disease Recall | Best For |
|---|---|---|---|
| Logistic Regression | 80.52% | 87% | Simple Baseline |
| KNN | 85.71% | 87% | Overall Accuracy |
| Decision Tree | 84.42% | 96% | Medical Safety |

## 🏆 Conclusion
- **Best overall accuracy:** KNN (85.71%)
- **Best for medical use:** Decision Tree (96% recall for sick patients)
- **Most stable:** Logistic Regression

## 🚀 How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run the app: `python -m streamlit run app.py`

## 🛠️ Tools Used
- Python, Jupyter Notebook, VS Code
- pandas, numpy, scikit-learn, matplotlib, seaborn, streamlit
