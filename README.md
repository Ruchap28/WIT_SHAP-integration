# 🔍 Enhancing the What-If Tool (WIT) with SHAP-Based Explainability

This project integrates **SHAP (SHapley Additive exPlanations)** into the **What-If Tool (WIT)** framework with firebase authentication enabling a more interpretable and transparent understanding of machine learning model predictions — especially for tabular data.

It allows users to:

- Upload custom datasets (with numerical and categorical features)
- Train an XGBoost model
- Visualize global and local feature contributions using SHAP plots
- Understand and explain individual predictions interactively
- Export insights for research or decision-making purposes

## 🚀 Features

- ✅ Upload a CSV dataset
- ✅ Automatic handling of numerical features
- ✅ Train XGBoost model internally
- ✅ SHAP plots:
  - Global Feature Importance
  - Beeswarm Plot
  - Waterfall Plot for specific predictions
- ✅ Interactive UI with **Streamlit**
- ✅ Detailed prediction breakdown (textual + visual)
- ✅ Graceful error handling and user feedback

---

## 🧪 Demo

| 📊 Global Importance -> Displays feature importance across the dataset |
| 🐝 Beeswarm Plot -> Visualizes SHAP values per instance and feature |
| 💧 Waterfall Plot -> Explains single predictions |
| 📃 Textual Explanation -> Breaks down feature-wise SHAP values |

📎 **Sample Output (PDF)**: See the `WIT_Output.pdf` included in this repository.

---
