# 🌸 Multiclass Classification using SVM 🌸

## Overview
This R script demonstrates how to perform multiclass classification using Support Vector Machines (SVM) on the Iris dataset. The Iris dataset is a well-known dataset that consists of 150 samples from three different species of iris flowers (`setosa`, `versicolor`, and `virginica`). The goal is to classify the species based on the four features: Sepal Length, Sepal Width, Petal Length, and Petal Width.

## Steps

### 1. **Install R Kernel**
   - Install necessary dependencies and the IRkernel to enable R integration in Jupyter notebooks.
   - Install R packages such as `e1071` (for SVM), `caret` (for data processing), and other visualization libraries like `ggplot2`, `pROC`, and `corrplot`.

### 2. **Load and Explore Data**
   - Load the Iris dataset and display the first few rows.
   - Summary statistics are shown to understand the distribution and central tendencies of the features.

### 3. **Exploratory Data Analysis (EDA)**
   - **Class Distribution**: Visualize the distribution of species in the dataset to check for any imbalance.
   - **Outliers**: Identify outliers using the IQR method and replace them with the median value.
   - **Visualizations**:
     - **Boxplots**: To visualize the distribution of `Sepal.Length` and `Sepal.Width`.
     - **Pairwise Plot**: A scatter plot matrix to observe the relationships between features.
     - **Correlation Heatmap**: Display the correlation between the features using a color-coded heatmap.

### 4. **Data Preprocessing**
   - **Missing Values**: Check for missing values in the dataset (none in this case).
   - **Normalization**: Normalize the numeric features to scale them between 0 and 1 for better SVM performance.
   - **Train-Test Split**: Split the dataset into training (70%) and testing (30%) sets.

### 5. **Model Building: SVM**
   - **Training the SVM Model**: Build an SVM model with a radial kernel, set `cost = 1` and `gamma = 0.5`, and enable probability estimation.
   - **Model Summary**: Check the number of support vectors and the levels of the species.

### 6. **Fine-Tuning the Model**
   - **Hyperparameter Tuning**: Perform cross-validation to find the best `cost` and `gamma` values.
   - **Best Model**: Extract and display the best model after tuning.

### 7. **Predictions**
   - Make predictions on the test data and extract the predicted probabilities for each species.

### 8. **Feature Importance (Optional)**
   - Visualize feature importance based on the SVM coefficients to see how each feature contributes to the classification.

### 9. **Confusion Matrix & Evaluation**
   - **Confusion Matrix**: Evaluate the model's performance by checking the confusion matrix, accuracy, and other metrics like sensitivity, specificity, and kappa.
   - **Confusion Matrix Heatmap**: Plot the confusion matrix using a heatmap for a better visual understanding.

## Example Output

### Model Accuracy:
- **Accuracy**: 93.33%
- **Sensitivity**:
  - Setosa: 100%
  - Versicolor: 93.33%
  - Virginica: 86.67%
- **Specificity**:
  - Setosa: 100%
  - Versicolor: 93.33%
  - Virginica: 96.67%

### Confusion Matrix Heatmap:
A heatmap visually representing the confusion matrix, with values showing the true and predicted classifications for each species.

---

## Conclusion
This workflow demonstrates how to classify flower species using an SVM model in R, including steps for data preprocessing, model building, tuning, and evaluation. With an accuracy of over 93%, the model performs well on the Iris dataset.

🌸 Happy Data Science! 🌸
