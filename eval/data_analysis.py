import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('Heart Prediction Quantum Dataset.csv')  

# Data visualization
plt.figure(figsize=(12, 6))
df.hist(figsize=(10, 10), bins=20)
plt.suptitle("Feature Distributions")
plt.savefig("feature_distributions.png", dpi= 300)
plt.show()


# Scatter plot matrix
sns.pairplot(df, hue="HeartDisease")
plt.savefig("scatter_matrix.png", dpi= 300)
plt.show()


# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlations")
plt.savefig("correlation_heatmap.png", dpi= 300)
plt.show()
