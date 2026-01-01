import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

import os
import matplotlib
matplotlib.use('Agg')

df = None
if os.path.exists("penguins.csv"):
    df = pd.read_csv("penguins.csv")
else:
    try:
        df = sns.load_dataset("penguins")
        print('Loaded penguins dataset from seaborn')
    except Exception as e:
        raise SystemExit('Could not load penguins dataset: ' + str(e))
print(df.head())
print(df.info())
print(df.isna().sum())
df = df.dropna()

# EDA
sns.countplot(data=df, x='species')
plt.title("Distribution of Penguin Species")
plt.show()

sns.scatterplot(data=df, x='bill_length_mm', y='bill_depth_mm', hue='species')
plt.title("Bill Length vs Bill Depth")
plt.show()

sns.boxplot(data=df, x='species', y='flipper_length_mm')
plt.title("Flipper Length by Species")
plt.show()

sns.boxplot(data=df, x='sex', y='body_mass_g')
plt.title("Body Mass by Sex")
plt.show()

# Correlation heatmap
numeric = df[['bill_length_mm','bill_depth_mm','flipper_length_mm','body_mass_g']]
sns.heatmap(numeric.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Scikit
X = numeric
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=model.classes_).plot()
plt.show()
