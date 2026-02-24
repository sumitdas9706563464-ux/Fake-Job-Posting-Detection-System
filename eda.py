import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create a directory for EDA results
os.makedirs("eda_results", exist_ok=True)

# Load dataset
df = pd.read_csv("data/fake_job_postings.csv")

# 1. Basic Info
print("--- Dataset Info ---")
print(df.info())
print("\n--- Missing Values ---")
print(df.isnull().sum())

# 2. Class Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='fraudulent', data=df)
plt.title('Distribution of Real vs Fake Job Postings')
plt.xlabel('Is Fraudulent (0=Real, 1=Fake)')
plt.ylabel('Count')
plt.savefig('eda_results/class_distribution.png')
print("\nClass distribution plot saved to eda_results/class_distribution.png")

# 3. Text Length Analysis
df['description_len'] = df['description'].apply(lambda x: len(str(x).split()))
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='description_len', hue='fraudulent', kde=True, bins=30)
plt.title('Distribution of Description Length (Words)')
plt.savefig('eda_results/text_length_dist.png')
print("Text length distribution plot saved to eda_results/text_length_dist.png")

# 4. Correlating meta-features with fraud (categorical)
plt.figure(figsize=(10, 6))
sns.countplot(x='employment_type', hue='fraudulent', data=df)
plt.title('Fraud by Employment Type')
plt.xticks(rotation=45)
plt.savefig('eda_results/fraud_by_employment_type.png')

# 5. Correlation of binary flags
binary_cols = ['telecommuting', 'has_company_logo', 'has_questions', 'fraudulent']
plt.figure(figsize=(8, 6))
sns.heatmap(df[binary_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Binary Features')
plt.savefig('eda_results/correlation_heatmap.png')

print("\nEDA completed. Plots are in 'eda_results/' directory.")
