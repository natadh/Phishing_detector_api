import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, f1_score

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Unsupervised
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

# === 1. Load the Dataset ===
df = pd.read_csv("data/phishing_email.csv")
print("✅ Dataset loaded successfully!")
print("Columns in dataset:", df.columns)
print(df.head())

# === 2. Preprocess the Email Text ===
if 'body' in df.columns:
    text_column = 'body'
elif 'email_content' in df.columns:
    text_column = 'email_content'
elif 'text_combined' in df.columns:
    text_column = 'text_combined'
else:
    raise ValueError("No column for email text found (expected 'body', 'email_content', or 'text_combined').")

# Apply text cleaning
df['cleaned_text'] = df[text_column].apply(clean_text)

print("\nClass Distribution:")
print(df['label'].value_counts())

# === 3. Convert Text to TF-IDF Features ===
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['label']  # 1 = Phishing, 0 = Legitimate

# === 4. Split the Data ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 5. Compare Multiple Models (Naïve Bayes, Random Forest, SVM, Logistic Regression) ===
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel="linear", probability=True, random_state=42)
}

best_model = None
best_f1 = 0
results = {}

print("\n=== Comparing Multiple Models ===")
for name, model_instance in models.items():
    model_instance.fit(X_train, y_train)
    y_pred = model_instance.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results[name] = {"accuracy": acc, "f1_score": f1}

    print(f"\n{name} Performance:")
    print(f"Accuracy: {acc:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(classification_report(y_test, y_pred))

    # Select best model by F1-score (you can change to accuracy if you prefer)
    if f1 > best_f1:
        best_f1 = f1
        best_model = model_instance

print("\n=== Summary of Model Performance ===")
for model_name, metrics in results.items():
    print(f"{model_name}: Accuracy={metrics['accuracy']:.2f}, F1 Score={metrics['f1_score']:.2f}")

print(f"\nBest Model Based on F1 Score: {best_model}")

# === 6. Hyperparameter Tuning for Logistic Regression ===
# (Even if LR wasn't the best, you can still see if tuning helps.)
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'saga']
}
grid_search = GridSearchCV(
    LogisticRegression(max_iter=1000),
    param_grid,
    cv=5,
    scoring='accuracy'
)
grid_search.fit(X_train, y_train)

print("\nBest parameters for Logistic Regression:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# Use the best LR model from GridSearch
tuned_lr_model = grid_search.best_estimator_
y_pred_tuned = tuned_lr_model.predict(X_test)

print("\n✅ Tuned Logistic Regression Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_tuned):.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred_tuned))

# === 7. Unsupervised Learning (K-Means) ===
num_clusters = 2
kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42, batch_size=100)
clusters = kmeans.fit_predict(X)
df['Cluster'] = clusters

print("\n🔹 K-Means Clustering Results (Cross Tabulation):")
print(pd.crosstab(df['label'], df['Cluster']))

# Optional: Visualize Clusters (commented out to avoid popping plots in a script)
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X.toarray())
# plt.figure(figsize=(8,6))
# sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette="coolwarm", alpha=0.6)
# plt.title("K-Means Clustering on Email Dataset (PCA Reduced)")
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")
# plt.show()

# === 8. Save the Final Model and Vectorizer ===
# Here, we choose to save the tuned Logistic Regression model
joblib.dump(tuned_lr_model, "models/phishing_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
print("\n✅ Tuned Model (Logistic Regression) and vectorizer saved successfully!")
