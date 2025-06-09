import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from collections import Counter
from imblearn.over_sampling import SMOTE
from scipy.stats import randint
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('dataset/cybersecurity_intrusion_data.csv')  # Update to the new dataset path
print("📄 Dataset loaded.")

# Drop the 'session_id' column as it is not a feature
if 'session_id' in df.columns:
    df = df.drop('session_id', axis=1)
    print("🗑️ Dropped 'session_id' column.")

# Encode categorical features
categorical_cols = ['protocol_type', 'browser_type', 'encryption_used']
label_encoders = {}

for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        print(f"✅ Encoded '{col}'.")

# Encode target column
if 'attack_detected' not in df.columns:
    raise ValueError("❌ 'attack_detected' column missing in dataset.")
le_label = LabelEncoder()
df['attack_detected'] = le_label.fit_transform(df['attack_detected'])
label_encoders['attack_detected'] = le_label
print("✅ Encoded 'attack_detected' (target).")

# Save encoders
joblib.dump(label_encoders, 'network_intrusion_label_encoders.pkl')

# Split features and target
X = df.drop('attack_detected', axis=1)
y = df['attack_detected']

# Verify data types
print(f"Data types of features:\n{X.dtypes}")
print(f"Data type of target: {y.dtype}")

# Balance classes using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print(f"📊 Post-SMOTE class distribution: {Counter(y_resampled)}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
)

# Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'network_intrusion_scaler.pkl')
print("⚖️  Scaling complete.")

# Hyperparameter tuning using RandomizedSearchCV
param_dist = {
    'n_estimators': randint(50, 150),
    'max_depth': [10, 20, 30, 50, None],
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 5)
}
print("🎯 Running RandomizedSearchCV...")
search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, class_weight='balanced'),
    param_distributions=param_dist,
    n_iter=10,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=2
)
# After RandomizedSearchCV fitting
search.fit(X_train_scaled, y_train)
model = search.best_estimator_
print(f"✅ Best Parameters: {search.best_params_}")

# Predict on train and test sets and calculate accuracy & F1
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

train_accuracy = accuracy_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred, average='weighted')

val_accuracy = accuracy_score(y_test, y_test_pred)
val_f1 = f1_score(y_test, y_test_pred, average='weighted')

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Training F1 Score: {train_f1:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Validation F1 Score: {val_f1:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le_label.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix on Test Data")
plt.show()

# Cross-validation accuracy scores and plot
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"🔍 Cross-validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

plt.figure(figsize=(8,5))
plt.plot(range(1, 6), cv_scores, marker='o', linestyle='-', color='blue')
plt.title("Cross-Validation Accuracy Scores")
plt.xlabel("Fold Number")
plt.ylabel("Accuracy")
plt.ylim([0, 1])
plt.grid(True)
plt.show()

# Classification report
print("\n📊 Classification Report:")
print(classification_report(y_test, y_test_pred, target_names=le_label.classes_.astype(str)))

# Save the trained model
joblib.dump(model, 'network_intrusion_model.pkl')
print("💾 Model saved as 'network_intrusion_model.pkl'.")

# Save the feature order
joblib.dump(X.columns.tolist(), 'network_intrusion_feature_order.pkl')
print("💾 Feature order saved as 'network_intrusion_feature_order.pkl'.")
