import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ---------------- Generate synthetic data ----------------
np.random.seed(42)
number_of_students = 1000

data = {
    "study_hours":           np.clip(np.random.normal(15, 5,   number_of_students), 0, 40),
    "attendance":            np.clip(np.random.normal(80, 10,  number_of_students), 0, 100),
    "previous_grades":       np.clip(np.random.normal(70, 12,  number_of_students), 0, 100),
    "assignments_completed": np.clip(np.random.normal(5,  1.5, number_of_students), 0, 6),
}

df = pd.DataFrame(data)

# Weighted score per student (roughly 0-1 scale)
score = (
    0.20 * (df["study_hours"]           / 15) +
    0.20 * (df["attendance"]            / 100) +
    0.30 * (df["previous_grades"]       / 100) +
    0.25 * (df["assignments_completed"] / 5)
)

# Sigmoid mapping: score -> probability of passing (one per student)
probability = 1 / (1 + np.exp(-(score - 0.6) * 10))

# Sample individual outcomes
df["result"] = np.random.binomial(1, probability)

# Friendly label column for plotting
df["result_label"] = df["result"].map({0: "Fail", 1: "Pass"})

print(df.head())
print("\nClass distribution:")
print(df["result"].value_counts())


# ---------------- Train / test split ----------------
x = df[["study_hours", "attendance", "previous_grades", "assignments_completed"]]
y = df["result"]

X_train, X_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000),
)

model.fit(X_train, y_train)
print("\nModel done training and evaluation.")


# ---------------- Evaluation ----------------
print("Model accuracy on test set: {:.2f}".format(model.score(X_test, y_test)))

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))


# ---------------- Predict a new student ----------------
new_student = pd.DataFrame({
    "study_hours":           [8],
    "attendance":            [85],
    "previous_grades":       [72],
    "assignments_completed": [5],
})

prediction = model.predict(new_student)
probabilities = model.predict_proba(new_student)

if prediction[0] == 1:
    print("\nPredicted Result: PASS")
else:
    print("\nPredicted Result: FAIL")

print("Probability of passing:", round(probabilities[0][1] * 100, 2), "%")


# =====================================================================
# DATA EXPLORATION PLOTS
# =====================================================================

# ---------------- 1. Count plot: pass / fail ----------------
plt.figure(figsize=(6, 4))
sns.countplot(
    data=df,
    x="result_label",
    hue="result_label",
    palette={"Fail": "salmon", "Pass": "lightgreen"},
    legend=False,
)
plt.title("Student Pass and Fail Distribution")
plt.xlabel("Student Result")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.show()


# ---------------- 2. Scatter: study hours vs previous grades ----------------
plt.figure(figsize=(10, 6))
ax = sns.scatterplot(
    data=df,
    x="study_hours",
    y="previous_grades",
    hue="result_label",
    palette={"Fail": "red", "Pass": "green"},
    alpha=0.7,
)
plt.title("Study Hours vs Previous Assessment Marks")
plt.xlabel("Study Hours per Week")
plt.ylabel("Previous Assessment Mark (%)")
ax.legend(title="Student Result")
plt.tight_layout()
plt.show()


# ---------------- 3. Box plot: attendance by result ----------------
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=df,
    x="result_label",
    y="attendance",
    hue="result_label",
    palette={"Fail": "salmon", "Pass": "lightgreen"},
    legend=False,
)
plt.title("Attendance Distribution by Student Result")
plt.xlabel("Student Result")
plt.ylabel("Attendance (%)")
plt.tight_layout()
plt.show()


# =====================================================================
# LOGISTIC REGRESSION VISUALIZATIONS
# =====================================================================

# ---------------- 4. Sigmoid curve ----------------
z = np.linspace(-6, 6, 300)
sigmoid = 1 / (1 + np.exp(-z))

# Convert masks to plain Python lists so type-checkers are happy
mask_pass = (sigmoid >= 0.5).tolist()
mask_fail = (sigmoid < 0.5).tolist()

plt.figure(figsize=(8, 5))
plt.plot(z, sigmoid, color="steelblue", linewidth=2, label="σ(z) = 1 / (1 + e⁻ᶻ)")
plt.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Decision threshold (0.5)")
plt.axvline(0,   color="gray", linestyle=":",  linewidth=1)
plt.fill_between(z, 0, sigmoid, where=mask_pass,
                 color="lightgreen", alpha=0.3, label="Predict PASS")
plt.fill_between(z, 0, sigmoid, where=mask_fail,
                 color="salmon", alpha=0.3, label="Predict FAIL")
plt.title("Logistic Regression — Sigmoid Function")
plt.xlabel("Linear score  z = w·x + b")
plt.ylabel("Probability of passing")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ---------------- 5. 2D decision boundary ----------------
feat_x, feat_y = "study_hours", "previous_grades"

# Cast to plain ndarrays so Pyright treats them as ArrayLike
X2 = np.asarray(df[[feat_x, feat_y]].values)
y2 = np.asarray(df["result"].values)

model2d = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model2d.fit(X2, y2)

x_min, x_max = X2[:, 0].min() - 2, X2[:, 0].max() + 2
y_min, y_max = X2[:, 1].min() - 2, X2[:, 1].max() + 2
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
grid = np.c_[xx.ravel(), yy.ravel()]
probs = model2d.predict_proba(grid)[:, 1].reshape(xx.shape)

plt.figure(figsize=(10, 7))
contour = plt.contourf(xx, yy, probs, levels=20, cmap="RdYlGn", alpha=0.6)
plt.colorbar(contour, label="P(pass)")
plt.contour(xx, yy, probs, levels=[0.5], colors="black",
            linewidths=2, linestyles="--")
plt.scatter(X2[:, 0], X2[:, 1], c=y2, cmap="RdYlGn",
            edgecolor="k", s=40, alpha=0.85)
plt.xlabel("Study Hours per Week")
plt.ylabel("Previous Assessment Mark (%)")
plt.title("Logistic Regression Decision Boundary\n"
          f"({feat_x} vs {feat_y}) — black dashed = 50% threshold")
plt.tight_layout()
plt.show()


# ---------------- 6. Feature coefficients ----------------
logreg = model.named_steps["logisticregression"]
coefs = logreg.coef_[0]
feature_names = ["study_hours", "attendance",
                 "previous_grades", "assignments_completed"]
order = np.argsort(np.abs(coefs))[::-1]
coefs_sorted = coefs[order]
names_sorted = [feature_names[i] for i in order]
colors = ["green" if c > 0 else "red" for c in coefs_sorted]

plt.figure(figsize=(8, 5))
plt.barh(names_sorted, coefs_sorted, color=colors, edgecolor="black")
plt.axvline(0, color="black", linewidth=1)
plt.title("Logistic Regression Coefficients\n"
          "(positive → increases chance of PASS)")
plt.xlabel("Coefficient value (on standardized features)")
plt.gca().invert_yaxis()
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.show()