
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="PCA LDA ICA Analysis", layout="wide")

st.title("Comparative Analysis of PCA, LDA and ICA")
st.write("Breast Cancer Wisconsin Dataset")

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

labels = pd.Series(y).map({
    0: "Malignant",
    1: "Benign"
})

# Standardisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

st.header("1. Dataset Overview")

col1, col2, col3 = st.columns(3)
col1.metric("Total Samples", X.shape[0])
col2.metric("Original Features", X.shape[1])
col3.metric("Classes", len(np.unique(y)))

st.dataframe(X.head())

# PCA
st.header("2. Principal Component Analysis (PCA)")

pca_all = PCA().fit(X_scaled)
variance = np.cumsum(pca_all.explained_variance_ratio_)

pca_count = np.argmax(variance >= 0.95) + 1

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

st.write("Components required for 95% variance:", pca_count)
st.write("Variance retained by 2 components:",
         round(sum(pca.explained_variance_ratio_) * 100, 2), "%")

fig, ax = plt.subplots()
ax.plot(range(1, len(variance)+1), variance, marker="o")
ax.axhline(0.95, linestyle="--")
ax.set_xlabel("Number of Components")
ax.set_ylabel("Cumulative Variance")
ax.set_title("PCA Explained Variance")
st.pyplot(fig)

fig, ax = plt.subplots()
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1],
                hue=labels, ax=ax)
ax.set_title("PCA Scatter Plot")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
st.pyplot(fig)

# LDA
st.header("3. Linear Discriminant Analysis (LDA)")

lda = LinearDiscriminantAnalysis(n_components=1)
X_lda = lda.fit_transform(X_scaled, y)

st.write("Number of LDA components:", X_lda.shape[1])

jitter = np.random.default_rng(42).normal(0, 0.02, len(y))

fig, ax = plt.subplots()
sns.scatterplot(x=X_lda[:, 0], y=jitter,
                hue=labels, ax=ax)
ax.set_title("LDA Class Separation")
ax.set_xlabel("Linear Discriminant 1")
ax.set_yticks([])
st.pyplot(fig)

# ICA
st.header("4. Independent Component Analysis (ICA)")

ica = FastICA(
    n_components=2,
    random_state=42,
    max_iter=1000,
    tol=0.001
)

X_ica = ica.fit_transform(X_scaled)

fig, ax = plt.subplots()
sns.scatterplot(x=X_ica[:, 0], y=X_ica[:, 1],
                hue=labels, ax=ax)
ax.set_title("ICA Scatter Plot")
ax.set_xlabel("Independent Component 1")
ax.set_ylabel("Independent Component 2")
st.pyplot(fig)

st.write("Number of ICA components:", X_ica.shape[1])

# Comparison
st.header("5. Comparative Analysis")

scores = [
    silhouette_score(X_pca, y),
    silhouette_score(X_lda, y),
    silhouette_score(X_ica, y)
]

comparison = pd.DataFrame({
    "Technique": ["PCA", "LDA", "ICA"],
    "Approach": ["Unsupervised", "Supervised", "Unsupervised"],
    "Objective": [
        "Maximum Variance",
        "Class Separation",
        "Statistical Independence"
    ],
    "Uses Class Labels": ["No", "Yes", "No"],
    "Components": [2, 1, 2],
    "Silhouette Score": scores
})

st.dataframe(comparison.round(4), use_container_width=True)

st.subheader("Silhouette Score Comparison")

fig, ax = plt.subplots()
ax.bar(comparison["Technique"], comparison["Silhouette Score"])
ax.set_ylabel("Silhouette Score")
ax.set_title("Comparison of PCA, LDA and ICA")
st.pyplot(fig)

# ICA components
st.header("6. ICA Component Analysis")

ica_table = pd.DataFrame(
    ica.mixing_,
    index=data.feature_names,
    columns=["IC 1", "IC 2"]
)

st.write("ICA Mixing Matrix (first 10 features)")
st.dataframe(ica_table.head(10).round(3))

# Conclusion
st.header("7. Conclusion")

st.write("""
PCA reduces dimensions by preserving maximum variance.
LDA uses class labels to achieve better class separation.
ICA extracts statistically independent components.

All three techniques have different objectives and
are useful in different machine learning applications.
""")
