"""Data cleaning, preprocessing and model training for the loan approval system."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score)

TARGET = "Loan_Approved"
ID_COL = "Applicant_ID"
DATA_FILE = "loan_approval_data.csv"


def load_data(file=None):
    """Read the CSV (uploaded file object or default path)."""
    return pd.read_csv(file if file is not None else DATA_FILE)


def prepare_data(raw):
    """Drop duplicates / ID, split features and target, find column types."""
    df = raw.drop_duplicates().copy()
    if ID_COL in df.columns:
        df = df.drop(columns=ID_COL)
    df = df.dropna(subset=[TARGET])

    X = df.drop(columns=TARGET)
    y_raw = df[TARGET].astype(str)
    classes = sorted(y_raw.unique())
    positive = "Yes" if "Yes" in classes else classes[-1]
    y = (y_raw == positive).astype(int)

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    return df, X, y, y_raw, num_cols, cat_cols, positive


def build_preprocessor(num_cols, cat_cols):
    """Impute + scale numbers, impute + one-hot encode categories."""
    numeric = Pipeline([("imp", SimpleImputer(strategy="mean")),
                        ("sc", StandardScaler())])
    categorical = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                            ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore",
                                                  sparse_output=False))])
    return ColumnTransformer([("num", numeric, num_cols),
                              ("cat", categorical, cat_cols)])


def train_models(X, y, num_cols, cat_cols, test_size=0.2, seed=42):
    """Train Logistic Regression, KNN and Naive Bayes.

    Returns (fitted_pipelines, metrics_dataframe, confusion_matrices).
    """
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
    }

    fitted, rows, cms = {}, [], {}
    for name, clf in models.items():
        pipe = Pipeline([("pre", build_preprocessor(num_cols, cat_cols)), ("clf", clf)])
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)

        fitted[name] = pipe
        cms[name] = confusion_matrix(y_te, pred)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_te, pred),
            "Precision": precision_score(y_te, pred, zero_division=0),
            "Recall": recall_score(y_te, pred, zero_division=0),
            "F1 Score": f1_score(y_te, pred, zero_division=0),
        })
    return fitted, pd.DataFrame(rows).set_index("Model"), cms


def predict_one(pipeline, inputs, columns):
    """Predict a single applicant. Returns (0/1 label, approval probability)."""
    row = pd.DataFrame([inputs])[columns]
    return int(pipeline.predict(row)[0]), float(pipeline.predict_proba(row)[0][1])
