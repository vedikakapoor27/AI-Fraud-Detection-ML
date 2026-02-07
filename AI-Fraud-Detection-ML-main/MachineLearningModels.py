import pandas as pd
import seaborn as sns
import plotly.express as px
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# ---------------- GLOBAL STORAGE ---------------- #
all_performances = pd.DataFrame(
    columns=["model_name", "precision", "recall", "f1_score", "AUC"]
)

list_clf_name = []
list_pred = []
list_model = []


# ---------------- HELPER FUNCTIONS ---------------- #
def fit_model(model, X_train, y_train):
    return model.fit(X_train, y_train)


def add_list(name, model, y_pred):
    list_clf_name.append(name)
    list_model.append(model)
    list_pred.append(y_pred)


def add_all_performances(name, precision, recall, f1, auc):
    global all_performances

    row = pd.DataFrame(
        [[name, precision, recall, f1, auc]],
        columns=["model_name", "precision", "recall", "f1_score", "AUC"],
    )

    all_performances = pd.concat([all_performances, row], ignore_index=True)
    all_performances.drop_duplicates(inplace=True)


def calculate_scores(y_test, y_pred, name, model):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)

    print(f"\n{name} Results:")
    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("AUC      :", round(auc, 4))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    add_list(name, model, y_pred)
    add_all_performances(name, precision, recall, f1, auc)


def model_performance(model, X_train, X_test, y_train, y_test):
    name = model.__class__.__name__

    trained_model = fit_model(model, X_train, y_train)
    y_pred = trained_model.predict(X_test)

    print(f"\n***** {name} DONE *****")
    calculate_scores(y_test, y_pred, name, model)


def display_all_confusion_matrices(y_test):
    total = len(list_pred)

    plt.figure(figsize=(5 * total, 4))

    for i in range(total):
        plt.subplot(1, total, i + 1)

        cf_matrix = confusion_matrix(y_test, list_pred[i])
        disp = ConfusionMatrixDisplay(cf_matrix)
        disp.plot(colorbar=False)

        plt.title(list_clf_name[i])

    plt.tight_layout()
    plt.show()


# ---------------- MAIN EXECUTION ---------------- #
if __name__ == "__main__":

    print("🚀 Running Fraud Detection ML Pipeline...\n")

    # Using built-in dataset for demo (works instantly)
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Models to train
    models = [
        LogisticRegression(max_iter=5000),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
    ]

    # Train & evaluate
    for model in models:
        model_performance(model, X_train, X_test, y_train, y_test)

    # Show performance table
    print("\n📊 Overall Model Performance:")
    print(all_performances.sort_values(by="f1_score", ascending=False))

    # Show confusion matrices
    display_all_confusion_matrices(y_test)
