import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)    


def class_accuracies(y_test, y_pred):
    cleaned_y_test = y_test.argmax(axis=1)
    classes = pd.Series(cleaned_y_test).unique()
    for class_val in classes:
        correct = 0
        total = 0
        for i, val in enumerate(cleaned_y_test):
            if val == y_pred[i] and val == class_val:
                correct += 1
            if class_val == cleaned_y_test[i]:
                total += 1
        print(f"Class {class_val} Accuracy: {correct / total}")


def confusion_matrix_plot(y_test, y_pred, n_classes, title="Confusion Matrix"):
    cleaned_y_test = y_test.argmax(axis=1)
    confusion = confusion_matrix(cleaned_y_test, y_pred)
    classes = [i for i in range(n_classes)]

    df_cm = pd.DataFrame(confusion, index=classes, columns=classes)
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, cmap="coolwarm", annot=True, fmt="d")
    plt.ylabel("True Values")
    plt.xlabel("Predicted Values")
    plt.title(title)
    plt.show()


def stats(y_test, y_pred, n_classes):
    cleaned_y_test = y_test.argmax(axis=1)
    print(f"\nAccuracy: {accuracy_score(cleaned_y_test, y_pred):.2f}\n")

    print(
        "Micro Precision: "
        + f"{precision_score(cleaned_y_test, y_pred, average='micro'):.2f}"
    )
    print(
        "Micro Recall: "
        + f"{recall_score(cleaned_y_test, y_pred, average='micro'):.2f}"
    )
    print(
        "Micro F1-score: "
        + f"{f1_score(cleaned_y_test, y_pred, average='micro'):.2f}\n"
    )

    print(
        "Macro Precision: "
        + f"{precision_score(cleaned_y_test, y_pred, average='macro'):.2f}"
    )
    print(
        "Macro Recall: "
        + f"{recall_score(cleaned_y_test, y_pred, average='macro'):.2f}"
    )
    print(
        "Macro F1-score: "
        + f"{f1_score(cleaned_y_test, y_pred, average='macro'):.2f}\n"
    )

    print(
        "Weighted Precision: "
        + f"{precision_score(cleaned_y_test, y_pred, average='weighted'):.2f}"
    )
    print(
        "Weighted Recall: "
        + f"{recall_score(cleaned_y_test, y_pred, average='weighted'):.2f}"
    )
    print(
        "Weighted F1-score: "
        + f"{f1_score(cleaned_y_test, y_pred, average='weighted'):.2f}"
    )

    classes = ["Class " + str(i) for i in range(0, n_classes)]
    print("\nClassification Report\n")
    print(classification_report(cleaned_y_test, y_pred, target_names=classes))
