import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(
    csv_path: str,
    output_path: str,
    normalize: bool = False,
    title: str = "Confusion Matrix",
):
    """
    Read confusion_matrix.csv and save confusion matrix visualization.

    Args:
        csv_path: Path to confusion_matrix.csv.
        output_path: Path to save figure.
        normalize: Whether to normalize confusion matrix by row.
        title: Figure title.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Confusion matrix file not found: {csv_path}")

    cm_df = pd.read_csv(csv_path, index_col=0)
    labels = cm_df.index.tolist()
    cm = cm_df.values.astype(float)

    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1
        cm = cm / row_sum

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = format(cm[i, j], fmt) if normalize else format(int(cm[i, j]), fmt)
            plt.text(
                j,
                i,
                value,
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9,
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Confusion matrix figure saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot confusion matrix from CSV file.")

    parser.add_argument(
        "--csv_path",
        type=str,
        default="outputs/classification/roberta_textcnn/confusion_matrix.csv",
        help="Path to confusion_matrix.csv",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/classification/roberta_textcnn/confusion_matrix.png",
        help="Path to save confusion matrix image",
    )

    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize confusion matrix by row",
    )

    parser.add_argument(
        "--title",
        type=str,
        default="Confusion Matrix",
        help="Figure title",
    )

    args = parser.parse_args()

    plot_confusion_matrix(
        csv_path=args.csv_path,
        output_path=args.output_path,
        normalize=args.normalize,
        title=args.title,
    )

if __name__ == "__main__":
    main()
