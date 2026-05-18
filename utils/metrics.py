from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

def compute_metrics(y_true, y_pred):
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    f1_weighted = f1_score(
        y_true,
        y_pred,
        average="weighted",
    )

    acc = accuracy_score(y_true, y_pred)

    return {
        "accuracy": acc,
        "macro_f1": f1_macro,
        "weighted_f1": f1_weighted,
        "macro_precision": p_macro,
        "macro_recall": r_macro,
    }