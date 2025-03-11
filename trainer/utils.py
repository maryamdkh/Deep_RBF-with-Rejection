from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os
import seaborn as sns

def plot_confusion_matrix(true_labels, predicted_labels, class_names,save_path):
    """
    Plot and save the confusion matrix.

    Args:
        true_labels (list or np.array): Ground truth labels.
        predicted_labels (list or np.array): Predicted labels.
        class_names (list): Names of the classes.
    """
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(save_path,"CM.png"))
    plt.show()

def generate_classification_report(all_doctor_labels, all_predicted_labels,report_path):
    """
    Generate and save a classification report and confusion matrix.

    Args:
        all_doctor_labels (list): List of actual labels.
        all_predicted_labels (list): List of predicted labels.
        report_path : Path to a .txt file to save results.


    Returns:
        report (str): Classification report.
    """
    # Generate classification report
    target_names = ["control", "parkinson", "rejected"]
    report = classification_report(all_doctor_labels, all_predicted_labels, target_names=target_names)

    # Print and save classification report
    print("Classification Report:")
    print(report)

    # Save classification report to a file
    with open(os.path.join(save_path,"CM.txt"), "w") as f:
        f.write(report)

    print(f"Classification report saved to {report_path}")

    # Plot confusion matrix
    plot_confusion_matrix(all_doctor_labels, all_predicted_labels, target_names,save_path)

    return report