import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.preprocessing import label_binarize

# 1. Setup Labels
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
n_classes = len(EMOTIONS)

def plot_metrics(y_true, y_pred_probs):
    y_pred = np.argmax(y_pred_probs, axis=1)

    # --- PART 1: CONFUSION MATRIX ---
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.title('Confusion Matrix: Emotion Detection')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show(block=True) # Forces the window to stay open

    # --- PART 2: ROC CURVE ---
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    plt.figure(figsize=(10, 8))
    colors = ['#ff3cac', '#00ff9d', '#ff9900', '#00e5ff', '#788cff', '#ffe44d', '#8899aa']
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2,
                 label=f'ROC {EMOTIONS[i]} (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-Class ROC Curve')
    plt.legend(loc="lower right")
    plt.show(block=True) # Forces the window to stay open

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=EMOTIONS))

# --- IMPORTANT: CONNECT YOUR DATA ---
# Since you have a trained model, you need to load your validation data here.
# For the demo, if you just want to see what it looks like with your 64% accuracy:
if __name__ == "__main__":
    # Simulated data for demonstration
    # In a real test, replace these with: 
    # y_true = validation_generator.classes
    # y_pred_probs = model.predict(validation_generator)
    
    print("Generating metrics based on current model performance...")
    dummy_true = np.random.randint(0, 7, 100)
    dummy_preds = np.random.dirichlet(np.ones(7), 100) 
    plot_metrics(dummy_true, dummy_preds)