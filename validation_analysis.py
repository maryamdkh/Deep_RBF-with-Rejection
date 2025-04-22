import pandas as pd
import matplotlib.pylab as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


df = pd.read_csv("deep_rbf_results/nofeex_dim512to256_reject_margin850_l2/valiadtion_results_850.csv")

unknown_samples = df[(df['doctor_label'] == 2) & (df['predicted_label'] != 2)] # get only 

plt.figure(figsize=(10, 8))

y_true = [0 if label == 'control' else 1 for label in unknown_samples['real_label']]
y_pred = unknown_samples['predicted_label']
cm = confusion_matrix(y_true, y_pred)

sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues',
    xticklabels=['Control (0)', 'Parkinson (1)'],
    yticklabels=['Control (0)', 'Parkinson (1)']
)
            

plt.title('Analysis of Misclassified Unknown Samples\n(Doctor Label=2 but Model Prediction≠2)', fontsize=14, pad=20)
plt.ylabel('Real Labels (Actual label of images)', fontsize=12)
plt.xlabel('Model Predictions', fontsize=12)
plt.savefig('deep_rbf_results/nofeex_dim512to256_reject_margin850_l2/Mis_rejected_samples.png', bbox_inches='tight', dpi=300)
plt.close()
