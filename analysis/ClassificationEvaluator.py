import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

class ParkinsonMultiLabelEvaluator:
    """
    Evaluator for Parkinson's disease classification with three label sources:
    - true_label (0=control, 1=parkinson): Ground truth
    - doctor_label (0,1,2): Labels provided by doctors (2=unknown)
    - predicted_label (0,1,2): Model predictions
    
    Args:
        file_path (str): Path to input file (Excel or CSV)
        true_col (str): Column name for true labels (default 'true_label')
        doctor_col (str): Column name for doctor labels (default 'doctor_label')
        pred_col (str): Column name for predicted labels (default 'predicted_label')
        label_mapping (dict): Optional mapping for string labels
    """
    
    def __init__(self, file_path, 
                 true_col='true_label', 
                 doctor_col='doctor_label', 
                 pred_col='predicted_label',
                 label_mapping=None):
        
        self.file_path = file_path
        self.true_col = true_col
        self.doctor_col = doctor_col
        self.pred_col = pred_col
        
        # Default label mapping (handles strings and numbers)
        self.label_mapping = label_mapping or {
            'control': 0, 'Control': 0, 'CONTROL': 0, '0': 0,
            'parkinson': 1, 'Parkinson': 1, 'PARKINSON': 1, '1': 1,
            'unknown': 2, 'Unknown': 2, 'UNKNOWN': 2, '2': 2
        }
        
        self.df = None
        self.y_true = None
        self.y_doctor = None
        self.y_pred = None
        
        self._load_data()
        self._preprocess_labels()
        self._validate_labels()
    
    def _load_data(self):
        """Load data from Excel or CSV file"""
        file_extension = Path(self.file_path).suffix.lower()
        
        if file_extension in ('.xlsx', '.xls'):
            self.df = pd.read_excel(self.file_path)
        elif file_extension == '.csv':
            self.df = pd.read_csv(self.file_path)
        else:
            raise ValueError("Unsupported file format. Please provide Excel or CSV.")
        
        # Verify columns exist
        required_cols = [self.true_col, self.doctor_col, self.pred_col]
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
    
    def _preprocess_labels(self):
        """Convert all labels to integers using mapping"""
        # Convert true labels
        self.y_true = self.df[self.true_col].apply(
            lambda x: self.label_mapping.get(x, x) if isinstance(x, str) else int(x)
        )
        
        # Convert doctor labels
        self.y_doctor = self.df[self.doctor_col].apply(
            lambda x: self.label_mapping.get(x, x) if isinstance(x, str) else int(x)
        )
        
        # Convert predicted labels
        self.y_pred = self.df[self.pred_col].apply(
            lambda x: self.label_mapping.get(x, x) if isinstance(x, str) else int(x)
        )
        
        # Add to dataframe for easier analysis
        self.df['true_label_int'] = self.y_true
        self.df['doctor_label_int'] = self.y_doctor
        self.df['predicted_label_int'] = self.y_pred
    
    def _validate_labels(self):
        """Validate label values"""
        # True labels should only be 0 or 1
        invalid_true = set(self.y_true.unique()) - {0, 1}
        if invalid_true:
            raise ValueError(f"Invalid true labels found: {invalid_true}. Should be 0 or 1.")
        
        # Doctor and predicted labels should be 0, 1, or 2
        for label_type, values in [('doctor', self.y_doctor), ('predicted', self.y_pred)]:
            invalid = set(values.unique()) - {0, 1, 2}
            if invalid:
                raise ValueError(f"Invalid {label_type} labels found: {invalid}. Should be 0, 1, or 2.")
    
    def plot_doctor_vs_model(self, save_path=None):
        """Plot confusion matrix between doctor labels and model predictions"""
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(self.y_doctor, self.y_pred)
        
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Control (0)', 'Parkinson (1)', 'Unknown (2)'],
            yticklabels=['Control (0)', 'Parkinson (1)', 'Unknown (2)']
        )
        
        plt.title('Doctor Labels vs Model Predictions', fontsize=14, pad=20)
        plt.ylabel('Doctor Labels', fontsize=12)
        plt.xlabel('Model Predictions', fontsize=12)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
    
    def analyze_unknown_disagreements(self):
        """
        Analyze cases where:
        1. Doctor labeled as unknown (2)
        2. Model predicted either control (0) or parkinson (1)
        3. Show the true labels of these cases
        """
        # Get cases where doctor said unknown but model made a prediction
        mask = (self.y_doctor == 2) & (self.y_pred != 2)
        disagreement_cases = self.df[mask].copy()
        
        if disagreement_cases.empty:
            print("No cases found where doctor labeled as unknown but model predicted")
            return None
        
        # Add readable label names
        reverse_map = {v: k for k, v in self.label_mapping.items() if isinstance(k, str)}
        disagreement_cases['true_label_str'] = disagreement_cases['true_label_int'].map(
            lambda x: reverse_map.get(x, x)
        )
        disagreement_cases['predicted_label_str'] = disagreement_cases['predicted_label_int'].map(
            lambda x: reverse_map.get(x, x)
        )
        
        # Create summary statistics
        summary = disagreement_cases.groupby(['predicted_label_int', 'true_label_int']).size().unstack()
        summary.columns = [f"True {reverse_map.get(col, col)}" for col in summary.columns]
        summary.index = [f"Predicted {reverse_map.get(idx, idx)}" for idx in summary.index]
        
        return {
            'raw_data': disagreement_cases,
            'summary': summary,
            'counts': disagreement_cases['true_label_int'].value_counts(),
            'percentages': disagreement_cases['true_label_int'].value_counts(normalize=True) * 100
        }
    
    def plot_unknown_disagreement_analysis(self, save_path=None):
        """Visualize the true labels of cases where doctor said unknown but model predicted"""
        analysis = self.analyze_unknown_disagreements()
        if analysis is None:
            return
        
        counts = analysis['counts']
        percentages = analysis['percentages']
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(
            [self.label_mapping_inverse.get(i, i) for i in counts.index],
            counts.values,
            color=['blue', 'orange']
        )
        
        # Add annotations - FIXED THE tolist() ERROR HERE
        for i, bar in enumerate(bars.patches):  # Using patches instead of bars.tolist()
            height = bar.get_height()
            idx = counts.index[i]
            pct = percentages[idx]
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height} ({pct:.1f}%)',
                    ha='center', va='bottom')
        
        plt.title('True Labels of Model Predictions\n(Where Doctors Said "Unknown")', fontsize=14, pad=20)
        plt.xlabel('True Label')
        plt.ylabel('Number of Cases')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
    
    def generate_full_report(self, save_path=None):
        """Generate comprehensive report with all analyses"""
        report = "Parkinson's Disease Multi-Label Evaluation Report\n"
        report += "="*60 + "\n\n"
        
        # Basic statistics
        report += f"Dataset: {self.file_path}\n"
        report += f"Total samples: {len(self.df)}\n\n"
        
        # Doctor label distribution
        doctor_counts = self.y_doctor.value_counts()
        report += "Doctor Label Distribution:\n"
        report += "-"*40 + "\n"
        for label, count in doctor_counts.items():
            report += f"{self.label_mapping_inverse.get(label, label)}: {count} ({count/len(self.df)*100:.1f}%)\n"
        report += "\n"
        
        # Model vs doctor comparison
        report += "Model vs Doctor Label Comparison:\n"
        report += "-"*40 + "\n"
        comparison = pd.crosstab(
            self.y_doctor.map(self.label_mapping_inverse),
            self.y_pred.map(self.label_mapping_inverse),
            rownames=['Doctor'],
            colnames=['Model']
        )
        report += comparison.to_string() + "\n\n"
        
        # Unknown disagreement analysis
        analysis = self.analyze_unknown_disagreements()
        if analysis is not None and not analysis['raw_data'].empty:
            report += "Cases Where Doctor Said Unknown But Model Predicted:\n"
            report += "-"*40 + "\n"
            report += f"Total such cases: {len(analysis['raw_data'])}\n\n"
            
            report += "True Label Distribution of These Cases:\n"
            for label, count in analysis['counts'].items():
                pct = analysis['percentages'][label]
                report += f"{self.label_mapping_inverse.get(label, label)}: {count} ({pct:.1f}%)\n"
            
            report += "\nDetailed Breakdown:\n"
            report += analysis['summary'].to_string() + "\n\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report
    
    @property
    def label_mapping_inverse(self):
        """Get inverse label mapping for readable reports"""
        return {v: k for k, v in self.label_mapping.items() if isinstance(k, str)}
    
    def save_all_analyses(self, output_dir='./results'):
        """Save all analyses to files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save visualizations
        self.plot_doctor_vs_model(f"{output_dir}/doctor_vs_model_cm.png")
        self.plot_unknown_disagreement_analysis(f"{output_dir}/unknown_disagreement_analysis.png")
        
        # Save report
        self.generate_full_report(f"{output_dir}/full_report.txt")
        
        # Save raw data of unknown disagreements
        analysis = self.analyze_unknown_disagreements()
        if analysis is not None and not analysis['raw_data'].empty:
            analysis['raw_data'].to_csv(f"{output_dir}/unknown_disagreement_cases.csv", index=False)
        
        print(f"All analyses saved to {output_dir}")