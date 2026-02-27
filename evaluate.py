"""
Evaluation Module - Comprehensive Metrics
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import os


@torch.no_grad()
def evaluate_model(model, test_loader, config):
    """Complete model evaluation with all metrics."""
    model.eval()
    model = model.to(config.device)
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("\nEvaluating model...")
    for images, labels in tqdm(test_loader, desc="Evaluating"):
        images = images.to(config.device, non_blocking=True)
        labels = labels.to(config.device, non_blocking=True)
        
        if config.use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(images)
        else:
            outputs = model(images)
        
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Compute metrics
    results = {
        'accuracy': accuracy_score(all_labels, all_preds) * 100,
        'precision': precision_score(all_labels, all_preds, average='macro'),
        'recall': recall_score(all_labels, all_preds, average='macro'),
        'f1': f1_score(all_labels, all_preds, average='macro'),
        'confusion_matrix': confusion_matrix(all_labels, all_preds),
        'per_class_precision': precision_score(all_labels, all_preds, average=None),
        'per_class_recall': recall_score(all_labels, all_preds, average=None),
        'per_class_f1': f1_score(all_labels, all_preds, average=None),
    }
    
    return results


def print_results(results, class_names):
    """Print evaluation results."""
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy  : {results['accuracy']:.2f}%")
    print(f"  Precision : {results['precision']:.3f}")
    print(f"  Recall    : {results['recall']:.3f}")
    print(f"  F1-Score  : {results['f1']:.3f}")
    
    print(f"\nPer-Class Metrics:")
    for idx, class_name in enumerate(class_names):
        print(f"\n  {class_name}:")
        print(f"    Precision: {results['per_class_precision'][idx]:.3f}")
        print(f"    Recall   : {results['per_class_recall'][idx]:.3f}")
        print(f"    F1-Score : {results['per_class_f1'][idx]:.3f}")


def save_confusion_matrix(cm, class_names, save_path):
    """Save confusion matrix visualization."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Confusion matrix saved: {save_path}")


def save_results_table(all_results, config):
    """Save results table to file."""
    save_path = os.path.join(config.output_dir, "results_summary.txt")
    
    with open(save_path, 'w') as f:
        f.write("="*100 + "\n")
        f.write("BRAIN MRI TUMOR CLASSIFICATION - RESULTS SUMMARY\n")
        f.write("="*100 + "\n\n")
        
        f.write(f"{'Model':<20} {'Label %':<10} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
        f.write("-"*100 + "\n")
        
        for result in all_results:
            f.write(f"{result['model']:<20} {result['label_pct']:<10} "
                   f"{result['accuracy']:>10.2f}% {result['precision']:>11.3f} "
                   f"{result['recall']:>11.3f} {result['f1']:>11.3f}\n")
        
        f.write("="*100 + "\n")
    
    print(f"✓ Results table saved: {save_path}")


def create_comparison_plot(all_results, config):
    """Create comparison bar plot."""
    methods = ['supervised', 'simclr_v1_ft', 'simclr_v2_ft']
    label_pcts = config.label_percentages
    
    data = {method: [] for method in methods}
    
    for result in all_results:
        model_type = result['model'].split('_')[0] if '_' in result['model'] else result['model']
        if 'simclr' in result['model'].lower():
            if 'v1' in result['model'].lower():
                data['simclr_v1_ft'].append(result['accuracy'])
            elif 'v2' in result['model'].lower():
                data['simclr_v2_ft'].append(result['accuracy'])
        elif model_type in methods:
            data[model_type].append(result['accuracy'])
    
    # Create plot
    plt.figure(figsize=(12, 6))
    x = np.arange(len(label_pcts))
    width = 0.25
    
    plt.bar(x - width, data['supervised'][:len(label_pcts)], width, label='Supervised', color='#1f77b4')
    plt.bar(x, data['simclr_v1_ft'][:len(label_pcts)], width, label='SimCLR v1', color='#ff7f0e')
    plt.bar(x + width, data['simclr_v2_ft'][:len(label_pcts)], width, label='SimCLR v2', color='#2ca02c')
    
    plt.xlabel('Labeled Data Percentage', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title('Performance Comparison Across Label Percentages', fontsize=14, fontweight='bold')
    plt.xticks(x, [f"{int(p*100)}%" for p in label_pcts])
    plt.legend(fontsize=10)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(config.output_dir, "comparison_plot.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Comparison plot saved: {save_path}")
