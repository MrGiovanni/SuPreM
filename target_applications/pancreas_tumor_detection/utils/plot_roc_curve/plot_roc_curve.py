'''
python plot_roc_curve.py --datafile ./patientID_pr_gt_pos_flag.csv --saverocpath .
'''

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

def plot_roc_curve(TPR, FPR, thresholds, args,
                   linewidth=5, elinewidth=15, fontsize=80, figsize=(40,40), 
                   markersize=600, alpha=0.25,
                   zoomin=False,
                  ):
    
    baseline_sensitivity, baseline_specificity = 0.924, 0.905

    plt.rcParams.update({'font.size': fontsize})
    plt.rcParams['axes.linewidth'] = linewidth
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    plt.plot(FPR, TPR, color=[237/255,16/255,105/255], label='SuPreM', linewidth=elinewidth)
    plt.scatter(1-baseline_specificity, baseline_sensitivity, color='royalblue', label='Xia et al.', s=markersize)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')

    plt.axis('square')
    if zoomin:
        plt.xlim(0, 0.2); plt.xticks([0.0, 0.05, 0.1, 0.15, 0.2])
        plt.ylim(0.8, 1.0); plt.yticks([0.85, 0.9, 0.95, 1.0])
    else:
        plt.xlim(0, 1.0); plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        plt.ylim(0.,1.0); plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0])

    plt.grid(alpha=alpha, linewidth=linewidth)

    lg = plt.legend(['SuPreM', 
                     'Xia et al.',
                    ],
                    loc='lower right',
                    fancybox=False, shadow=False, framealpha=1.0, edgecolor='lightgray')
    lg.get_frame().set_linewidth(linewidth)

    plt.plot([1-baseline_specificity, 1-baseline_specificity], [0, 1], '--', color='royalblue', linewidth=linewidth)
    plt.plot([0, 1], [baseline_sensitivity, baseline_sensitivity], '--', color='royalblue', linewidth=linewidth)
    
    if zoomin:
        plt.text(0.15, baseline_sensitivity - 0.02, 'Sensitivity = 92.4%\nSpecificity = 90.5%', 
                fontsize=fontsize, color='royalblue',
                horizontalalignment='center', verticalalignment='center',
                )
        fig.savefig(os.path.join(args.saverocpath, 'roc_curve_zoomin.png'), 
                    bbox_inches='tight', pad_inches=0.5, dpi=200,
                    )
        fig.savefig(os.path.join(args.saverocpath, 'roc_curve_zoomin.pdf'), 
                    bbox_inches='tight', pad_inches=0.5, dpi=200,
                    )
    else:
        plt.text(0.5, baseline_sensitivity - 0.06, 'Sensitivity = 92.4%\nSpecificity = 90.5%', 
                fontsize=fontsize, color='royalblue',
                horizontalalignment='center', verticalalignment='center',
                )

        fig.savefig(os.path.join(args.saverocpath, 'roc_curve.png'), 
                    bbox_inches='tight', pad_inches=0.5, dpi=200,
                    )
        fig.savefig(os.path.join(args.saverocpath, 'roc_curve.pdf'), 
                    bbox_inches='tight', pad_inches=0.5, dpi=200,
                    )
    
    csv_file_path = os.path.join(args.saverocpath, 'roc_curve.csv') 
    data = {'TPR': TPR, 'FPR': FPR, 'thresholds': thresholds}
    df = pd.DataFrame(data)
    df.to_csv(csv_file_path, index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--saverocpath', type=str, help='Path to save ROC curve', default='roc_curve')
    parser.add_argument('--datafile', type=str, required=True, help='Path to the patientID_pr_gt_pos_flag.csv file')
    args = parser.parse_args()

    # Load data from CSV
    data = pd.read_csv(args.datafile, names=["patientID", "pr", "gt_pos_flag"])
    GT = data['gt_pos_flag'].astype(int).values  # Convert boolean flags to integers
    PR = data['pr'].values

    # Calculate ROC metrics
    fpr, tpr, thresholds = metrics.roc_curve(GT, PR)
    plot_roc_curve(tpr, fpr, thresholds, args, zoomin=False)
    plot_roc_curve(tpr, fpr, thresholds, args, zoomin=True)

if __name__ == "__main__":
    main()
