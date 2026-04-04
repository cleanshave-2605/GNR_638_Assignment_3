"""
plot_logs.py
------------
Reads the train.log and val.log files produced during training and saves
a set of plots to the result directory.

Usage:
    python plot_logs.py --result_path results/
    python plot_logs.py --result_path results/ --save_fmt pdf
"""

import os
import argparse
import csv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Plot iResNet CIFAR training logs')
parser.add_argument('--result_path', default='results', type=str,
                    help='Path to the result directory containing train.log and val.log')
parser.add_argument('--save_fmt', default='png', type=str, choices=['png', 'pdf', 'svg'],
                    help='Format to save plots (default: png)')


# ---------------------------------------------------------------------------
# Log reader
# ---------------------------------------------------------------------------

def read_log(filepath):
    """Read a tab-delimited log file produced by utils.Logger.
    Returns a dict of {column_name: [values...]}.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError("Log file not found: {}".format(filepath))

    data = {}
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            for key, val in row.items():
                data.setdefault(key, []).append(float(val))
    return data


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_logs(result_path, save_fmt='png'):
    train_log_path = os.path.join(result_path, 'train.log')
    val_log_path   = os.path.join(result_path, 'val.log')

    print("Reading: {}".format(train_log_path))
    train = read_log(train_log_path)

    print("Reading: {}".format(val_log_path))
    val = read_log(val_log_path)

    epochs = train['epoch']

    # -----------------------------------------------------------------------
    # Figure layout: 2 rows x 2 cols
    #   [0,0] Loss (train + val)
    #   [0,1] Top-1 Accuracy (train + val)
    #   [1,0] Top-5 Accuracy (train + val)
    #   [1,1] Learning Rate
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('iResNet CIFAR Training Summary\n({})'.format(result_path),
                 fontsize=13, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.3)

    # -- Loss ----------------------------------------------------------------
    ax_loss = fig.add_subplot(gs[0, 0])
    ax_loss.plot(epochs, train['loss'], label='Train', linewidth=1.5)
    ax_loss.plot(val['epoch'], val['loss'], label='Val', linewidth=1.5, linestyle='--')
    ax_loss.set_title('Loss')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Cross-Entropy Loss')
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    # -- Top-1 Accuracy ------------------------------------------------------
    ax_acc1 = fig.add_subplot(gs[0, 1])
    ax_acc1.plot(epochs, train['acc1'], label='Train', linewidth=1.5)
    ax_acc1.plot(val['epoch'], val['acc1'], label='Val', linewidth=1.5, linestyle='--')
    ax_acc1.set_title('Top-1 Accuracy')
    ax_acc1.set_xlabel('Epoch')
    ax_acc1.set_ylabel('Accuracy (%)')
    ax_acc1.legend()
    ax_acc1.grid(True, alpha=0.3)

    # Annotate best val acc1
    best_acc1 = max(val['acc1'])
    best_epoch = val['epoch'][val['acc1'].index(best_acc1)]
    ax_acc1.annotate('Best: {:.2f}%\n(epoch {:.0f})'.format(best_acc1, best_epoch),
                     xy=(best_epoch, best_acc1),
                     xytext=(best_epoch, best_acc1 - 8),
                     arrowprops=dict(arrowstyle='->', color='red'),
                     color='red', fontsize=9)

    # -- Top-5 Accuracy ------------------------------------------------------
    ax_acc5 = fig.add_subplot(gs[1, 0])
    ax_acc5.plot(epochs, train['acc5'], label='Train', linewidth=1.5)
    ax_acc5.plot(val['epoch'], val['acc5'], label='Val', linewidth=1.5, linestyle='--')
    ax_acc5.set_title('Top-5 Accuracy')
    ax_acc5.set_xlabel('Epoch')
    ax_acc5.set_ylabel('Accuracy (%)')
    ax_acc5.legend()
    ax_acc5.grid(True, alpha=0.3)

    # -- Learning Rate -------------------------------------------------------
    ax_lr = fig.add_subplot(gs[1, 1])
    ax_lr.plot(epochs, train['lr'], color='darkorange', linewidth=1.5)
    ax_lr.set_title('Learning Rate Schedule')
    ax_lr.set_xlabel('Epoch')
    ax_lr.set_ylabel('LR')
    ax_lr.set_yscale('log')
    ax_lr.grid(True, alpha=0.3, which='both')

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    out_path = os.path.join(result_path, 'training_curves.{}'.format(save_fmt))
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print("Saved plot to: {}".format(out_path))
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Print summary table to terminal
    # -----------------------------------------------------------------------
    print("\n{:-<55}".format(''))
    print(" Training Summary")
    print("{:-<55}".format(''))
    print(" Total epochs trained : {:.0f}".format(max(epochs)))
    print(" Best Val Top-1 Acc   : {:.2f}% (epoch {:.0f})".format(best_acc1, best_epoch))
    print(" Best Val Top-5 Acc   : {:.2f}%".format(max(val['acc5'])))
    print(" Final Train Loss     : {:.4f}".format(train['loss'][-1]))
    print(" Final Val   Loss     : {:.4f}".format(val['loss'][-1]))
    print(" Final LR             : {:.6f}".format(train['lr'][-1]))
    print("{:-<55}".format(''))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    args = parser.parse_args()
    plot_logs(args.result_path, save_fmt=args.save_fmt)
