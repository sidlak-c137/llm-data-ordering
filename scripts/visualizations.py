import numpy as np
import pandas as pd
import seaborn as sns
import json
from matplotlib import pyplot as plt


def plot_losses(filepaths: list, output_file: str):
    """
    Plots training and validation losses vs. steps.
    """
    # create dataframe
    sns.set_theme()
    df = pd.DataFrame(columns=['Loss', 'Dataset', 'Steps', 'Loss Function'])
    for filepath in filepaths:
        results = pd.read_csv(filepath)
        newdf = pd.DataFrame(columns=df.columns)
        for i, loss in enumerate(results['train_loss']):
            newdf = pd.concat([newdf, pd.DataFrame([{'Loss': loss, 
                                                     'Dataset': 'Train', 
                                                     'Steps': results.at[i, 'step'],
                                                     }])], ignore_index=True)
            newdf = pd.concat([newdf, pd.DataFrame([{'Loss': results.at[i, 'val_loss'], 
                                                     'Dataset': 'Val', 
                                                     'Steps': results.at[i, 'step'],
                                                     }])], ignore_index=True)
        df = pd.concat([df, newdf])
            
    sns.lineplot(data=df, x='Steps', y='Loss', style='Dataset')
    plt.title("Loss vs. Steps")
    plt.savefig(output_file)

# def read_json(filepath: str):
#     """
#     Reads contents from a given file and populates a dictionary with result statistics, including:
#     training losses, validation losses, steps at which the losses were calculated, learning rates,
#     and the experiment settings currently being used.
#     """
#     f = open(filepath)
#     data = json.load(f)

#     log_history = data['log_history']
#     ops = data["total_flos"]

#     results = {'train_losses':[], 
#             'val_losses':[],
#             'epochs':[],
#             'learning_rate': [],
#             'compute': ops}


#     for log in log_history:
#         if 'loss' in log.keys():
#             results['train_losses'].append(log['loss'])
#             results['epochs'].append(log['epoch'])
#             results['learning_rate'].append(log['learning_rate'])
#         if 'eval_loss' in log.keys():
#             results['val_losses'].append(log['eval_loss'])

#     return results          

def main():
    
    fpaths = [
                'scripts/experiment_data/snli_pretrained_all_metrics.csv',
    ]
    plot_losses(fpaths, "figures/snli/snli_pretrained_all_metrics_loss.png")

if __name__ == "__main__":
    main()

    # srun python scripts/visualizations.py