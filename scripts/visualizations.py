import numpy as np
import pandas as pd
import seaborn as sns
import json
from matplotlib import pyplot as plt


def plot_losses(filepaths: list):
    """
    Plots training and validation losses vs. epochs.
    """
    # create dataframe
    sns.set_theme()
    df = pd.DataFrame(columns=['Loss', 'Dataset', 'Epochs', 'Model'])
    for filepath in filepaths:
        results = read_json(filepath)
        newdf = pd.DataFrame(columns=df.columns)
        for i, loss in enumerate(results['train_losses']):
            newdf = pd.concat([newdf, pd.DataFrame([{'Loss': loss, 
                                                     'Dataset': 'Train', 
                                                     'Epochs': results['epochs'][i]
                                                     }])], ignore_index=True)
            newdf = pd.concat([newdf, pd.DataFrame([{'Loss': results['val_losses'][i], 
                                                     'Dataset': 'Val', 
                                                     'Epochs': results['epochs'][i],
                                                     }])], ignore_index=True)
        df = pd.concat([df, newdf])
            
    sns.lineplot(data=df, x='Epochs', y='Loss', style='Dataset')
    plt.title("Loss vs. Epochs")
    plt.show()

def read_json(filepath: str):
    """
    Reads contents from a given file and populates a dictionary with result statistics, including:
    training losses, validation losses, steps at which the losses were calculated, learning rates,
    and the experiment settings currently being used.
    """
    f = open(filepath)
    data = json.load(f)

    log_history = data['log_history']
    ops = data["total_flos"]

    results = {'train_losses':[], 
            'val_losses':[],
            'epochs':[],
            'learning_rate': [],
            'compute': ops}


    for log in log_history:
        if 'loss' in log.keys():
            results['train_losses'].append(log['loss'])
            results['epochs'].append(log['epoch'])
            results['learning_rate'].append(log['learning_rate'])
        if 'eval_loss' in log.keys():
            results['val_losses'].append(log['eval_loss'])

    return results          

def main():
    
    fpaths = [
                'scripts/data/snli-10-8-1000-frozen/trainer_state.json',
    ]
    plot_losses(fpaths)

if __name__ == "__main__":
    main()