import numpy as np
import pandas as pd
import seaborn as sns
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed
from training.hardness_datasets import SNLICartographyDataset, SNLINgramPerplexityDataset
from matplotlib import pyplot as plt

def replace_df_names(df):
    df.replace('increasing', 'Ascending', inplace=True)
    df.replace('decreasing', 'Descending', inplace=True)
    df.replace('inv-triangle', 'Descending', inplace=True)
    df.replace('random', 'Baseline Random', inplace=True)
    df.replace('triangle', 'Ascending', inplace=True)
    df.replace('confidence', 'Confidence', inplace=True)
    df.replace('perplexity', 'Perplexity', inplace=True)
    df.replace('variability', 'Variability', inplace=True)
    df.replace('even-scaled-confidence', 'Normalized Confidence', inplace=True)
    df.replace('even-scaled-variability', 'Normalized Variability', inplace=True)
    df.replace('baseline', 'Baseline', inplace=True) 
    df['Experiment'] = df['Hardness Calculation'] + ", " + df['Order']
    return df

def plot_simple_validation_losses(filepaths: list):
    """
    Plots training and validation losses vs. steps for single GPU experiments.
    """
    # create dataframe
    df = pd.DataFrame(columns=['Validation Loss', 'Steps', 'Hardness Calculation', 'Order'])
    for filepath in filepaths:
        # parse for hardness and order from filename
        path = filepath.split('/')[-1]
        vals = path.split('_')
        hardness, order = vals[2], vals[1]
        results = pd.read_csv(filepath)
        newdf = pd.DataFrame(columns=df.columns)
        for i, loss in enumerate(results['val_loss']):
            newdf = pd.concat([newdf, pd.DataFrame([{'Validation Loss': loss, 
                                                     'Steps': results.at[i, 'step'],
                                                     'Hardness Calculation': hardness,
                                                     'Order': order,
                                                     }])], ignore_index=True)
        df = pd.concat([df, newdf])

    # replace all hardness and order values with actual names
    df = replace_df_names(df)

    baselines = df['Hardness Calculation'] == 'Baseline'
    df_baselines = df[baselines]
    confidences = df['Hardness Calculation'] == 'Confidence'
    df_confidences = df[confidences]
    perplexities = df['Hardness Calculation'] == 'Perplexity'
    df_perplexities = df[perplexities]
    variabilities = df['Hardness Calculation'] == 'Variability'
    df_variabilities = df[variabilities]

    palette = ["#000000", "#0091ea", "#1de9b6"]
    sns.set_palette(palette)

    ax = sns.lineplot(data=pd.concat([df_baselines, df_confidences]), x='Steps', y='Validation Loss', hue='Order')
    ax.set_title("Validation Loss Over Steps: Confidence")
    plt.savefig('figures/snli/snli_ordered_baseline_confidences.svg')
    plt.close()

    ax = sns.lineplot(data=pd.concat([df_baselines, df_variabilities]), x='Steps', y='Validation Loss', hue='Order')
    ax.set_title("Validation Loss Over Steps: Variability")
    plt.savefig('figures/snli/snli_ordered_baseline_variabilities.svg')
    plt.close()

    ax = sns.lineplot(data=pd.concat([df_baselines, df_perplexities]), x='Steps', y='Validation Loss', hue='Order')
    ax.set_title("Validation Loss Over Steps: Perplexity")
    plt.savefig('figures/snli/snli_ordered_baseline_perplexities.svg')
    plt.close()

def plot_multi_validation_losses(filepaths: list):
    """
    Plots training and validation losses vs. steps for multi-gpu settings with all five hardness ordering metrics.
    """
    # create dataframe
    df = pd.DataFrame(columns=['Validation Loss', 'Steps', 'Hardness Calculation', 'Order'])
    for filepath in filepaths:
        # parse for hardness and order from filename
        path = filepath.split('/')[-1]
        vals = path.split('_')
        hardness, order = vals[2], vals[1]
        results = pd.read_csv(filepath)
        newdf = pd.DataFrame(columns=df.columns)
        for i, loss in enumerate(results['val_loss']):
            newdf = pd.concat([newdf, pd.DataFrame([{'Validation Loss': loss, 
                                                     'Steps': results.at[i, 'step'],
                                                     'Hardness Calculation': hardness,
                                                     'Order': order,
                                                     }])], ignore_index=True)
        df = pd.concat([df, newdf])

    # replace all hardness and order values with actual names
    df = replace_df_names(df)

    baselines = df['Hardness Calculation'] == 'Baseline'
    df_baselines = df[baselines]
    confidences = df['Hardness Calculation'] == 'Confidence'
    df_confidences = df[confidences]
    es_confidences = df['Hardness Calculation'] == 'Normalized Confidence'
    df_es_confidences = df[es_confidences]
    perplexities = df['Hardness Calculation'] == 'Perplexity'
    df_perplexities = df[perplexities]
    variabilities = df['Hardness Calculation'] == 'Variability'
    df_variabilities = df[variabilities]
    es_variabilities = df['Hardness Calculation'] == 'Normalized Variability'
    df_es_variabilities = df[es_variabilities]

    palette = ["#000000", "#0091ea", "#1de9b6"]
    sns.set_palette(palette)

    ax = sns.lineplot(data=pd.concat([df_baselines, df_confidences]), x='Steps', y='Validation Loss', hue='Order')
    ax.set_title("Validation Loss Over Steps: Confidence")
    plt.savefig('figures/snli/snli_scaled_baseline_confidences.svg')
    plt.close()

    ax = sns.lineplot(data=pd.concat([df_baselines, df_variabilities]), x='Steps', y='Validation Loss', hue='Order')
    ax.set_title("Validation Loss Over Steps: Variability")
    plt.savefig('figures/snli/snli_scaled_baseline_variabilities.svg')
    plt.close()

    ax = sns.lineplot(data=pd.concat([df_baselines, df_es_confidences]), x='Steps', y='Validation Loss', hue='Order')
    ax.set_title("Validation Loss Over Steps: Normalized Confidence")
    plt.savefig('figures/snli/snli_scaled_baseline_es_confidences.svg')
    plt.close()

    ax = sns.lineplot(data=pd.concat([df_baselines, df_es_variabilities]), x='Steps', y='Validation Loss', hue='Order')
    ax.set_title("Validation Loss Over Steps: Normalized Variability")
    plt.savefig('figures/snli/snli_scaled_baseline_es_variabilities.svg')
    plt.close()

    ax = sns.lineplot(data=pd.concat([df_baselines, df_perplexities]), x='Steps', y='Validation Loss', hue='Order')
    ax.set_title("Validation Loss Over Steps: Perplexity")
    plt.savefig('figures/snli/snli_scaled_baseline_perplexities.svg')
    plt.close()

def plot_validation_accuracies(filepaths: list):
    """
    Plots training and validation losses vs. steps.
    """
    # create dataframe
    df = pd.DataFrame(columns=['Validation Accuracy', 'Steps', 'Hardness Calculation', 'Order'])
    for filepath in filepaths:
        # parse for hardness and order from filename
        path = filepath.split('/')[-1]
        vals = path.split('_')
        hardness, order = vals[2], vals[1]
        results = pd.read_csv(filepath)
        newdf = pd.DataFrame(columns=df.columns)
        for i, acc in enumerate(results['val_acc']):
            newdf = pd.concat([newdf, pd.DataFrame([{'Validation Accuracy': acc, 
                                                     'Steps': results.at[i, 'step'],
                                                     'Hardness Calculation': hardness,
                                                     'Order': order,
                                                     }])], ignore_index=True)
        df = pd.concat([df, newdf])

    # replace all hardness and order values with actual names
    df = replace_df_names(df)

    baselines = df['Hardness Calculation'] == 'Baseline'
    df_baselines = df[baselines]
    confidences = df['Hardness Calculation'] == 'Confidence'
    df_confidences = df[confidences]
    perplexities = df['Hardness Calculation'] == 'Perplexity'
    df_perplexities = df[perplexities]
    variabilities = df['Hardness Calculation'] == 'Variability'
    df_variabilities = df[variabilities]

    palette = ["#000000", "#0091ea", "#1de9b6"]
    sns.set_palette(palette)

    ax = sns.lineplot(data=pd.concat([df_baselines, df_confidences]), x='Steps', y='Validation Accuracy', hue='Order')
    ax.set_title("Validation Accuracy Over Steps: Confidence")
    plt.savefig('figures/snli/snli_ordered_baseline_confidences_val_acc.svg')
    plt.close()

    ax = sns.lineplot(data=pd.concat([df_baselines, df_variabilities]), x='Steps', y='Validation Accuracy', hue='Order')
    ax.set_title("Validation Accuracy Over Steps: Variability")
    plt.savefig('figures/snli/snli_ordered_baseline_variabilities_val_acc.svg')
    plt.close()

    ax = sns.lineplot(data=pd.concat([df_baselines, df_perplexities]), x='Steps', y='Validation Accuracy', hue='Order')
    ax.set_title("Validation Accuracy Over Steps: Perplexity")
    plt.savefig('figures/snli/snli_ordered_baseline_perplexities_val_acc.svg')
    plt.close()

def plot_triangle_validation_accuracies(filepaths: list):
    """
    Plots training and validation accuracies vs. steps for multi-gpu experiments for all hardness ordering metrics.
    """
    # create dataframe
    df = pd.DataFrame(columns=['Validation Accuracy', 'Steps', 'Hardness Calculation', 'Order'])
    for filepath in filepaths:
        # parse for hardness and order from filename
        path = filepath.split('/')[-1]
        vals = path.split('_')
        hardness, order = vals[2], vals[1]
        results = pd.read_csv(filepath)
        newdf = pd.DataFrame(columns=df.columns)
        for i, acc in enumerate(results['val_acc']):
            newdf = pd.concat([newdf, pd.DataFrame([{'Validation Accuracy': acc, 
                                                     'Steps': results.at[i, 'step'],
                                                     'Hardness Calculation': hardness,
                                                     'Order': order,
                                                     }])], ignore_index=True)
        df = pd.concat([df, newdf])

    # replace all hardness and order values with actual names
    df = replace_df_names(df)

    baselines = df['Hardness Calculation'] == 'Baseline'
    df_baselines = df[baselines]
    confidences = df['Hardness Calculation'] == 'Confidence'
    df_confidences = df[confidences]
    es_confidences = df['Hardness Calculation'] == 'Normalized Confidence'
    df_es_confidences = df[es_confidences]
    perplexities = df['Hardness Calculation'] == 'Perplexity'
    df_perplexities = df[perplexities]
    variabilities = df['Hardness Calculation'] == 'Variability'
    df_variabilities = df[variabilities]
    es_variabilities = df['Hardness Calculation'] == 'Normalized Variability'
    df_es_variabilities = df[es_variabilities]

    palette = ["#000000", "#0091ea", "#1de9b6"]
    sns.set_palette(palette)

    ax = sns.lineplot(data=pd.concat([df_baselines, df_confidences]), x='Steps', y='Validation Accuracy', hue='Order')
    ax.set_title("Validation Accuracy Over Steps: Confidence")
    plt.savefig('figures/snli/snli_scaled_baseline_confidences_val_acc.svg')
    plt.close()

    ax = sns.lineplot(data=pd.concat([df_baselines, df_variabilities]), x='Steps', y='Validation Accuracy', hue='Order')
    ax.set_title("Validation Accuracy Over Steps: Variability")
    plt.savefig('figures/snli/snli_scaled_baseline_variabilities_val_acc.svg')
    plt.close()

    ax = sns.lineplot(data=pd.concat([df_baselines, df_es_confidences]), x='Steps', y='Validation Accuracy', hue='Order')
    ax.set_title("Validation Accuracy Over Steps: Normalized Confidence")
    plt.savefig('figures/snli/snli_scaled_baseline_normalized_confidences_val_acc.svg')
    plt.close()

    ax = sns.lineplot(data=pd.concat([df_baselines, df_es_variabilities]), x='Steps', y='Validation Accuracy', hue='Order')
    ax.set_title("Validation Accuracy Over Steps: Normalized Variability")
    plt.savefig('figures/snli/snli_scaled_baseline_normalized_variabilities_val_acc.svg')
    plt.close()

    ax = sns.lineplot(data=pd.concat([df_baselines, df_perplexities]), x='Steps', y='Validation Accuracy', hue='Order')
    ax.set_title("Validation Accuracy Over Steps: Perplexity")
    plt.savefig('figures/snli/snli_scaled_baseline_perplexities_val_acc.svg')
    plt.close()

def get_ngram_perplexities(dataset, perplexities_jsonl_path, tokenizer):
    if dataset == 'snli':
        train_snli = load_dataset(dataset, split="train")
        train_snli = train_snli.filter(lambda sample: sample["label"] != -1)
        train_dataset = SNLINgramPerplexityDataset(perplexities_jsonl_path, train_snli, 50000, tokenizer, False)
    else:
        raise ValueError(f"Dataset {dataset} unsupported.")
    train_dataset.sort_by_hardness("increasing")
    return train_dataset['hardness']

def get_datamap_hardnesses(dataset, coordinates_path, tokenizer, hardness):
    if dataset == 'snli':
        train_snli = load_dataset(dataset, split="train")
        train_snli = train_snli.filter(lambda sample: sample["label"] != -1)
        train_dataset = SNLICartographyDataset(coordinates_path, train_snli, 50000, tokenizer, False, hardness)
    else:
        raise ValueError(f"Dataset {dataset} unsupported.")
    train_dataset.sort_by_hardness("increasing")
    return train_dataset['hardness']

def plot_hardness_distributions(output_path, perplexities=None, confidences=None, variabilities=None):
    """
    Plots hardness distributions.
    """
    palette = ["#0091ea", "#1de9b6", "#ff7043"]
    sns.set_palette(palette)
    df = pd.DataFrame(columns=['Hardness', 'Classification'])
    if perplexities is not None:
        new_rows = pd.DataFrame({'Classification': ['Perplexity'] * len(perplexities), 'Hardness': perplexities})
        df = pd.concat([df, new_rows], ignore_index=True)
    if confidences is not None:
        new_rows = pd.DataFrame({'Classification': ['Confidence'] * len(confidences), 'Hardness': confidences})
        df = pd.concat([df, new_rows], ignore_index=True)
    if variabilities is not None:
        new_rows = pd.DataFrame({'Classification': ['Variability'] * len(variabilities), 'Hardness': variabilities})
        df = pd.concat([df, new_rows], ignore_index=True)

    sns.kdeplot(df, x='Hardness', hue='Classification', multiple="stack")
    plt.title('Distribution of Data Hardness Classifications')
    plt.savefig(output_path)
    plt.close()

def plot_all_hardness_distributions():
    # to get the correct samples, have to get datasets (for now)
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # plot distributions of hardness values
    perplexities = get_ngram_perplexities("snli", "data/interpolated_ngram_perplexity.jsonl", tokenizer)
    print(f"lowest perplexities: {perplexities[:10]}, highest: {perplexities[-10:]}")
    variabilities = get_datamap_hardnesses("snli", "data/snli_data_map_coordinates.pickle", tokenizer, "variability")
    print(f"lowest var: {variabilities[:10]}, highest: {variabilities[-10:]}")
    confidences = get_datamap_hardnesses("snli", "data/snli_data_map_coordinates.pickle", tokenizer, "confidence")
    print(f"lowest confidences: {confidences[:10]}, highest: {confidences[-10:]}")
    plot_hardness_distributions(perplexities=perplexities, output_path='figures/snli/snli_hardness_distributions_perplexities.svg')
    plot_hardness_distributions(variabilities=variabilities, output_path='figures/snli/snli_hardness_distributions_variabilities.svg')
    plot_hardness_distributions(confidences=confidences, output_path='figures/snli/snli_hardness_distributions_confidences.svg')

    # plot distributions of scaled hardness values
    if perplexities is not None and variabilities is not None and confidences is not None:
        min_p, max_p = min(perplexities).item(), max(perplexities).item()
        scaled_perplexities = [(val.item() - min_p) / (max_p - min_p) for val in perplexities]
        min_v, max_v = min(variabilities).item(), max(variabilities).item()
        scaled_variabilities = [(val.item() - min_v) / (max_v - min_v) for val in variabilities]
        min_c, max_c = min(confidences).item(), max(confidences).item()
        scaled_confidences = [(val.item() - min_c) / (max_c - min_c) for val in confidences]
        print(f"lens: {len(scaled_perplexities)}, {len(scaled_variabilities)}, {len(scaled_confidences)}")
        print(f"perplexities: {scaled_perplexities[:10]}")
        plot_hardness_distributions(perplexities=scaled_perplexities,
                                    variabilities=scaled_variabilities,
                                    confidences=scaled_confidences, 
                                    output_path='figures/snli/snli_hardness_distributions_scaled.svg')

def plot_test_accuracies(filepaths, output_path):
    df = pd.DataFrame(columns=['Test Accuracy', 'Hardness Calculation', 'Order'])
    for filepath in filepaths:
        # parse for hardness and order from filename
        filename = filepath.split('/')[-1]
        _, order, hardness, _ = filename.split('_')
        with open(filepath, 'r') as f:
            _, _, test_acc, _, _, test_loss = f.readlines()[-1].replace(': ', ' ').replace(', ', ' ').split()
            test_acc, test_loss = float(test_acc), float(test_loss)
        newdf = pd.DataFrame({'Hardness Calculation': [hardness], 'Order': [order], 'Test Accuracy': [test_acc]})
        df = pd.concat([df, newdf], ignore_index=True)

    df = replace_df_names(df)

    palette = ["#0091ea", "#1de9b6", "#ff7043"]
    sns.set_palette(palette)

    # Plot grouped bars
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="Hardness Calculation", y="Test Accuracy", hue="Order", data=df[df['Hardness Calculation'] != "Baseline"])
    # set y limits
    plt.ylim((.6, .75))
    # Add a horizontal line for the baseline
    baseline_value = df[df['Hardness Calculation'] == 'Baseline']['Test Accuracy'].iloc[0]
    plt.axhline(y=baseline_value, color='black', linestyle='--', label='Baseline')

    # Set labels and title
    plt.xlabel('Hardness Calculation')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy by Hardness Calculation and Order')

    # Show the plot
    plt.legend(title='Order')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    sns.set_style('whitegrid')
    palette = ["#0091ea", "#1de9b6", "#ff7043"]
    sns.set_palette(palette)
    set_seed(42)
    ordered_fpaths = [
        'scripts/experiment_data/single-gpu-ordered-experiments/snli_random_baseline_metrics.csv',
        'scripts/experiment_data/single-gpu-ordered-experiments/snli_decreasing_confidence_metrics.csv',
        'scripts/experiment_data/single-gpu-ordered-experiments/snli_increasing_confidence_metrics.csv',
        'scripts/experiment_data/single-gpu-ordered-experiments/snli_decreasing_perplexity_metrics.csv',
        'scripts/experiment_data/single-gpu-ordered-experiments/snli_increasing_perplexity_metrics.csv',
        'scripts/experiment_data/single-gpu-ordered-experiments/snli_decreasing_variability_metrics.csv',
        'scripts/experiment_data/single-gpu-ordered-experiments/snli_increasing_variability_metrics.csv'
    ]
    multi_fpaths = [
        'scripts/experiment_data/multi-gpu-experiments/snli_inv-triangle_confidence_metrics.csv',
        'scripts/experiment_data/multi-gpu-experiments/snli_inv-triangle_even-scaled-confidence_metrics.csv',
        'scripts/experiment_data/multi-gpu-experiments/snli_inv-triangle_perplexity_metrics.csv',
        'scripts/experiment_data/multi-gpu-experiments/snli_inv-triangle_variability_metrics.csv',
        'scripts/experiment_data/multi-gpu-experiments/snli_inv-triangle_even-scaled-variability_metrics.csv',
        'scripts/experiment_data/multi-gpu-experiments/snli_random_baseline_metrics.csv',
        'scripts/experiment_data/multi-gpu-experiments/snli_triangle_confidence_metrics.csv',
        'scripts/experiment_data/multi-gpu-experiments/snli_triangle_even-scaled-confidence_metrics.csv',
        'scripts/experiment_data/multi-gpu-experiments/snli_triangle_even-scaled-variability_metrics.csv',
        'scripts/experiment_data/multi-gpu-experiments/snli_triangle_perplexity_metrics.csv',
        'scripts/experiment_data/multi-gpu-experiments/snli_triangle_variability_metrics.csv'        
    ]

    ordered_fpaths_test = [
        'scripts/experiment_data/single-gpu-ordered-experiments/snli_random_baseline_test.txt',
        'scripts/experiment_data/single-gpu-ordered-experiments/snli_decreasing_confidence_test.txt',
        'scripts/experiment_data/single-gpu-ordered-experiments/snli_increasing_confidence_test.txt',
        'scripts/experiment_data/single-gpu-ordered-experiments/snli_decreasing_perplexity_test.txt',
        'scripts/experiment_data/single-gpu-ordered-experiments/snli_increasing_perplexity_test.txt',
        'scripts/experiment_data/single-gpu-ordered-experiments/snli_decreasing_variability_test.txt',
        'scripts/experiment_data/single-gpu-ordered-experiments/snli_increasing_variability_test.txt'
    ]
    multi_fpaths_test = [
        'scripts/experiment_data/multi-gpu-experiments/snli_inv-triangle_confidence_test.txt',
        'scripts/experiment_data/multi-gpu-experiments/snli_inv-triangle_even-scaled-confidence_test.txt',
        'scripts/experiment_data/multi-gpu-experiments/snli_inv-triangle_perplexity_test.txt',
        'scripts/experiment_data/multi-gpu-experiments/snli_inv-triangle_variability_test.txt',
        'scripts/experiment_data/multi-gpu-experiments/snli_inv-triangle_even-scaled-variability_test.txt',
        'scripts/experiment_data/multi-gpu-experiments/snli_random_baseline_test.txt',
        'scripts/experiment_data/multi-gpu-experiments/snli_triangle_confidence_test.txt',
        'scripts/experiment_data/multi-gpu-experiments/snli_triangle_even-scaled-confidence_test.txt',
        'scripts/experiment_data/multi-gpu-experiments/snli_triangle_even-scaled-variability_test.txt',
        'scripts/experiment_data/multi-gpu-experiments/snli_triangle_perplexity_test.txt',
        'scripts/experiment_data/multi-gpu-experiments/snli_triangle_variability_test.txt'        
    ]


    plot_simple_validation_losses(ordered_fpaths)
    plot_multi_validation_losses(multi_fpaths)
    plot_validation_accuracies(ordered_fpaths)
    plot_triangle_validation_accuracies(multi_fpaths)
    plot_test_accuracies(multi_fpaths_test, 'figures/snli/snli_scaled_test_accuracies.svg')
    plot_test_accuracies(ordered_fpaths_test, 'figures/snli/snli_ordered_test_accuracies.svg')
    
    plot_all_hardness_distributions()
    
if __name__ == "__main__":
    main()