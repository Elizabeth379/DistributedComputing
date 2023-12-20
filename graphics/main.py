import glob
import os.path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme()


def plot_results(path_to_result_dir,
                 path_to_plots_dir,
                 x_axis,
                 y_axis,
                 filename_pattern):
    # Use filename_pattern for filtering files
    paths = glob.glob(os.path.join(path_to_result_dir, f"{filename_pattern}*.csv"))

    # Check if any files were found
    if not paths:
        print(f"No files found with the pattern '{filename_pattern}' in the specified directory.")
        return

    frames = [pd.read_csv(path).mean(axis=0) for path in paths]

    processor_names = {
        0: 'CPU',
        1: 'CPU and GPU',
        2: "GPU",
    }

    df = pd.concat(frames, axis=1).rename(columns=processor_names)

    print(df)

    plt.figure(figsize=(10, 6))

    sns.lineplot(data=df)

    # plt.suptitle(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.savefig(os.path.join(path_to_plots_dir, f"{filename_pattern}_plot.png"))


if __name__ == '__main__':
    plot_results(
        'C:/projects/avs/graphics',
        'C:/projects/avs/graphics/plots',
        'Размер матрицы',
        'Время (мс)',
        'results',
    )
