import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# seaborn 스타일 적용
sns.set_style("whitegrid")

def plot_histogram(data: pd.DataFrame, numeric_columns: list[str], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    for col in numeric_columns:
        plt.figure(figsize=(10, 6), dpi=150)
        sns.histplot(data[col].dropna(), bins=50, kde=True, color='skyblue', edgecolor='black')

        plt.title(f'Histogram of {col}', fontsize=16)
        plt.xlabel(col, fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        save_path = output_dir / f'histogram_{col}.png'
        plt.savefig(save_path)
        plt.close()


if __name__ == '__main__':
    data = pd.read_csv("all_data.csv")

    exclude_columns = ['file_path', 'file_name']
    numeric_columns = [col for col in data.columns if col not in exclude_columns]

    stats = data.describe()
    stats.to_excel("statistics.xlsx")

    output_dir = Path("histograms")
    plot_histogram(data, numeric_columns, output_dir)
