import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Define the metrics
metrics = [
    "Accuracy", "Precision", "Sensitivity", "Specificity", "F-Measure",
    "MCC", "NPV", "FPR", "FNR"
]

# Data from the first table
data1 = {
    "Without preprocessing": [0.910, 0.885, 0.870, 0.950, 0.877, 0.860, 0.945, 0.050, 0.130],
    "Without feature extraction": [0.925, 0.895, 0.890, 0.960, 0.892, 0.875, 0.955, 0.040, 0.110],
    "Without feature selection": [0.940, 0.900, 0.905, 0.970, 0.902, 0.890, 0.965, 0.030, 0.095],
    "proposed model": [0.983177, 0.915886, 0.915886, 0.990654, 0.915886, 0.90654, 0.990654, 0.009346, 0.084114]
}

# Data from the second table
data2 = {
    "Without preprocessing": [0.920, 0.890, 0.880, 0.955, 0.885, 0.870, 0.950, 0.045, 0.120],
    "Without feature extraction": [0.935, 0.905, 0.900, 0.970, 0.902, 0.890, 0.965, 0.030, 0.100],
    "Without feature selection": [0.950, 0.920, 0.915, 0.980, 0.917, 0.905, 0.975, 0.020, 0.085],
    "proposed model": [0.992879, 0.964397, 0.964397, 0.996044, 0.964397, 0.960441, 0.996044, 0.003956, 0.035603]
}

# Create dataframes
df1 = pd.DataFrame(data1, index=metrics)
df2 = pd.DataFrame(data2, index=metrics)

# Plotting function
def plot_bar_chart(df, title):
    ax = df.plot(kind="bar", figsize=(14, 6))
    plt.title(title)
    plt.ylabel("Score")
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=n_methods, frameon=False)

    plt.tight_layout()
    plt.show()

# Plot both charts
plot_bar_chart(df1, "Performance Comparison - Table 1")
plot_bar_chart(df2, "Performance Comparison - Table 2")
