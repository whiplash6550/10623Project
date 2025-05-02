import matplotlib.pyplot as plt
import pandas as pd

# Step 1: Create the data manually
data = {
    'Model': [
        'Gemma-7B (baseline)',
        'Gemma-7B + QLoRA (CoT)',
        'Gemma-7B + QLoRA (Numeric-only)',
        'Gemma-7B (three-shot)',
        'Gemma-7B + QLoRA (one-shot)',
        'Gemma-7B + QLoRA (two-shot)',
        'Gemma-7B + QLoRA (three-shot)',
        'Gemma-7B + QLoRA (Numeric-only three-shot)'
    ],
    'Accuracy (%)': [
        16.53,
        56.10,
        14.25,
        55.04,
        55.04,
        54.28,
        54.97,
        52.24
    ],
    'Validity': [
        0.3600,
        0.6643,
        0.3417,
        0.6069,
        0.6503,
        0.6423,
        0.6474,
        0.4667
    ],
    'Redundancy': [
        0.1249,
        0.0605,
        0.1168,
        0.1079,
        0.1019,
        0.0768,
        0.0717,
        0.2698
    ]
}

# Step 2: Create a DataFrame
df = pd.DataFrame(data)

# Set a prettier style
plt.style.use('seaborn-v0_8-darkgrid')

# Step 3: Plot Accuracy
plt.figure(figsize=(12, 8))
plt.barh(df['Model'], df['Accuracy (%)'], color='skyblue', edgecolor='black')
plt.xlabel('Accuracy (%)', fontsize=17)
plt.title('Model Accuracy Comparison', fontsize=20)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('accuracy_comparison.png')
plt.close()

# Step 4: Plot Validity
plt.figure(figsize=(12, 8))
plt.barh(df['Model'], df['Validity'], color='mediumseagreen', edgecolor='black')
plt.xlabel('Validity', fontsize=14)
plt.title('Model Validity Comparison', fontsize=16)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('validity_comparison.png')
plt.close()

# Step 5: Plot Redundancy
plt.figure(figsize=(12, 8))
plt.barh(df['Model'], df['Redundancy'], color='salmon', edgecolor='black')
plt.xlabel('Redundancy', fontsize=14)
plt.title('Model Redundancy Comparison', fontsize=16)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('redundancy_comparison.png')
plt.close()

# Step 6: Scatter plot (Validity vs Redundancy)
plt.figure(figsize=(10, 8))
plt.scatter(df['Validity'], df['Redundancy'], color='purple', s=100, edgecolors='w', linewidth=1.5)
for i, txt in enumerate(df['Model']):
    plt.annotate(txt, (df['Validity'][i]+0.005, df['Redundancy'][i]+0.002), fontsize=12)
plt.xlabel('Validity', fontsize=17)
plt.ylabel('Redundancy', fontsize=17)
plt.title('Validity vs Redundancy', fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig('validity_vs_redundancy.png')
plt.close()

print("Graphs saved successfully in the current folder!")