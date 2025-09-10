import matplotlib.pyplot as plt
import csv
import os

# Path to the data file
DATA_FILE = os.path.join(os.path.dirname(__file__), '../target/results/main_1_2_blocksize_vs_time.txt')
IMG_FILE = os.path.join(os.path.dirname(__file__), 'main_1_2_blocksize_vs_time.png')

block_labels = []
kernel_times = []

with open(DATA_FILE, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        if len(row) < 3:
            continue
        label = f"{row[0].strip()}x{row[1].strip()}"
        block_labels.append(label)
        kernel_times.append(float(row[2].strip()))

plt.figure(figsize=(8, 5))
plt.bar(block_labels, kernel_times, color='salmon')
plt.xlabel('Block Size (X x Y)')
plt.ylabel('Kernel Time (microseconds)')
plt.title('Block Size (X x Y) vs Kernel Time')
plt.tight_layout()
plt.savefig(IMG_FILE)
plt.show()
