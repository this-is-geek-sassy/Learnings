import matplotlib.pyplot as plt
import csv
import os

# Path to the data file
DATA_FILE = os.path.join(os.path.dirname(__file__), '../target/results/main_1_1_blocksize_vs_time.txt')
IMG_FILE = os.path.join(os.path.dirname(__file__), 'main_1_1_blocksize_vs_time.png')

block_sizes = []
kernel_times = []

with open(DATA_FILE, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        if len(row) < 2:
            continue
        block_sizes.append(int(row[0].strip()))
        kernel_times.append(float(row[1].strip()))

plt.figure(figsize=(8, 5))
plt.bar([str(b) for b in block_sizes], kernel_times, color='skyblue')
plt.xlabel('Block Size')
plt.ylabel('Kernel Time (microseconds)')
plt.title('Block Size vs Kernel Time')
plt.tight_layout()
plt.savefig(IMG_FILE)
plt.show()
