import matplotlib.pyplot as plt
import csv
import os

# Path to the data file
DATA_FILE = os.path.join(os.path.dirname(__file__), '../target/results/main_3_blocksize_vs_time.txt')
IMG_FILE = os.path.join(os.path.dirname(__file__), 'main_3_blocksize_vs_time.png')

tile_sizes = []
kernel_times = []

with open(DATA_FILE, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        if len(row) < 2:
            continue
        tile_sizes.append(str(row[0].strip()))
        kernel_times.append(float(row[1].strip()))

plt.figure(figsize=(8, 5))
plt.bar(tile_sizes, kernel_times, color='cornflowerblue')
plt.xlabel('Tile Size')
plt.ylabel('Kernel Time (microseconds)')
plt.title('Tile Size vs Kernel Time (main_3)')
plt.tight_layout()
plt.savefig(IMG_FILE)
plt.show()
