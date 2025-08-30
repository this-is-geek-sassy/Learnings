import matplotlib.pyplot as plt
import csv
import os

# Path to the data file
DATA_FILE = os.path.join(os.path.dirname(__file__), '../target/results/main_4_tilesize_vs_time.txt')
IMG_FILE = os.path.join(os.path.dirname(__file__), 'main_4_tilesize_vs_time.png')

tile_labels = []
kernel_times = []

with open(DATA_FILE, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        if len(row) < 3:
            continue
        label = f"{row[0].strip()}x{row[1].strip()}"
        tile_labels.append(label)
        kernel_times.append(float(row[2].strip()))

plt.figure(figsize=(8, 5))
plt.bar(tile_labels, kernel_times, color='goldenrod')
plt.xlabel('Tile Size (M x N)')
plt.ylabel('Kernel Time (microseconds)')
plt.title('Tile Size (M x N) vs Kernel Time (main_4)')
plt.tight_layout()
plt.savefig(IMG_FILE)
plt.show()
