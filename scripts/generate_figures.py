import urllib.request
import numpy as np
import matplotlib.pyplot as plt

def generate_joy_division_plot(output_filename="unknown_pleasures.pdf"):
    # URL to the original digitized pulsar CP 1919 / PSR B1919+21 dataset
    url = "https://gist.githubusercontent.com/borgar/31c1e476b8e92a11d7e9/raw/0fae97dab6836ee9da3a5a260ada3edd7fa997ec/pulsar.csv"
    
    print("Attempting to fetch original pulsar data...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = np.genfromtxt(response, delimiter=',')
        print(f"Successfully loaded original data. Shape: {data.shape}")
    except Exception as e:
        print(f"Could not download data ({e}). Generating high-fidelity synthetic fallback...")
        # High-fidelity synthetic fallback matching the original 80 lines x 300 points structure
        np.random.seed(1979)
        num_lines, num_points = 80, 300
        x_test = np.linspace(0, 100, num_points)
        data = []
        for i in range(num_lines):
            window = np.exp(-((x_test - 50) / 23)**4)  # Flattens the outer edges
            line = np.random.normal(0, 0.02, num_points)
            for _ in range(np.random.randint(2, 5)):
                c = np.random.uniform(35, 65)
                w = np.random.uniform(1.5, 4.5)
                h = np.random.uniform(0.3, 1.8)
                line += h * np.exp(-((x_test - c) / w)**2)
            data.append(line * window)
        data = np.array(data)

    num_lines, num_points = data.shape
    x = np.linspace(0, 100, num_points)

    # Establish canvas: Square aspect ratio, solid black canvas
    fig, ax = plt.subplots(figsize=(9, 9), facecolor='black')
    ax.set_facecolor('black')

    # Spacing constant to vertically stagger the lines
    # Adjust this value if you want more or less overlap between lines
    spacing = 0.22 

    # Plot from top to bottom (row 0 is background, row 79 is foreground)
    # Higher z-order ensures the foreground lines properly mask the lines behind them
    for i in range(num_lines):
        offset = (num_lines - 1 - i) * spacing
        y_values = data[i] + offset
        
        # 1. Fill below the curve with solid black to mask any background traces
        ax.fill_between(x, -1, y_values, color='black', zorder=i)
        
        # 2. Draw the clean white trace over the top of the mask
        ax.plot(x, y_values, color='white', linewidth=0.9, zorder=i + 0.5)

    # Replicate the iconic framing (generous black margins surrounding the grid)
    ax.set_xlim(-25, 125)
    ax.set_ylim(-2, (num_lines * spacing) + 3)
    
    # Strip away all coordinate axis lines, ticks, and labels
    ax.axis('off')
    
    # Tight layout to eliminate unintended bounding margins
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Save cleanly as a vector PDF file
    plt.savefig(output_filename, facecolor='black', edgecolor='none', format='pdf')
    plt.close()
    print(f"Render complete! Saved vector image to '{output_filename}'.")

if __name__ == "__main__":
    from pathlib import Path
    OUT_DIR = Path('paper/figures')
    if not OUT_DIR.exists():
        OUT_DIR.mkdir(parents=True, exist_ok=True)
    generate_joy_division_plot(OUT_DIR/"joy.pdf")
