import matplotlib.pyplot as plt
import math

def plot_data(probabilities, figure_name, notShowX=False):
  num_qbits = int(math.log2(len(probabilities)))
  bitstrings = [format(i, f'0{num_qbits}b') for i in range(len(probabilities))]

  # Plotting the probabilities
  plt.figure(figsize=(10, 8))
  plt.bar(bitstrings, probabilities)
  plt.ylabel('Probability')
  plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
  if notShowX: plt.xticks([])
  plt.tight_layout()  # Adjust layout to prevent clipping of tick-labels
  plt.savefig(f'./{figure_name}')