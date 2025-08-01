import numpy as np
import matplotlib.pyplot as plt
 
# Load the data from CSV
data = np.loadtxt('final_intensity_LP71 3.csv', delimiter=',')
 
# Plot the data
plt.figure(figsize=(8, 6))
plt.pcolormesh(data, shading='auto', cmap='jet')
plt.colorbar(label='Intensity')
plt.title('Imported Field Intensity from CSV')
plt.xlabel('X pixels')
plt.ylabel('Y pixels')
plt.tight_layout()
plt.show()