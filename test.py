import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw, ImageFont
import numpy as np




# Function to plot misclassification with colored borders for each recommendation
def plot_misclassification(text, recommended_images, labels, top_k):
    # Convert text to image
    text_img = text_to_image(text)

    fig, ax = plt.subplots(1, top_k + 1, figsize=(3 * (top_k + 1), 6))

    # Display the input text as an image
    ax[0].imshow(text_img)
    ax[0].set_title("Query")
    ax[0].axis('off')

    # Display each recommended image with a colored border
    for i in range(top_k):
        ax[i + 1].imshow(recommended_images[i])
        ax[i + 1].set_title(f"Recommendation {i + 1}")
        
        # Set border color based on label (correct/incorrect)
        border_color = 'green' if labels[i] else 'red'
        
        # Add border to indicate correct (green) or incorrect (red)
        ax[i + 1].add_patch(Rectangle((0, 0), recommended_images[i].shape[1], recommended_images[i].shape[0], 
                                      linewidth=6, edgecolor=border_color, facecolor='none'))
        ax[i + 1].axis('off')

    plt.tight_layout()
    plt.show()

# Example Usage
text = "Sample input text"
top_k = 5

# Generate random images to simulate recommendations
recommended_images = [np.random.rand(100, 100, 3) for _ in range(top_k)]

# Labels for recommendations: True for correct, False for incorrect
labels = [True, False, True, False, False]

# Call the function to plot
plot_misclassification(text, recommended_images, labels, top_k)