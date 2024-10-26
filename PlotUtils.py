import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

class PlotUtils:

    def __init__(self) -> None:
        pass

    # Function to plot incorrect recommendations
    def plot_recommendations(self, text, recommended_images, labels, ground_truth_image = None, top_k = 10):
        if top_k == None:
            top_k = len(recommended_images)

        len_ = top_k if ground_truth_image == None else top_k + 1
        
        images = recommended_images + [ground_truth_image]

        _, ax = plt.subplots(1, len_, figsize = (3 * (len_), 6))
        plt.suptitle(text, fontsize=16)

        for i in range(len_):
            width, height = images[i].size
            ax[i].imshow(images[i])

            if i == top_k:
                ax[i].set_title("Actual Image")
            else:
                ax[i].set_title(f"Rec. {i + 1}")
            
            # Set border color based on label (correct/incorrect)
            border_color = 'green' if labels[i] else 'red'
            
            border_width = 6
            rect = Rectangle((0, 0), width, height,linewidth = border_width, edgecolor = border_color, facecolor = 'none')
            ax[i].add_patch(rect)
            ax[i].axis('off')
        
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    plotUtils = PlotUtils()
    
    text = "Sample input text"
    top_k = 5

    # Generate random images to simulate recommendations
    recommended_images = [np.random.rand(100, 100, 3) for _ in range(top_k)]

    # Labels for recommendations: True for correct, False for incorrect
    labels = [True, False, True, False, False]
    plotUtils.plot_misclassification(text, recommended_images, labels, top_k)
