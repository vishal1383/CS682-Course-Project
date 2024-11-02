import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

class PlotUtils:

    def __init__(self) -> None:
        pass

    def plot_metric(self, metric_values, k_values):
        plt.figure(figsize = (5, 4))
        bar_width = 0.4
        x_positions = np.arange(len(k_values))

        plt.bar(x_positions, metric_values, color='blue', width=bar_width)

        # Set the x-ticks to be the k values
        plt.xticks(x_positions, k_values)

        # Add labels and title
        plt.xlabel('k')
        plt.ylabel('Recall')
        plt.title('Recall@k Values')

        # Display the recall values on top of the bars
        for i, v in enumerate(metric_values):
            plt.text(i, v + 0.02, f"{v:.2f}", ha='center', va='bottom')

        # Show the plot
        plt.ylim(0, 1)  # Set the y-axis limit
        plt.show()
        return

    # Function to plot recommendations
    def plot_recommendations(self, text, recommended_images, labels, ground_truth_images = {'ids': [], 'imgs': []}, top_k = 10):
        len_ = top_k if ground_truth_images == {'ids': [], 'imgs': []} else top_k + 1

        images = recommended_images['imgs'][:top_k] + ground_truth_images['imgs']
        ids = recommended_images['ids'][:top_k] + ground_truth_images['ids']
        
        len_ground_truth_images = len(ground_truth_images['imgs'])
        labels_ = labels[:top_k] + [True] * len_ground_truth_images if len_ground_truth_images > 0 else labels[:top_k]
        
        len_ = len(images)
        assert len_ == len(labels_)

        _, ax = plt.subplots(1, len_, figsize = (3 * (len_), 6))
        plt.suptitle(text, fontsize=16)

        for i in range(len_):
            width, height = images[i].size
            ax[i].imshow(images[i])

            if i >= top_k:
                ax[i].set_title("Actual Image")
            else:
                ax[i].set_title(f"Rec. {i + 1} ({ids[i]})")
            
            # Set border color based on label (correct/incorrect)
            border_color = 'green' if labels_[i] else 'red'
            
            border_width = 6
            rect = Rectangle((0, 0), width, height,linewidth = border_width, edgecolor = border_color, facecolor = 'none')
            ax[i].add_patch(rect)
            ax[i].axis('off')
        
        plt.tight_layout()
        plt.show()

# if __name__ == '__main__':
#     plotUtils = PlotUtils()
    
#     text = "Sample input text"
#     top_k = 5

#     # Generate random images to simulate recommendations
#     recommended_images = [np.random.rand(100, 100, 3) for _ in range(top_k)]

#     # Labels for recommendations: True for correct, False for incorrect
#     labels = [True, False, True, False, False]
#     plotUtils.plot_misclassification(text, recommended_images, labels, top_k)
