# CS682-Course-Project
## One size does not fit all: Improving Fashion Recommendation relevance via fit-aware Neural Re-ranking
Fashion recommendation systems are crucial for enhancing user experience and boosting sales through personalized suggestions. In our project, we aim to explore methods for improving the relevance of image retrieval systems given a textual description of custom needs for fashion recommendation. We plan to train neural networks in a learning to rank framework with custom features extracted from the images and the given query, to first retrieve set of all relevant images with high recall and further enhance the precision of the top-k retrieved items through re-ranking.

#### Our Pipeline:
In our project, we first retrieve images with high recall and use re-ranking with custom image features such as bounding box, sleeve length etc. along with the embeddings of original image and query to achieve better precision as shown below:

![Project Pipeline](https://github.com/vishal1383/CS682-Course-Project/blob/main/pipeline_682.jpg)
# Installation Steps

- Create a conda environment and activate it
- Use the following commands to install relevant packages:

```
- conda install numpy
- conda install anaconda::pillow
- conda install pytorch::pytorch
- conda install conda-forge::transformers
```
