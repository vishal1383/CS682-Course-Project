import json
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from TextPreprocessor import TextPreprocessor
import NumpyUtils
ROOT_EMBEDDINGS_FOLDER = './Embeddings/'



class GenerateEmbeddings:
    def __init__(self, data_path, root_embeddings_path = ROOT_EMBEDDINGS_FOLDER):
        self.data_path = data_path
        self.text_preprocessor = TextPreprocessor()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        self.root_embeddings_path = root_embeddings_path

    def generate_embeddings(self, texts, images):
        preprocessed_texts = []
        for text in texts:
            preprocessed_texts.append(self.text_preprocessor.preprocess_text(text))
        
        inputs = self.processor(preprocessed_texts, images, return_tensors = "pt", truncation = True)
        outputs = self.model(**inputs)
        return [outputs['text_embeds'], outputs['image_embeds'], outputs.logits_per_image]

    # Generate embedding for the image data
    def generate_image_embedding(self, img):
        # print(img)
        inputs = self.processor(images = Image.open(img), return_tensors = "np")
        outputs = self.model(**inputs)
        return outputs['image_embeds']

    # Generate embedding for the text data
    def generate_text_embedding(self, text):
        preprocessed_texts = []
        for text in texts:
            preprocessed_texts.append(self.text_preprocessor.preprocess_text(text))
        
        inputs = self.processor(text = preprocessed_texts, images = None, return_tensors = "np")
        outputs = self.model(**inputs)
        return outputs['text_embeds']

    # Save generated embeddings dictionary to a json file
    def save_embeddings(self, embeddings, embeddings_path):
        json_data = json.dumps(embeddings, cls = NumpyUtils.NumpyEncoder)
        with open(embeddings_path, 'w') as file:
            file.write(json_data)
    
    def load_embeddings(embeddings_path):
        with open(embeddings_path, 'r') as file:
            json_data = file.read()
        embeddings = json.loads(json_data, cls=NumpyUtils.NumpyDecoder)
        return embeddings

if __name__ == '__main__':
    generateEmbeddings = GenerateEmbeddings(data_path ='./destination')
    imgs = [Image.open('BatBall.jpeg')]
    texts = ["Bat and Ball", "I am not the right text"]
    outputs = generateEmbeddings.generate_embeddings(texts, images = imgs)
    
    # Note: We might have to normalize the embeddings
    # Access text embeddings using outputs['text_embeds]
    # Access image embeddings using outputs['image_embeds]

    # Save embeddings using save_embeddings
    
    # Calculate the image-text similarity score
    logits_per_image = outputs[2]
    probs = logits_per_image.softmax(dim = 1)
    generateEmbeddings.save_embeddings(outputs[0], ROOT_EMBEDDINGS_FOLDER + 'sample')
    print(probs)
    
