import os
import json
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from TextPreprocessor import TextPreprocessor
import NumpyUtils

ROOT_EMBEDDINGS_FOLDER = './embeddings/'
MODEL = 'openai/clip-vit-base-patch32'
EMBEDDING_DIM = 512

class GenerateEmbeddings:
    def __init__(self, root_embeddings_path = ROOT_EMBEDDINGS_FOLDER):
        self.text_preprocessor = TextPreprocessor()
        self.model = CLIPModel.from_pretrained(MODEL)
        self.processor = CLIPProcessor.from_pretrained(MODEL)

        self.root_embeddings_path = root_embeddings_path
        return

    def generate_embeddings(self, texts, images):
        preprocessed_texts = []
        for text in texts:
            preprocessed_texts.append(self.text_preprocessor.preprocess_text(text))
        
        inputs = self.processor(preprocessed_texts, images, return_tensors = "pt", truncation = True)
        outputs = self.model(**inputs)
        return {
            'text_embs': outputs['text_embeds'],
            'image_embeds': outputs['image_embeds'],
            'logits_per_text': outputs['logits_per_text'],
            'logits_per_image': outputs['logits_per_image']
        }

    # Generate embedding for the image(s)
    def generate_image_embedding(self, images):
        inputs = self.processor(images, return_tensors = "np")
        outputs = self.model(**inputs)
        return outputs['image_embeds']

    # Generate embedding for the text(s)
    def generate_text_embedding(self, texts):
        preprocessed_texts = []
        for text in texts:
            preprocessed_texts.append(self.text_preprocessor.preprocess_text(text))
        
        inputs = self.processor(text = preprocessed_texts, return_tensors = "np")
        outputs = self.model(**inputs)
        return outputs['text_embeds']

    # Save generated embeddings dictionary to a json file
    def save_embeddings(self, embeddings, embeddings_file_name):
        json_data = json.dumps(embeddings, cls = NumpyUtils.NumpyEncoder)

        embeddings_path = os.path.join(self.root_embeddings_path, embeddings_file_name)
        with open(embeddings_path, 'w') as file:
            file.write(json_data)
        return
    
    def load_embeddings(self, embeddings_file_name):
        embeddings_path = os.path.join(self.root_embeddings_path, embeddings_file_name)
        with open(embeddings_path, 'r') as file:
            json_data = file.read()
        
        embeddings = json.loads(json_data, cls = NumpyUtils.NumpyDecoder)
        return embeddings

# if __name__ == '__main__':
#     generateEmbeddings = GenerateEmbeddings(data_path = './destination')
    
    # imgs = [Image.open('BatBall.jpeg')]
    # texts = ['Bat and Bal', 'I am not the right text']
    # outputs = generateEmbeddings.generate_embeddings(texts, images = imgs)
    
    # Note: We might have to normalize the embeddings
    # Access text embeddings using outputs['text_embeds]
    # Access image embeddings using outputs['image_embeds]

    # Save embeddings using save_embeddings
    
    # Calculate the image-text similarity score
    # logits_per_image = outputs[3]
    # probs = logits_per_image.softmax(dim = 1)
    # print(probs)
    
