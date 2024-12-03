import numpy as np
import re
import json
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

nltk.download('wordnet', quiet = True)
nltk.download('stopwords', quiet = True)
nltk.download('punkt', quiet = True)
nltk.download('averaged_perceptron_tagger', quiet = True)

stop_words = set(stopwords.words('english'))

class TextPreprocessor:

	def __init__(self):
		return
	
	# Convert text to lowercase
	def to_lower(self, text):
		return text.lower()

	# Remove URLs
	def remove_urls(self, text):
		return re.sub(r'http\S+', '', text)

	# Remove numbers
	def remove_numbers(self, text) :
		return re.sub(r'\d+', '', text)

	# Remove mentions
	def remove_mentions(self, text):
		'''
			Should be used performed before removing special characters
		'''
		return re.sub(r'@\w+', '', text)
	
	# Remove hashtags
	def remove_hashtags(self, text):
		'''
			Should be used performed before removing special characters
		'''
		return re.sub(r'#\w+', '', text)

	# Remove punctuation
	def remove_punctuation(self, text):
		return text.translate(str.maketrans('', '', string.punctuation))
	
	# Remove non-ASCII characters
	def remove_non_ASCII(self, text):
		return re.sub(r'[^\x00-\x7f]',' ', text)
	
	# Remove stopwords
	def remove_stopwords(self, text):
		text = ' '.join([word for word in text.split() if word not in stop_words])
		return text

	# Remove leading/trailing whitespaces
	def remove_whitespaces(self, text):
		return text.strip()
	
	# Lemmatization
	def lemmatize_text(self, text):
		lemmatizer = WordNetLemmatizer()
		return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

	# Stemming
	def stem_text(self, text):
		stemmer = PorterStemmer()
		return ' '.join([stemmer.stem(word) for word in text.split()])

	# Extracting the nouns
	def pos_tagging(self, text) :
		text = word_tokenize(text)
		return [token for token, pos in pos_tag(text) if pos.startswith('N')]
	
	# Removes duplicate words from the text
	def remove_duplicates(self, text):
		return ' '.join(list(set(text.split())))
	
	# Preprocess text
	def preprocess_text(self, text):
		text = self.remove_whitespaces(text)
		text = self.to_lower(text)
		text = self.remove_urls(text)
		text = self.remove_numbers(text)
		text = self.remove_mentions(text)
		text = self.remove_hashtags(text)
		text = self.remove_punctuation(text)
		text = self.remove_non_ASCII(text)
		text = self.remove_duplicates(text)
		return text