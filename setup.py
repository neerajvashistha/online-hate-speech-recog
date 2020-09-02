from inltk.inltk import setup
import nltk
import wget
import zipfile

setup('hi')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('indian')
nltk.download('stopwords')

wget.download('https://ndownloader.figshare.com/files/24546650', out='model/model.zip')
with zipfile.ZipFile('model/model.zip', 'r') as zip_ref:
	zip_ref.extractall('model/')
print('Uncompressing Completed!')
