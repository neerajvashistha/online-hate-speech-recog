import nltk
from hate_speech import HateSpeech

nltk.download('punkt')
nltk.download('stopwords')

model = HateSpeech(training=True)

