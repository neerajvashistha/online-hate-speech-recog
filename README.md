# An online Multilingual Hate Speech Detection System

## Setting up Environment and requirements

To get started with the project, follow the below instructions 

```
virtualenv -p python3 py3tf
source py3tf2/bin/activate

pip install git+https://github.com/neerajvashistha/indic_nlp_library
pip install git+https://github.com/neerajvashistha/ekphrasis

git clone https://github.com/neerajvashistha/online-hate-speech-recog.git

cd online-hate-speech-recog

pip install -r requirements.txt

```

Onces all the requirements are installed. Add module data and models to the project. Run the below command in `online-hate-speech-recog` directory in `py3tf` virtual environment .

```
python setup.py
```

## Project Structure

The project structure is defined as
```
-> online-hate-speech-recog/
	-> data/
		-> en/
			-> hasoc2019/format_data.ipynb
			-> hate-speech-offensive-language/format_data.ipynb
			-> hate_speech_icwsm18/format_data.ipynb
			-> ousidhoum-etal-multilingual-hate-speech-2019/format_data.ipynb
			-> semeval2018/format_data.ipynb
		-> hi/
			-> hasoc2019/format_data.ipynb
			-> Hinglish-Offensive-Text-Classification/
				-> Hinglish_Profanity_List.csv
				-> format_data.ipynb
		-> dataset_en.csv
		-> dataset_hi.csv
		-> dataset_hi_cdmx.csv
	-> model/
		-> hi-sentiment/
	-> src/
		-> utils
		-> online-application/
			-> static/
			-> templates/
			-> app.py
		-> feature_generation.py
		-> base_line_LR.py
		-> base_line_DNN.py
		-> base_line_model.ipynb
		-> base_line_CNN_LSTM.ipynb
		-> en_Bert_Based_Model.ipynb
		-> hi_Bert_Based_Model.ipynb
		-> hi_cdmx_Bert_Based_Model.ipynb
	-> setup.py
	-> requirements.txt
	-> README.md
```

