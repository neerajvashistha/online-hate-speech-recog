# An online Multilingual Hate Speech Detection System

## Setting up Requirements

To get started with the project, follow the below instructions 

```
virtualenv -p python3 py3tf
source py3tf2/bin/activate

pip install git+https://github.com/neerajvashistha/indic_nlp_library
pip install git+https://github.com/neerajvashistha/ekphrasis

pip install tensorflow-gpu==1.13.1
pip install torch==1.3.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install inltk

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
		-> utils/
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

All the important files are mentioned above. We describe the information about each file below.


- The `format_data.ipynb` files present in `data/en` and `data/hi` are responsible for converting original categories into homogeneous set of classes of Hate, Abusive or neither. They are also responsible for curating some of the text from Twitter API.
- `Hinglish_Profanity_List.csv` was originally created by P Mathur et.al is now updated with more words and scripted devanagari hindi words.
- `model/hi-sentiment` this directory contains our implementation of Sentiment Analysis in Hindi language using transfer learning technique, built in Fastai. `model` directory also contains, BERT CNN_LSTM and Logistic regression models.  
- `src/utils` utility functions and modules on hindi tokentisation and cleaning 'byte encoded' emoji from text.
- `static`, `templates` and `app.py` are responsible for online application, providing web interface for live chat room environment, utilises models and generates `new_train.csv` 
- The `feature_generation.py` is the main feature generation process used for pre-processing english and hindi tweets both for Logistic regression and CNN LSTM models. It also serves as the file for generating feature vetor (TFIDF and POS vectors) for Logistic regression
- `base_line_LR.py` file contains Logistic regression model, with functionality to train, test, predict and generate classification and confusion matrix 
- `base_line_DNN.py`, this file contains ProcessData class and CNN_LSTM class. The ProcessData class is reposibile of converting tweet samples into word embedding sequences. The CNN_LSTM class contains the network and training and prediction functionality modules.
- `base_line_model.ipynb` illustrates the logistic regression model while `base_line_CNN_LSTM.ipynb` illustrates the CNN LSTM model. In order to run `base_line_CNN_LSTM.ipynb` we require a GPU.
- `en_Bert_Based_Model.ipynb`, `hi_Bert_Based_Model.ipynb` and `hi_cdmx_Bert_Based_Model.ipynb` contains the code for running BERT based model. This has been extensively tested on Google Colab TPU.

### Executing Models and evaluating Performace

In order to run the models, execute,  

```
(py3tf)$ cd online-hate-speech-recog/src
(py3tf)$ jupyter notebook
```

- `base_line_model.ipynb` requires CPU
- `base_line_CNN_LSTM.ipynb` requires GPU and 
- `xxx_Bert_Based_Model.ipynb` requires TPU.

In order to run the online application, please make sure, port 8893 is open.
```
(py3tf)$ cd online-hate-speech-recog/src
(py3tf)$ python app.py
```
