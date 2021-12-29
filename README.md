### Spellchecker using T5 transformer

T5 Base is finetuned with data from simple wikipedia for building a simple spellchecker. 

### Environment set up
```
python3 -m virtualenv env
source env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Downloading data
To download simple data dump and extract from simple wikipedia, use the following commands

```
source download_wiki_dump.sh simple
source extract_and_clean_wiki_dump.sh data/simplewiki-latest-pages-articles.xml.bz2
```
The dump is downloaded and extracted file in the .txt format in the ```data``` directory.

### Preprocessing data
To preprocess the data, run the command below. ```processed_text.txt``` gets written in ```data``` folder.  
```
python data_preprocess.py
```
### Train and valid prep
```data_prep_train_valid.ipynb``` has code which prepares the correct and corrupted sentences for train and valid csv/

### Finetuning T5 model
```finetuning.py``` has the code for finetuning T5 on the dataset.
To run finetuning.py, 
```True
python finetuning_t5.py --output model_dec28 --overwrite true --do_train True --do_eval True
```

### REST - API
The spellchecker model is deployed as REST - API - API live at ```0.0.0.0:8000``` and endpoint : ```predict```
```
python api.py
```
#### cURL Request 

```
curl -X POST http://0.0.0.0:8000/predict -H 'Content-Type: application/json' -d '{ "text": "chrims is celibrated on decmber 25 evry ear" }'
```

#### Output 
```
{"result":"christmas is celebrated on december 25 every year"}
```
### Gradio 
Uncomment the lines in ```gradio_predict.py``` and run 
```
python gradio_predict.py
```
You will get the local URL and public URL for the gradio app. Click on the links and you will be able to use the Gradio app. 

![gradio output image](/img/gradio.png)

