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


