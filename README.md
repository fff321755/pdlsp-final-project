# Practical Deep Learning System Performance - Final Project
## Ensembling Visual-Language Models for Hateful Meme Detection



![](https://i.imgur.com/xMbCbqS.png)

### Project Description

Multimodal problems has been a difficult task to solve in the deep learning field. In this project, we aim to tackle the classification problem on Facebook’s Hate Meme dataset.

The definition of hateful[1] meme is as follows:

“A direct or indirect attack on people based on characteristics, including ethnicity, race, nationality, immigration status, religion, caste, sex, gender identity, sexual orientation, and disability or disease. We define attack as violent or dehumanizing (comparing people to non-human things, e.g. animals) speech, statements of inferiority, and calls for exclusion or segregation. Mocking hate crime is also considered hate speech.”

This task is difficult because the decisions are more subtle than unimodal cases, and often require real-life context and common sense. Unimodal models won't capture the semantics well.

To tackle this, we trained 4 different vision-language models, and apply different ensemble techniques (Bagging, Random Forest, voting…) to achieve better performance.


### Repository Description

The repository contains implementations on the four vision-language models, i.e. UNITER[2], OSCAR[3], ViLBERT[4], Visual-BERT[5]. It also have our implementation on several ensemble techniques. (In ensemble.ipynb)

#### Steps
1. Data preprocess and extract additional features for UNITER and OSCAR model
2. Train and inference on UNITER
3. Train and inference on OSCAR
4. Train and inference on ViLBERT
5. Train and inference on Visual-BERT
6. Ensemble and Evaluated


#### Directory Structure
The brief file structure is as follows:
```
pdlsp-final-project/
├── dev.jsonl
├── esmemble.ipynb
├── eval.ipynb
├── random_forest.joblib
├── test.jsonl
├── train.jsonl
├── uniter-oscar
│   ├── inference.ipynb
│   ├── new_req.txt
│   ├── oscar.ipynb
│   ├── train_oscar.ipynb
│   ├── train_uniter.ipynb
│   ├── training.ipynb
│   ├── uniter-inference.ipynb
│   └── vilio
├── vilbert
│   ├── expirements.ipynb
│   ├── finetune_vilbert.yaml
│   ├── infer_vilbert.yaml
│   ├── output
│   └── save
└── visual_bert
    ├── code.ipynb
    ├── finetune_visual_bert.yaml
    ├── infer_visual_bert.yaml
    ├── output
    ├── readme.md
    └── save
```


### Usage (Example commands)

#### Prerequisites


##### MMF
Install MMF from source code
```
git clone https://github.com/facebookresearch/mmf.git
cd mmf
pip install --editable .
```

Or from pypl
```
pip install --upgrade --pre mmf
```

For windows user
```
pip install -f https://download.pytorch.org/whl/torch_stable.html --editable .
```

Install Required Packages
```
cd uniter-oscar/vilio; pip install -r requirements.txt
```

##### Hateful meme dataset

from kaggle api
```
kaggle datasets download -d parthplc/facebook-hateful-meme-dataset
```

for MMF framework
```
mmf_convert_hm --zip_file <zip_file_path> --password <password> --bypass_checksum=1
```


#### train

##### UNITER
Going through the following file:
```
uniter-oscar/train_uniter.ipynb
```
The file provides end-to-end process for training UNITER. Include file download, environment setup,  feature extraction, and training.

##### OSCAR
Going through the following file:
```
uniter-oscar/train_oscar.ipynb
```
The file provides end-to-end process for training OSCAR. Include file download, environment setup,  feature extraction, and training.



##### ViLBERT
```
cd vilbert
mmf_run config=finetune_vilbert.yaml model=vilbert dataset=hateful_memes
cd ..
```

##### Visual-BERT
```
cd visual_bert
mmf_run config=finetune_visual_bert.yaml model=visual_bert dataset=hateful_memes
cd..
```
 

#### Inference

For UNITER and OSCAR, going through
```
uniter-oscar/inference.ipynb
```


##### Visual-BERT
```
cd vilbert
mmf_predict config=infer_visual_bert.yaml model=visual_bert dataset=hateful_memes run_type=test checkpoint.resume=save/visual_bert_final
cd ..
```
##### ViLBERT
```
cd visual_bert
mmf_predict config=infer_vilbert.yaml model=vilbert dataset=hateful_memes run_type=test checkpoint.resume=save/vilbert_final.pth
cd..
```


#### Demo

Run `emsemble.ipynb`

### Results & Observation
The following table illustrates the accuracy and ROC AUC score on single models.
| Model      | acc | roc_auc     |
| :---        |    :----:   |          ---: |
| UNITER      | 0.787       | 0.7895   |
| OSCAR   | 0.81        | 0.8075      |
| Visual BERT   | 0.749        | 0.698     |
| VilBERT   | 0.751       | 0.7304      |

After applying ensemble techniques, the result is as follows.
| Ensemble      | acc | roc_auc     |
| :---        |    :----:   |          ---: |
| Vote      | 0.819       | 0.8434  |
| Simple Average  |    0.799    |   0.8507   |
| Weighted Average  |   0.825     | 0.8574  |
| Random Forest  w/o OSCAR  |    0.776   | 0.8127|


Within the four single model, OSCAR outperforms the rest three models. We think that the reason is due to the additional "anchor point" offered by OSCAR in the input, which captures the semantics between image and text. This is beneficial when detect hateful memes, because the model is required to consider semantics between two domains.

Ensemble not always outperformance the best single model, but the area under ROC do increase, which means the model has better ability to distinguish between hateful or unhateful meme.

Compare our AUROC performance with the competition leader board, we would be rank 9 under 3926 teams. [Reference](https://www.drivendata.org/competitions/64/hateful-memes/leaderboard/)


### References
[1] [The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes](https://arxiv.org/pdf/2005.04790.pdf). 
[2] [The Risk of Racial Bias in Hate Speech Detection](https://aclanthology.org/P19-1163/)  
[3] [UNITER: UNiversal Image-TExt Representation Learning](https://arxiv.org/abs/1909.11740)  
[4] [Oscar: Object-Semantics Aligned Pre-training for Vision-Language Tasks](https://arxiv.org/pdf/2004.06165.pdf) 
[5] [ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks](https://arxiv.org/pdf/1908.02265.pdf)  
[6] [VISUALBERT: A SIMPLE AND PERFORMANT BASELINE FOR VISION AND LANGUAGE](https://arxiv.org/pdf/1908.03557.pdf)  
[7] [MMF, a PyTorch powered MultiModal Framework](https://mmf.sh/)  
[8] [Vilio: State-of-the-art VL models in PyTorch & PaddlePaddle](https://github.com/Muennighoff/vilio)  
