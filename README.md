---
license: mit
language:
- en
base_model:
- andrewdalpino/ESM2-150M-Protein-Molecular-Function
pipeline_tag: text-classification
tags:
- biology
- protein
---
# Contrastively Learned Attention based Stratified PTM Predictor (CLASPP) a unified PTM prediction model

<!-- Provide a quick summary of what the model is/does. -->


CLASPP is a ESM2-150m protein lanuguage model that can pred PTM envents occuring on the substrate based 
off primary protein sequence. This is done on multiple differnt PTM types (12) as a form of multi-label
classifcation. The encoder is training on a supervised Contrastive learing task then the classifcation
head is finetunted on the multi-label classifcation. 

Post-Translational Modifications (PTMs) are a fundamental mechanism for regulating cellular functions and 
increasing the functional diversity of the proteome. Despite the identification of hundreds of unique PTMs 
through mass-spectrometry (MS) studies, accurately predicting many PTM types based on sequence data alone 
remains a significant challenge. 

Existing PTM prediction models predominantly focus on either single PTM types or employ ensemble methods 
that combine multiple models to predict different PTM types. This fragmentation is largely driven by the 
vast imbalance in data availability across PTM types making it difficult to predict multiple PTM types 
with a single model. To address this limitation, we present the Contrastively Learned Attention-Based 
Stratified PTM Predictor (CLASPP), a unified PTM prediction model.


<p align="center">
  <img width="100%" src= "figures/Screenshot%20from%202025-07-11%2014-10-57.png">
</p>







## Quick overview of the dependencies 

![Python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)
![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Numpy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white) 

### From conda:    

![python=3.9.23](https://img.shields.io/badge/Python-3.9.23-green)  


### From pip:  

![numpy=2.0.2](https://img.shields.io/badge/numpy-2.0.2-blue) ![transformers=4.53.2](https://img.shields.io/badge/transformers-4.53.2-blue) ![datasets=4.0.0](https://img.shields.io/badge/datasets-4.0.0-blue) ![torch=2.7.1](https://img.shields.io/badge/torch-2.7.1-blue)      


### For torch/PyTorch 

Make sure you go to this website [pytorch](https://pytorch.org/get-started/locally/) 

Follow along with its recommendation  

Installing torch can be the most complex part  

  



## Model Details



<p align="center">
  <img width="100%" src= "figures/Screenshot%20from%202025-07-11%2014-19-21.png">
</p>

| PTM type  | Residue trained on | Number of clusters allocated|output indexes|input indexes (training)|
| -------------------- | ------------- |--------------------------|------------|-------------|
| ST_Phosphorylation | S,T | 5 | 0 or 1 | 0-4 |
| Y_Phosphorylation | Y | 1 | 3 | 25 |
| K_Ubiquitination | K | 20 | 2 | 5-24 |
| K_Acetylation | K | 10 | 4 | 26-35 |
| AM_Acetylation | A,M | 1 | 13 or 14 | 49 |
| N_N-linked-Glycosylation | N | 1 | 5 | 36 |
| ST_O-linked-Glycosylation | S,T | 5 | 6 or 7 | 37-41 |
| RK_Methylation | RK | 4 | 8 or 9 | 42-45 |
| K_Sumoylation | K | 1 | 10 | 46 |
| K_Malonylation | K | 1 | 11 | 53 |
| M_Sulfoxidation | M | 1 | 12 | 48 |
| C_Glutathionylation | C | 1 | 15 | 50 |
| C_S-palmitoylation | C | 1 | 16 | 51 |
| PK_Hydroxylation | P,K | 1 | 17 or 18 | 52 |
|negitve| all res | N/A | 19 | 53|



### Model Sources [optional]



| Repo  | Link | Discription|
| ------------- | ------------- |------------------------------------------|
| GitHub  | [github version Data_cur](https://github.com/gravelCompBio/Claspp_data_cur/tree/main)  | This verstion contains code but needs the to run the code to generate all the helper-files (will take some time run this code)|
| Zenodo  | [zenodo version Data_cur](https://github.com/gravelCompBio/Claspp_data_cur/tree/main) | This version contains code and helper files already genrated. mostly for proof of concept and seeing the all the data intermeidate states |
| GitHub  | [github version Forward](https://github.com/gravelCompBio/Claspp_data_cur/tree/main)  | This verstion contains code but NOT any weights (file to big)|
| Huggingface | [huggingface version Forward](https://huggingface.co/gravelcompbio/Claspp)  | This verstion contains code and training weights |
| Zenodo | [zenodo version training_data](https://github.com/gravelCompBio/Claspp_data_cur/tree/main)  | zenodo version of training/testing/validation data|
| Hugingface | [huggingface version training_data](https://github.com/gravelCompBio/Claspp_data_cur/tree/main)  | huggingface version of training/testing/validation data|
| Hugingface  | [gradio lab webtool](https://github.com/gravelCompBio/Claspp_data_cur/tree/main)  | webtool hosted on gradio lab (huggingface)|
| webtool | [website version of webtool](https://github.com/gravelCompBio/Claspp_data_cur/tree/main)  | wettool hosted on a server|

- **Repository:** [More Information Needed]
- **Paper [optional]:** [More Information Needed]
- **Demo [optional]:** [More Information Needed]



## How to Get Started with the Model


### Downloading this repository   

make sure [git lfs](https://git-lfs.com/) is install 

```   
git clone https://huggingface.co/gravelcompbio/Claspp
```   


```   
cd Claspp_data_cur
``` 



### Creating this conda environment 
(yml file is include but torch sometimes makes it not useable depending on your nvidia driver)

Just type these lines of code into the terminal after you download this repository (this assumes you have [anaconda](https://www.anaconda.com/) already installed) 

```   
conda create -n claspp_forward python=3.9.23 
``` 

```   
conda deactivate 
``` 

```   
conda activate claspp_forward
``` 

```   
pip3 install numpy==2.0.2
```

```   
pip3 install transformers==4.53.2
```

```   
pip3 install datasets==4.0.0
```


### **For torch you will have to download to the torch's specification if you want gpu acceleration from this website ** [pytroch](https://pytorch.org/get-started/locally/) ** 

  

```   
pip3 install torch torchvision torchaudio 
``` 

  

### the terminal line above might look different for you  

  

We provided code to test Phosformer-ST (see section below) 

  
:tada: you are know ready to run the code :tada: 
  

Use the code below to get started with the model.



## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

[More Information Needed]


### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. More information needed for further recommendations.





## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

[More Information Needed]

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing [optional]

[More Information Needed]


#### Training Hyperparameters

- **Training regime:** [More Information Needed] <!--fp32, fp16 mixed precision, bf16 mixed precision, bf16 non-mixed precision, fp16 non-mixed precision, fp8 mixed precision -->


 

## Hardware

[More Information Needed]



## Citation [optional]

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

[More Information Needed]

**APA:**

[More Information Needed]






- **Developed by:** [Nathan Gravel]
- **Funded by [optional]:** [NIH]
- **Shared by [optional]:** [More Information Neede]
- **Model type:** [Text classication]
- **Language(s) (NLP):** [Protein Sequence]
- **License:** [MIT]
- **Finetuned from model [optional]:** [ESM-2 150M]

[More Information Needed]
