
# Contrastively Learned Attention based Stratified PTM Predictor (CLASPP) a unified PTM prediction model



<p align="center">
  <img width="100%" src= "figures/Screenshot%20from%202025-07-11%2014-10-57.png">
</p>


CLASPP is a ESM2-150m protein lanuguage model that can pred PTM envents occuring on the substrate based 
off primary protein sequence. This is done on multiple differnt PTM types (12) as a form of multi-label
classifcation. The encoder is training on a supervised Contrastive learing task then the classifcation
head is finetunted on the multi-label classifcation. Existing PTM prediction models predominantly focus 
on either single PTM types or employ ensemble methods that combine multiple models to predict different 
PTM types. This fragmentation is largely driven by the vast imbalance in data availability across PTM 
types making it difficult to predict multiple PTM types with a single model. To address this 
limitation, we present the Contrastively Learned Attention-Based Stratified PTM Predictor (CLASPP), 
a unified PTM prediction model.







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

# How to Get Started with the Model


### Downloading this repository   

make sure [git lfs](https://git-lfs.com/) is installed 

Can not store weight files here (too big)

```   
git clone https://huggingface.co/esbglab/Claspp_forward
```   


```   
cd Claspp_data_cur
``` 



### Creating this conda environment 


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

  

We provided code to test CLASPP (see section below) 

  
:tada: you are know ready to run the code :tada: 
  

Use the code below to get started with the model.



## Model Details



<p align="center">
  <img width="100%" src= "https://github.com/gravelCompBio/Claspp_forward/blob/main/figures/Screenshot%20from%202025-08-05%2011-49-49.png">
</p>


| PTM type  | Residue trained on | Number of clusters allocated|output indexes|input label indexes (training)|
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


## Data organization and number of clusters

<p align="center">
  <img width="100%" src= "https://github.com/gravelCompBio/Claspp_forward/blob/main/figures/Screenshot%20from%202025-08-05%2011-48-48.png">
</p>


| Repo  | Link | Discription|
| ------------- | ------------- |------------------------------------------|
| GitHub  | [github version Data_cur](https://github.com/gravelCompBio/Claspp_data_cur)  | This verstion contains code but but no data. It needs you to run the code to generate all the helper-files (will take some time run this code)|
| GitHub  | [github version Forward](https://github.com/gravelCompBio/Claspp_forward)  | This verstion contains code but NOT any weights (file too big for github)|
| Huggingface | [huggingface version Forward](https://huggingface.co/esbglab/Claspp_forward)  | This verstion contains code and training weights |
| Zenodo | [zenodo version training_data](https://zenodo.org/records/16739128?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjY4MDBkYjUwLTVlZjktNDZlNy04MjZjLTgzZjA0NjZiYmZlYyIsImRhdGEiOnt9LCJyYW5kb20iOiJlMThhZGNlMWUxN2EzNjYxNzllYjg5MWRiZjhiMWYxNSJ9.7Os5ZzQLT3TJu3Clv1Sxvh8oVtFTxxoeYLACFgKwZRjCApQdfQO2-AvctQ-eIIEojKTBGHLCcHlMTDG38AKn8A)  | zenodo version of training/testing/validation data|
| webtool | [website version of webtool](https://github.com/gravelCompBio/Claspp_data_cur/tree/main)  | webtool hosted on a server|

- **Repository:** [More Information Needed]
- **Paper [optional]:** [More Information Needed]
- **Demo [optional]:** [More Information Needed]








## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use
```
Usage: python3 claspp_forward.py [OPTION]... --input INPUT [FASTA_FILE or TXT_FILE]...
predict PTM events on peptides or full sequences

Example 1: python3 claspp_forward.py -B 100 -S 0 -i random.txt
Example 2: python3 claspp_forward.py -B 50 -S 1 -i random.fasta

FASTA_FILE contain protein sequences in proper fasta or a2m format
TXT_FILE cointain protien peptides 21 in length with the center
residue being the PTM modification site


Pattern selection and interpretation:
  -B, --batch_size          (int) that describes how many predictions
                            can be predicted at a time on the GPU
                            (reduce if you get run out of GPU space)

  -S  --scrape_fasta        (int) should be a 1 or a 0 
                            1 = read a fasta and scrape posible 21 peptides
                            that can be modified by a PTM 
                            0 = read a txt file that has the 21mer already 
                            sperated and all peptides should be sperated by 
                            a '\\n' (can be faster) than fasta option
  
  -h  --help                your reading it right now

  -i  --input               location of the input fasta or txt

  -o  --output              location of the output csv

```




- **Developed by:** Major author for most code Nathan Gravel. Finetuning code inspried by Zhongliang Zhou,
- Contrastive learing code inspried by Ruili Fang, Codebase testing and verstion controle by Austin Downes,
- Webtool dev Saber Soleymani
- **Funded by [optional]:** [NIH]
- **Shared by [optional]:** [More Information Neede]
- **Model type:** [Text classication]
- **Language(s) (NLP):** [Protein Sequence]
- **License:** [MIT]
- **Finetuned from model [optional]:** [ESM-2 150M]

[More Information Needed]
