import os
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn

from transformers import DataCollatorWithPadding
from transformers import EsmTokenizer
from datasets import (
    load_dataset,
    Dataset,
)

from modeling_esm import EsmForSequenceClassificationCustomWidehead


print("intilizing checkpoint --might take a few min if this is the first time--")
tokenizer = EsmTokenizer.from_pretrained("finalCheckpoint_25_05_11/")
model = EsmForSequenceClassificationCustomWidehead.from_pretrained("finalCheckpoint_25_05_11/", num_labels=54).cuda()
print("finished downloading")


###############################################################################
#helper code to make the model run smooth
###############################################################################
    # labs=['ST-Phosphorylation_nc0_tot5', 
    #       'ST-Phosphorylation_nc1_tot5', 
    #       'ST-Phosphorylation_nc2_tot5', 
    #       'ST-Phosphorylation_nc3_tot5', 
    #       'ST-Phosphorylation_nc4_tot5', 
    #       'K-Ubiquitination_nc0_tot20', 
    #       'K-Ubiquitination_nc1_tot20', 
    #       'K-Ubiquitination_nc2_tot20', 
    #       'K-Ubiquitination_nc3_tot20', 
    #       'K-Ubiquitination_nc4_tot20', 
    #       'K-Ubiquitination_nc5_tot20', 
    #       'K-Ubiquitination_nc6_tot20', 
    #       'K-Ubiquitination_nc7_tot20', 
    #       'K-Ubiquitination_nc8_tot20', 
    #       'K-Ubiquitination_nc9_tot20', 
    #       'K-Ubiquitination_nc10_tot20', 
    #       'K-Ubiquitination_nc11_tot20', 
    #       'K-Ubiquitination_nc12_tot20', 
    #       'K-Ubiquitination_nc13_tot20', 
    #       'K-Ubiquitination_nc14_tot20', 
    #       'K-Ubiquitination_nc15_tot20', 
    #       'K-Ubiquitination_nc16_tot20', 
    #       'K-Ubiquitination_nc17_tot20', 
    #       'K-Ubiquitination_nc18_tot20', 
    #       'K-Ubiquitination_nc19_tot20', 
    #       'Y-Phosphorylation_nc0_tot1', 
    #       'K-Acetylation_nc0_tot10', 
    #       'K-Acetylation_nc1_tot10', 
    #       'K-Acetylation_nc2_tot10', 
    #       'K-Acetylation_nc3_tot10', 
    #       'K-Acetylation_nc4_tot10', 
    #       'K-Acetylation_nc5_tot10', 
    #       'K-Acetylation_nc6_tot10', 
    #       'K-Acetylation_nc7_tot10', 
    #       'K-Acetylation_nc8_tot10', 
    #       'K-Acetylation_nc9_tot10', 
    #       'N-N-linked-Glycosylation_nc0_tot1', 
    #       'ST-O-linked-Glycosylation_nc0_tot5', 
    #       'ST-O-linked-Glycosylation_nc1_tot5', 
    #       'ST-O-linked-Glycosylation_nc2_tot5', 
    #       'ST-O-linked-Glycosylation_nc3_tot5', 
    #       'ST-O-linked-Glycosylation_nc4_tot5', 
    #       'RK-Methylation_nc0_tot4', 
    #       'RK-Methylation_nc1_tot4', 
    #       'RK-Methylation_nc2_tot4', 
    #       'RK-Methylation_nc3_tot4', 
    #       'K-Sumoylation_nc0_tot1', 
    #       'K-Malonylation_nc0_tot1', 
    #       'M-Sulfoxidation_nc0_tot1', 
    #       'AM-Acetylation_nc0_tot1', 
    #       'C-Glutathionylation_nc0_tot1', 
    #       'C-S-palmitoylation_nc0_tot1', 
    #       'PK-Hydroxylation_nc0_tot1', 
    #       'NegLab']

labsoi=set()
lab2map={}
labsoi.add("S_Phosphorylation")
lab2map["S_Phosphorylation"]=0
labsoi.add("T_Phosphorylation")
lab2map["T_Phosphorylation"]=1
labsoi.add("Y_Phosphorylation")
lab2map["Y_Phosphorylation"]=3
labsoi.add("A_Acetylation")
lab2map["A_Acetylation"]=13
labsoi.add("M_Acetylation")
lab2map["M_Acetylation"]=14
labsoi.add("K_Acetylation")
lab2map["K_Acetylation"]=4
labsoi.add("K_Ubiquitination")
lab2map["K_Ubiquitination"]=2
labsoi.add("S_O-linked-Glycosylation")
lab2map["S_O-linked-Glycosylation"]=6
labsoi.add("T_O-linked-Glycosylation")
lab2map["T_O-linked-Glycosylation"]=7
labsoi.add("N_N-linked-Glycosylation")
lab2map["N_N-linked-Glycosylation"]=5
labsoi.add("K_Methylation")
lab2map["K_Methylation"]=9
labsoi.add("R_Methylation")
lab2map["R_Methylation"]=8
labsoi.add("K_Malonylation")
lab2map["K_Malonylation"]=11
labsoi.add("K_Sumoylation")
lab2map["K_Sumoylation"]=10
labsoi.add("C_Glutathionylation")
lab2map["C_Glutathionylation"]=15
labsoi.add("P_Hydroxylation")
lab2map["P_Hydroxylation"]=17
labsoi.add("K_Hydroxylation")
lab2map["K_Hydroxylation"]=18
labsoi.add("C_S-palmitoylation")
lab2map["C_S-palmitoylation"]=16
lab2map['M_Sulfoxidation']=12
pos2lab={}
for lab in lab2map.keys():
    pos=lab2map[lab]
    pos2lab[pos]=lab
# labsoi.add("K-Succinylation")
# lab2map["K-Succinylation"]=14


def preprocess_function(examples):
    toks={}
    toks['input_ids']=[]
    toks['attention_mask']=[]
    
    for info in examples["pep"]:
        info=info.replace(".", "<mask>")
        t=tokenizer(info.replace("-", "<pad>"))
        toks['input_ids'].append(t['input_ids'])
        toks['attention_mask'].append(t['attention_mask'])
    
    
    return toks


def getlab(elab,res):
    output=np.zeros((20))
    if res=='S':
        output[0]=max(elab[:5])
        output[1]=0
    elif res=='T':
        output[0]=0
        output[1]=max(elab[:5])
    else:
         output[0]=0
         output[1]=0
    #print(labs[:5])['ST-Phosphorylation_nc0_tot5', 'ST-Phosphorylation_nc1_tot5', 'ST-Phosphorylation_nc2_tot5', 'ST-Phosphorylation_nc3_tot5', 'ST-Phosphorylation_nc4_tot5']
    output[2]=max(elab[5:25])
    #print(labs[5:25])['K-Ubiquitination_nc0_tot20', 'K-Ubiquitination_nc1_tot20', 'K-Ubiquitination_nc2_tot20', 'K-Ubiquitination_nc3_tot20', 'K-Ubiquitination_nc4_tot20', 'K-Ubiquitination_nc5_tot20', 'K-Ubiquitination_nc6_tot20', 'K-Ubiquitination_nc7_tot20', 'K-Ubiquitination_nc8_tot20', 'K-Ubiquitination_nc9_tot20', 'K-Ubiquitination_nc10_tot20', 'K-Ubiquitination_nc11_tot20', 'K-Ubiquitination_nc12_tot20', 'K-Ubiquitination_nc13_tot20', 'K-Ubiquitination_nc14_tot20', 'K-Ubiquitination_nc15_tot20', 'K-Ubiquitination_nc16_tot20', 'K-Ubiquitination_nc17_tot20', 'K-Ubiquitination_nc18_tot20', 'K-Ubiquitination_nc19_tot20'] 
    output[3]=max(elab[25:26])
    #print(labs[25:30])['Y-Phosphorylation_nc0_tot5', 'Y-Phosphorylation_nc1_tot5', 'Y-Phosphorylation_nc2_tot5', 'Y-Phosphorylation_nc3_tot5', 'Y-Phosphorylation_nc4_tot5']
    output[4]=max(elab[26:36])
    #print(labs[30:40])['K-Acetylation_nc0_tot10', 'K-Acetylation_nc1_tot10', 'K-Acetylation_nc2_tot10', 'K-Acetylation_nc3_tot10', 'K-Acetylation_nc4_tot10', 'K-Acetylation_nc5_tot10', 'K-Acetylation_nc6_tot10', 'K-Acetylation_nc7_tot10', 'K-Acetylation_nc8_tot10', 'K-Acetylation_nc9_tot10']
    output[5]=max(elab[36:37])
    #print(labs[40:41])['N-N-linked-Glycosylation_nc0_tot1']
    if res=='S':
        output[6]=max(elab[37:42])
        output[7]=0
    elif res=='T':
        output[6]=0
        output[7]=max(elab[37:42])
    else:
         output[6]=0
         output[7]=0
    #print(labs[41:46])['ST-O-linked-Glycosylation_nc0_tot5', 'ST-O-linked-Glycosylation_nc1_tot5', 'ST-O-linked-Glycosylation_nc2_tot5', 'ST-O-linked-Glycosylation_nc3_tot5', 'ST-O-linked-Glycosylation_nc4_tot5']
    if res=="R":
        output[8]=max(elab[42:46])
        output[9]=0
    elif res=="K":
        output[8]=0
        output[9]=max(elab[42:46])
    else:
        output[8]=0
        output[9]=0
    #print(labs[46:50])['RK-Methylation_nc0_tot4', 'RK-Methylation_nc1_tot4', 'RK-Methylation_nc2_tot4', 'RK-Methylation_nc3_tot4']
    output[10]=max(elab[46:47])
    #print(labs[50:52])['K-Sumoylation_nc0_tot2', 'K-Sumoylation_nc1_tot2']
    output[11]=max(elab[47:48])
        #'K-Malonylation_nc0_tot1'
    output[12]=max(elab[48:49])
        #"M-Sulfoxidation_nc0_tot1'
    if res=="A":
        output[13]=max(elab[49:50])
        output[14]=0
    elif res=="M":
        output[13]=0
        output[14]=max(elab[49:50])
    else:
        output[13]=0
        output[14]=0
    #print(elab[50:51])
    output[15]=max(elab[50:51])
    #print(labs[57:58])['C-Glutathionylation_nc0_tot1']
    output[16]=max(elab[51:52])
    #print(labs[58:59])['C-S-palmitoylation_nc0_tot1']
    if res=="P":
        output[17]=max(elab[52:53])
        output[18]=0
    elif res=="K":
        output[17]=0
        output[18]=max(elab[52:53])
    else:
        output[17]=0
        output[18]=0
    #print(labs[52:54])['K-Malonylation_nc0_tot2', 'K-Malonylation_nc1_tot2']
    output[19]=max(elab[53:54])
    return(output)
    #print(labs[59:60])['NegLab']




###############################################################################
#prediction code
###############################################################################


def predict(input_batches):
    sig=nn.Sigmoid()
    outputpreds=[]
    r='\r'
    for i,batches in enumerate(input_batches):
        print(f"{i} / {len(input_batches)} batches done",end=r)
        # tok_input_ids=tokenizer(batches)['input_ids']
        # tensor_input_ids=torch.tensor(tok_input_ids)
        # print(tensor_input_ids)
        # print(torch.tensor([tokenizer(batches)['input_ids']]).cuda().shape)
        # print(torch.tensor([tokenizer(batches)['attention_mask']]).cuda()["logits"][0].shape)
        #print(torch.tensor([tokenizer(batches)['input_ids']]).cuda().squeeze().shape)
        # print(tokenizer(batches)['input_ids'])
        # print(torch.tensor([tokenizer(batches)['input_ids']]).squeeze().cuda())
        pred=(sig(model(torch.tensor([tokenizer(batches)['input_ids']]).squeeze().cuda(),torch.tensor([tokenizer(batches)['attention_mask']]).squeeze().cuda())["logits"]).tolist())
        #print(len(pred[0]))
        for p in pred:
            # print(p)
            outputpreds.append(p)
    return outputpreds


def write_output(pred,listofpeps,file_output):
    hf=open(f"{file_output}",'w+')
    n="\n"
    writethisline="pep"
    for i in range(len(labsoi)):
        writethisline+=','+pos2lab[i]
    hf.write(writethisline+n)
    for p,ip in zip(pred,listofpeps):
        writethisline=f"{ip}"
        r=ip[10]
        #print(p)
        easyreadlab=getlab(p,r)
        for sp in easyreadlab:
            writethisline+=f",{sp}"
        
        writethisline=writethisline[:]+n
        hf.write(writethisline)
    hf.close()
        

DOC_HELP='''
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


Report bugs on the github: https://github.com/gravelCompBio/Claspp_forward


'''
WARNING_MESSAGE="""
        #################################
        PLEASE READ HELP MESSAGE TO ENSURE
        YOU KNOW HOW TO FORMAT/USE THE
        MODEL 
        #################################
              """




def main():
    batch_size=50
    scrape=0
    file_output="output_predictions.csv"
    input_file="N/A"
    for i in range(len(sys.argv)-1):
        if sys.argv[i]=='--scrape_fasta' or sys.argv[i]=='-S':
            scrape = int(sys.argv[i+1])
        if sys.argv[i]=='--batch_size' or sys.argv[i]=='-B':
             batch_size = int(sys.argv[i+1])
        if sys.argv[i]=='--input' or sys.argv[i]=='-i':
             input_file = sys.argv[i+1]
        if sys.argv[i]=='--output' or sys.argv[i]=='-o':
             file_output = sys.argv[i+1]
        if sys.argv[i]=='-h' or sys.argv[i]=='--h' or sys.argv[i]=='-help' or sys.argv[i]=='--help' :
            print(DOC_HELP)
    if input_file=='N/A':
        print(WARNING_MESSAGE)
        print(DOC_HELP)
        return
    
    if scrape==0:
        #todo make readerfuc
        listofpeps=[]
        rf=open(input_file,"r")
        lines=rf.readlines()
        for line in lines:
            pep=line[:-1]
            listofpeps.append(pep)
            


    else:
        #todo make readerfuc
        listofpeps=[]
        acc2seq={}
        #seq2acc={}
        rf=open(input_file,"r")
        lines=rf.readlines()
        seq=""
        acc=""
        for line in lines:
            if line[0]=='>':
                if seq!='':
                    acc2seq[acc]=seq
                    #seq2acc[seq]=acc
                    seq=""
                acc=line[1:-1]
            else:
                seq+=line.replace('\n','')
        acc2seq[acc]=seq
        #seq2acc[seq]=acc 
        for acc in acc2seq.keys():
            seq=acc2seq[acc]
            paddedseq='----------'+seq+'----------'
            for i,c in enumerate(seq):
                pep=paddedseq[i:i+21]
                listofpeps.append(pep)
        setofpeps=set(listofpeps)
        listofpeps=list(setofpeps)
    
            
            
            

        


        
    
    input_batches=[]
    temp=[]
    for i,pep in enumerate(listofpeps):
        if i%batch_size==0 and i!=0:
            input_batches.append(temp)
            temp=[]
        if pep=='':
            continue
        temp.append(pep.replace("-", "<pad>"))
    input_batches.append(temp)
    # print(listofpeps)
    # print(input_batches)
    pred=predict(input_batches=input_batches)
    write_output(pred,listofpeps,file_output)

    

    

                


        





if __name__ == "__main__":
    main()
    #df=pd.read_csv("output_predictions.csv")
    #print(df)
       

    
