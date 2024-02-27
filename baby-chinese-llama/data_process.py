import os
import json
import glob
import numpy as np
from tqdm import tqdm
from chatglm_tokenizer.tokenization_chatglm import ChatGLMTokenizer
import pandas as pd 
import pdb

def process_baidu():
    BATCH_SIZE=100000
    cnt=0
    batch_cnt=0
    token=0
    doc_ids=[]
    path=os.path.join("/", "data", "xfli", "data", "563w_baidubaike.json")
    f1=open(path, 'r', encoding='utf-8')

    while True:
        line = f1.readline()
        if not line:
            break
        line = json.loads(line)
        text=''
        try:
            text+=line['title']+": "+line['summary']
        except:
            pass
        for per in line['sections']:
            text += per['title'] + ":" + per["content"] + "。"
        
        text_id = tokenizer.encode(text, add_special_tokens=False)
        text_id.append(tokenizer.special_tokens['<eos>'])
        if len(text_id) > 5:
            doc_ids+=text_id
        cnt+=1
        if cnt%BATCH_SIZE==0:
            batch_cnt+=1
            arr=np.array(doc_ids)
            doc_ids=[]
            with open(os.path.join("/data/xfli/data/tokenized/baicubaike_563w_{}.bin".format(batch_cnt)), 'wb') as f2:
                f2.write(arr.tobytes())
            del arr

def process_wiki_clean():
    path = os.path.join('/data', 'xfli', 'data', 'wikipedia-cn-20230720-filtered.json')
    with open(path, 'r', encoding='utf-8') as f:
        data=json.load(f)
    doc_ids=[]
    for line in tqdm(data):
        text=line['completion']
        text_id=tokenizer.encode(text, add_special_tokens=False)
        text_id.append(tokenizer.special_tokens['<eos>'])
        if len(text_id)>5:
            doc_ids+=text_id
    arr = np.array(doc_ids, dtype=np.uint16)
    spath=os.path.join("/data", 'xfli', 'data', 'wiki.bin')
    with open(spath, 'wb') as f:
        f.write(arr.tobytes())

if __name__ == "__main__":
    tokenizer = ChatGLMTokenizer(vocab_file="./chatglm_tokenizer/tokenizer.model")
    # process_wiki_clean()
    process_baidu()
    print('data processing finished!')
    data_lst=[]
    for data_path in tqdm(data_path_list):
        with open(data_path, 'rb') as f:
            data=np.fromfile(f, dtype=np.uint16)
            data_lst.append(data)
    arr=np.concatenate(data_lst)
    print(arr.shape)
    with open('/data/xfli/data/tokenized/pretrain_xfli.bin', 'wb') as f:
        f.write(arr.tobytes())