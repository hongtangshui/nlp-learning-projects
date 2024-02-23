import json
import glob
import numpy as np
from tqdm import tqdm
from chatglm_tokenizer.tokenization_chaglm import chatglm_tokenizer
import pandas as pd 



def process_wiki_clean():
    path = os.path.join('data', 'xfli', 'data', 'wikipedia-cn-20230720-filtered.json')
    with open(path, 'r', encoding='utf-8') as f:
        data=json.load(f)
    doc_ids=[]
    for line in tqdm(data):
        text=line['completion']
        text_id=tokenizer.encode(text, add_specail_tokens=False)
        text_id.append(tokenizer.special_tokens['<eos>'])
        if len(text_id)>5:
            doc_ids+=text_id
    arr = np.array(doc_ids, dtype=np.uint16)
    spath=os.path.join("data", 'xfli', 'data', 'wiki.bin')
    with open(spath, 'wb') as f:
        f.write(arr.tobytes())

if __name__ == "__main__":
    tokenizer = ChatGLMTokenizer(vocabfile="./chatglm_tokenizer/tokenizer.model")
    process_wiki_clean()
    print('data processing finished!')
    exit()
    data_lst=[]
    for data_path in tqdm(data_path_list):
        with open(data_path, 'rb') as f:
            data=np.fromfile(f, dtype=np.uint16)
            data_lst.append(data)
    arr=np.concatenate(data_lst)
    print(arr.shape)
    with open('./data/pretrain_data.bin', 'wb') as f:
        f.write(arr.tobytes())