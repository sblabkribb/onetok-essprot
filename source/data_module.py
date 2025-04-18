import torch, gc, os, time, multiprocessing
import pandas as pd
import numpy as np
from esm.models.esmc import ESMC
from esm.sdk.api import ESM3InferenceClient, ESMProtein, LogitsConfig
from transformers import EsmModel, EsmTokenizer, BertModel, BertTokenizer, T5EncoderModel, T5Tokenizer
from functools import partial



EMBEDDING_CONFIG = LogitsConfig(sequence=True, return_embeddings=True)



def load_embedding_model(embed_ver:str):
    embed_ver = embed_ver.lower()
    if "esm3" in embed_ver:
        model = ESMC.from_pretrained("esmc_600m")
        tokenizer = None
    elif "esm2" in embed_ver:
        model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
        model = model.eval()
        tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    elif "bert" in embed_ver:
        model = BertModel.from_pretrained("Rostlab/prot_bert_bfd")
        model = model.eval()
        tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False)
    elif "t5" in embed_ver:
        model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
        model = model.eval()
        tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
    gc.collect()
    
    return model, tokenizer



def embed_sequence(model:ESM3InferenceClient, sequence:str):
    protein = ESMProtein(sequence=sequence)
    protein_tensor = model.encode(protein)  # integer encoding
    output = model.logits(protein_tensor, EMBEDDING_CONFIG)  # embedding
    return output.embeddings.squeeze()



def embed_sequences(model, batch, embed_ver:str):
    # embed tokenized squences with multi-streamming
    num_streams = min(len(batch), torch.get_num_threads())
    streams = [torch.cuda.Stream() for _ in range(num_streams)]
    results = []
    for i, seq_token in enumerate(batch):
        stream = streams[i % num_streams]
        with torch.cuda.stream(stream):
            if 'esm3' in embed_ver:
                embeddings = embed_sequence(model, seq_token)
            else:
                with torch.no_grad():
                    embeddings = model(*seq_token)
                embeddings = embeddings.last_hidden_state.squeeze()
            results.append(embeddings)
    return results



def extract_embedding(embedding, embed_ver:str):
    if 't5' in embed_ver.lower():
        emb_allmean = embedding.mean(0)
        emb_aamean = embedding[:-1].mean(0)
        emb_bos = None
        emb_eos = embedding[-1]
        emb_first = embedding[0]
        emb_center = embedding[(len(embedding) - 1) // 2]
        emb_last = embedding[-2]
    else:
        emb_allmean = embedding.mean(0)
        emb_aamean = embedding[1:-1].mean(0)
        emb_bos = embedding[0]
        emb_eos = embedding[-1]
        emb_first = embedding[1]
        emb_center = embedding[len(embedding) // 2]
        emb_last = embedding[-2]
    
    return emb_allmean, emb_aamean, emb_bos, emb_eos, emb_first, emb_center, emb_last



def embed_batch(batch, model, embed_ver:str):
    # embed protein sequences
    embeddings = embed_sequences(model, batch, embed_ver)
    # extract protein embedding by type
    extract_emb = partial(extract_embedding, embed_ver=embed_ver)
    emb_allmean, emb_aamean, emb_bos, emb_eos, emb_first, emb_center, emb_last = zip(
        *list(map(extract_emb, embeddings))
    )
    emb_allmean = torch.stack(emb_allmean).to('cpu', non_blocking=True)
    emb_aamean = torch.stack(emb_aamean).to('cpu', non_blocking=True)
    if 't5' in embed_ver.lower():
        emb_bos = None
    else:
        emb_bos = torch.stack(emb_bos).to('cpu', non_blocking=True)
    emb_eos = torch.stack(emb_eos).to('cpu', non_blocking=True)
    emb_first = torch.stack(emb_first).to('cpu', non_blocking=True)
    emb_center = torch.stack(emb_center).to('cpu', non_blocking=True)
    emb_last = torch.stack(emb_last).to('cpu', non_blocking=True)
    
    return emb_allmean, emb_aamean, emb_bos, emb_eos, emb_first, emb_center, emb_last



def load_embedding(emb_type:str, data_path:str, data_ver:str, emb_ver:str, set_ver:str):
    file_name = f'{data_ver}-{emb_ver}_{set_ver}.h5'
    if 't5' in emb_ver.lower() and emb_type.lower() == 'bos':
        return None
    return emb_type, pd.read_hdf(os.path.join(data_path, file_name), key=emb_type)



def parallel_load(embed_types, data_path:str, data_ver:str, embed_ver:str, set_ver:str):    
    par_arg = [(emb_type, data_path, data_ver, embed_ver, set_ver) for emb_type in embed_types]
    with multiprocessing.Pool() as pool:
        results = pool.starmap(load_embedding, par_arg)
    # gather datasts to a dictionary
    return {key: df for result in results if result for key, df in [result]}



def split_dataset(df_info, excluded_strains, test_strains):
    # get indices of samples to be excluded
    loc_exclude = {}        
    for exc_ver, ids in excluded_strains.items():
        loc_exclude[exc_ver] = df_info['file_id'].isin(ids)
        print("Dataset to exclude(" + exc_ver + "):", sum(loc_exclude[exc_ver]))
    indice_train = [sum(loc) == 0 for loc in zip(*loc_exclude.values())]

    # get indices of test samples
    loc_test = {}
    for ts_ver, ids in test_strains.items():
        loc_test[ts_ver] = df_info['file_id'].isin(ids)
    indice_test = [sum(loc) >= 1 for loc in zip(*loc_test.values())]
    
    return indice_train, indice_test



def save_dataset(df, save_path:str, save_key:str):
    time_start = time.time()
    df.to_hdf(save_path, key=save_key, complevel=5, complib='zlib')
    print(f"{save_key}: {time.time() - time_start:.1f} sec")
    display(df)



def layernorm(arr, epsilon=1e-9):
    mean = np.mean(arr, axis=1, keepdims=True)
    std = np.std(arr, axis=1, keepdims=True)
    normalized = (arr - mean) / (std + epsilon)    
    return normalized