#!/usr/bin/env python
import argparse
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

torch.__version__

from utils.dna_loader import *
from utils.train_pytorch import *


# Get cpu, gpu or mps device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device, version {torch.__version__}")


def model_loader(model_path, n_out = 95):
    model = Model(n_out)

    try:
        weights = torch.load(model_path, map_location=device)
        model.load_state_dict(weights)
    except:
        model = nn.DataParallel(model)
        weights = torch.load(model_path, map_location=device)
        model.load_state_dict(weights)

    return model



def get_evals(label, pred, n_out = 95):
    evals = {
        'auroc': np.zeros(n_out),
        'aupr': np.zeros(n_out),
    }
    for i in range(n_out):
        try:
            auroc = roc_auc_score(y_true=label[:, i], y_score=pred[:, i])
            aupr = average_precision_score(y_true=label[:, i], y_score=pred[:, i])
            evals['auroc'][i] = auroc
            evals['aupr'][i] =  aupr
        # only one class present
        except ValueError:
            evals['auroc'][i] = np.nan
            evals['aupr'][i] = np.nan
    return evals


class Eval_Model(object):
    def __init__(self, model, use_logit = 0):
        device = (
            "cuda"
            if torch.cuda.is_available()
            # else "mps"
            # if torch.backends.mps.is_available()
            else "cpu"
        )
        self.device = device
        self.use_logit = use_logit
        self.model = model.to(self.device)
        
    def pred_step(self, x):
        x = x.to(self.device)
        o = self.model(x)
        if self.use_logit == 0:
            o_prob = o.detach().cpu().numpy()
        else:
            o_prob = torch.sigmoid(o).detach().cpu().numpy()
            
        #o_pred = (o_prob > 0.5).astype(float)

        return o_prob

    def pred_all(self, data_loader_fw, data_loader_rc ):
        self.model.eval()
       
        ground_truth = []
        pred = []

        ## forward
        progress_bar = tqdm(enumerate(data_loader_fw), total=len(data_loader_fw))
        for batch_idx, (x, y) in progress_bar:
            ground_truth.append(y)
            pred.append(self.pred_step(x))
            #if batch_idx > 200: break

        ## reverse complement
        progress_bar = tqdm(enumerate(data_loader_rc), total=len(data_loader_rc))
        for batch_idx, (x, y) in progress_bar:
            ground_truth.append(y)
            pred.append(self.pred_step(x))
            #if batch_idx > 200: break

        ground_truth = np.concatenate(ground_truth, axis = 0)
        pred = np.concatenate(pred, axis = 0)
        return ground_truth, pred



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Evaluate deep neural network...)", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--test", nargs='+', help = "test data *_seq.bin generated by 'write_data_bin.py", required = True)
    parser.add_argument("--out", default = "output.pkl", help = "output")
    parser.add_argument('--window_len', type=int, default = 200, help = 'window len, default 200bp')
    parser.add_argument('--n_center_windows', type=int, default = 1, help = 'n center windows: n windows the model will predict lables')
    parser.add_argument('--n_pad_windows', type=int, default = 25, help = 'n padding windows: n windows paded in both sides. For example: seq len 1000 bp has n_pad_windows = 2 (*200) - or 400bp each sides; seq len 10200 bp has n_pad_windows = 25 (*200) - or 5000bp each sides.')
    parser.add_argument('--batch_size', type=int, default = 256, help = 'batch size')
    parser.add_argument("--model", required = True, help = "model class (in a python script) to import")
    parser.add_argument("--model_weight", required = True, help = "model weights (ih *.th file) to import")
    parser.add_argument('--threads', type=int, default = 0, help = 'CPU cores for data pipeline loading')
    parser.add_argument('--logit', type=int, default = 1, help = 'need sigmoid transformation?')
    

    ### input args
    args = parser.parse_args()
    test_files = args.test
    out = args.out
    batch_size = args.batch_size
    window_len = args.window_len
    N_CENTER_WINDOW = args.n_center_windows
    n_pad_windows = args.n_pad_windows
    num_threads = args.threads
    model_in = args.model
    model_weight = args.model_weight
    use_logit = args.logit

    N_OUT = get_output_dim(test_files, N_CENTER_WINDOW, n_pad_windows, window_len)

    ## import model
    model_impoter = f'exec("from {model_in} import *")'
    eval(model_impoter)
    print(f"loaded model class {model_in}")

    model = model_loader(model_weight, N_OUT)

    test_data_fw = load_data_long_sequence(test_files, N_CENTER_WINDOW, n_pad_windows, eval_mode = True,\
        random_reverse_complement=False, batch_size=batch_size, num_workers = num_threads)

    test_data_rc = load_data_long_sequence(test_files, N_CENTER_WINDOW, n_pad_windows, eval_mode = True,\
        random_reverse_complement=True, batch_size=batch_size, num_workers = num_threads)


    m = Eval_Model(model, use_logit)
    ground_truth, pred = m.pred_all(test_data_fw, test_data_rc)
    res = get_evals(ground_truth, pred, N_OUT)
    
    print(f'test file: {test_files}')
    print('AUC:')
    print(np.nanmean(res['auroc']))
    print('AP:')
    print(np.nanmean(res['aupr']))
    # write file
    import pickle
    with open(out, 'wb') as pickle_file:
        pickle.dump(res, pickle_file)



