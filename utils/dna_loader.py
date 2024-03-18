from torch.utils.data import Dataset, DataLoader
import numpy as np
import os, glob


## some functions
def load_seq_and_lab(seq_path, seq_len = 200, dtype = 'int8'):
    chr = os.path.basename(seq_path).replace('_seq.bin', '')
    lab_path = seq_path.replace('seq', 'lab')
    seq = np.fromfile(seq_path, dtype =  dtype).reshape(-1, seq_len)
    lab = np.fromfile(lab_path, dtype = dtype).reshape(seq.shape[0], -1)
    return chr, seq, lab

def load_all(seq_paths, seq_len = 200, dtype = 'int8'):
    chr_sizes = []
    all_seq = dict()
    all_lab = dict()
    seq_paths = sorted(seq_paths)
    for path in seq_paths:
        chr, seq, lab = load_seq_and_lab(path, seq_len, dtype)
        all_seq[chr] = seq
        all_lab[chr] = lab
        chr_sizes.append(seq.shape[0])
    chr_sizes = np.array(chr_sizes)
    return chr_sizes, all_seq, all_lab
        


re_encoding_map = {1: 4, 2: 3, 3: 2, 4: 1, 0: 0}

def reverse_complement(seq):
    return np.vectorize(re_encoding_map.get)(seq)[::-1]


    
class Long_Sequence_Dataset(Dataset):
    def __init__(self, seq_paths, n_center_windows, n_pad_windows, eval_mode = False, random_reverse_complement = False, dtype = 'int8', window_len = 200):
    
        chr_sizes, all_seq, all_lab = load_all(seq_paths, window_len, dtype)
        
        self.n_center_windows = n_center_windows
        self.dtype = dtype
        self.window_len = window_len
        self.n_pad_windows = n_pad_windows
        self.eval_mode = eval_mode
        self.random_reverse_complement = random_reverse_complement

        self.chr_longseq_sizes = chr_sizes//n_center_windows - 1
        # chr_sizes//n_center_windows to compute number of sequences, -1 to avoid noises/ errors in the end of chromosomes
        
        self.chr_sizes = chr_sizes
        self.all_seq = all_seq
        self.all_lab = all_lab
        self.chr_names = list(all_seq.keys())
        self.N = self.chr_longseq_sizes.sum()
        self.probs = chr_sizes/chr_sizes.sum()
        
        ## for onehot encoding
        vocab = np.array([1,2,3,4], dtype = np.int8)
        self.vocab = vocab[:, None]

    def one_hot(self, dna_idx):
        dna_idx = dna_idx[None,:]
        return np.float32(dna_idx == self.vocab)

        
    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        if self.eval_mode == False:
            rand = np.argmax(np.random.multinomial(1, self.probs))
            chr = self.chr_names[rand]
            n = self.chr_longseq_sizes[rand]
            length = self.chr_sizes[rand]
            center = np.random.randint(0, n)
            center = center*self.n_center_windows + np.random.randint(0, self.n_center_windows)
            # get starting, random shift by np.random.randint(0, self.n_center_windows)
            start = center - self.n_pad_windows
            end = center + self.n_center_windows + self.n_pad_windows
            if start >= 0 and end <= length:
                seq = self.all_seq[chr][start:end]
            elif start < 0:
                pad = np.zeros(((0-start),self.window_len), dtype = self.dtype)
                seq = self.all_seq[chr][:end]
                seq = np.concatenate((pad, seq))
            elif end > length:
                pad = np.zeros(((end - length),self.window_len), dtype = self.dtype)
                seq = self.all_seq[chr][start:]
                seq = np.concatenate((pad, seq))
            lab = self.all_lab[chr][center : center+self.n_center_windows]
            
        elif self.eval_mode == True:
            assert len(self.all_seq) == 1, 'eval_mode is only used with 1 chromosome input!'
            center = idx *self.n_center_windows
            chr = self.chr_names[0]
            length = self.chr_sizes[0]
            start = center - self.n_pad_windows
            end = center + self.n_center_windows + self.n_pad_windows
            
            if start >= 0 and end <= length:
                seq = self.all_seq[chr][start:end]
            elif start < 0:
                pad = np.zeros(((0-start),self.window_len), dtype = self.dtype)
                seq = self.all_seq[chr][:end]
                seq = np.concatenate((pad, seq))
            elif end > length:
                pad = np.zeros(((end - length),self.window_len), dtype = self.dtype)
                seq = self.all_seq[chr][start:]
                seq = np.concatenate((pad, seq))
            
            lab = self.all_lab[chr][center : center+self.n_center_windows]

        seq = seq.reshape(-1)
        # data argumentation by reverse complement, apply for training
        if self.random_reverse_complement == True:
            random_value = np.random.choice([0, 1], p=[0.5, 0.5])
            ## hanlde both random with p=0.5 and training and p=1 when validation with random_reverse_complement
            if random_value == 1 or self.eval_mode == True:
                seq = reverse_complement(seq)
                
        seq = self.one_hot(seq)
        lab = np.float32(lab)
        if self.n_center_windows == 1:
            lab = np.squeeze(lab, axis = 0)
        return seq, lab




def load_data_long_sequence(seq_paths, n_center_windows, n_pad_windows, eval_mode = False,\
                       random_reverse_complement = False, dtype = 'int8', window_len = 200, batch_size = 32, num_workers=0, **kwargs):
    #shuffle is always true to avoid undefinded ROC 
    dataset = Long_Sequence_Dataset(seq_paths, n_center_windows, n_pad_windows, eval_mode, \
                       random_reverse_complement, dtype, window_len)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)



        
def get_output_dim(bin_files, n_center_windows = 1, n_pad_windows = 25, window_len = 200):
    my_data = load_data_long_sequence(bin_files, n_center_windows, n_pad_windows, window_len = window_len, eval_mode = False, random_reverse_complement=False, batch_size=4)
    i = iter(my_data)
    x,y = next(i)
    return y.shape[-1]
