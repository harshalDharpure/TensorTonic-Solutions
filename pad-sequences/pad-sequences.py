import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    # If empty input
    if len(seqs) == 0:
        return np.array([]).reshape(0, 0)
    
    # Find max length
    if max_len is None:
        max_len = max(len(seq) for seq in seqs)
    
    result = []
    
    for seq in seqs:
        if len(seq) > max_len:
            # Truncate
            new_seq = seq[:max_len]
        else:
            # Pad
            new_seq = seq + [pad_value] * (max_len - len(seq))
        
        result.append(new_seq)
    
    return np.array(result, dtype=int)