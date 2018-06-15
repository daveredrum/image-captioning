import torch

def decode_outputs(sequence, cap_lengths, dictionary, phase):
    decoded = []
    if phase == "train":
        # get the indices for each predicted word
        _, indices = torch.max(sequence, 2)
        # chop the sequences according to their lengths
        unpadded_sequence = [indices[i][:int(cap_lengths.tolist()[i])].tolist() for i in range(cap_lengths.size(0))]
        # decode the indices
        for sequence in unpadded_sequence:
            temp = []
            for idx in sequence:
                try:
                    temp.append(dictionary[idx])
                except Exception:
                    pass
            decoded.append(" ".join(temp))
    elif phase == "val":
        for i in range(len(sequence)):
            temp = []
            for j in range(len(sequence[i])):
                try:
                    temp.append(dictionary[sequence[i][j]])
                except Exception:
                    pass
            decoded.append(" ".join(temp))

    return decoded

# for model with attention
def decode_attention_outputs(sequence, cap_lengths, dictionary, phase):
    decoded = []
    if phase == "train":
        # get the indices for each predicted word
        _, indices = torch.max(sequence, 2)
        # chop the sequences according to their lengths
        unpadded_sequence = [indices[i][:int(cap_lengths.tolist()[i])-1].tolist() for i in range(cap_lengths.size(0))]
        # decode the indices
        for sequence in unpadded_sequence:
            temp = ['<START>']
            for idx in sequence:
                try:
                    temp.append(dictionary[idx])
                except Exception:
                    pass
            decoded.append(" ".join(temp))
    elif phase == "val":
        for i in range(len(sequence)):
            temp = ['<START>']
            for j in range(len(sequence[i])):
                try:
                    temp.append(dictionary[sequence[i][j]])
                except Exception:
                    pass
            decoded.append(" ".join(temp))

    return decoded

def clip_grad_value_(optimizer, clip_value):
    '''
    in-place gradient clipping
    '''
    clip_value = float(clip_value)
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-clip_value, clip_value)