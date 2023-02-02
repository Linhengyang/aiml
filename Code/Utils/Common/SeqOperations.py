
def truncate_pad(line, num_steps, padding_token):
    '''
    inputs: line, num_steps, padding_token
        text: 1D list
        num_steps: integer, setup sequence length
        padding_token: element to pad up to num_steps length

    returns: a 1D list, denoted as L

    explains:
        truncate what exceeds num_steps, or pad padding_token when it shorts
    '''
    if len(line) >= num_steps:
        return line[:num_steps]
    else:
        return line + [padding_token] * (num_steps - len(line))
