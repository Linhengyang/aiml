# test.py
import torch
import typing as t
import math

cache_play = '../cache/playground/'


from src.core.utils.text.tokenizer import boostBBPETokenizer, ENDOFTEXT

def _regularize_batch(S: int, P: int, data: t.Dict[int, torch.Tensor]):
    '''
    把长度从 S(包含) 到 P(不包含) 的所有 seqBox, 都拆散. 其中整块的 S 部分并入 长度为S的 seqBox, 剩余零碎的 b 部分并入相应 长度为b的 seqBox
    '''
    # 对于 L = S 的 seq_box, 它是现成的, 无需操作 --> [_, 3, S]
    if S not in data:
        data[S] = torch.empty(0, 3, S, dtype=torch.long) # 空 Tensor, [0, 3, S]

    # 对于 L > S 的 seq_box, L = kS + b, k >=1, 0 <= b < S
    for L in range(S+1, P):
        if L in data: # data[L]: [B, 3, L]
            k, b = L // S, L % S
            splitted = data[L].split(S, dim=-1) # [B, 3, kS+b] --split--> k 个 [B, 3, S], 1 个 [B, 3, b]
            if b == 0:
                data[S] = torch.concat([data[S], *splitted], dim=0) # [_, 3, S] stack k 个 [B, 3, S]
            else:
                data[S] = torch.concat([data[S], *splitted[:-1]], dim=0) # [_, 3, S] stack k 个 [B, 3, S]
                # [_, 3, b] --> [_+B, 3, b]
                data[b] = torch.concat([data.get(b, torch.empty(0, 3, b, dtype=torch.long)), splitted[-1]], dim=0)
            del data[L] # L 已经拆散分配到 S 和 b. 分配完毕后 从 data 中消除对应 kv


def _size_stat(S: int, P: int, data: t.Dict[int, torch.Tensor]):
    '''
    计算长度从 S(包含) 到 P(不包含) 的 所有 seqBox 的总 size
    '''
    _size = 0
    for L in range(S, P):
        if L in data:
            _size += data[L].size(0) # [B, 3, L]
    return _size


def _pad(data: torch.Tensor, L):
    # data shape: [B, 3, q]
    # pad to [B, 3, 0]
    q = data.size(-1)
    if q < L:
        zeros = torch.zeros(data.shape[0], data.shape[1], L-q)
        return torch.cat([data, zeros], dim=-1)
    elif q > L:
        return data[:, :, :L]
    else:
        return data


def pack_text(tgt_L: int, data: t.Dict[int, torch.Tensor]):
    '''
    对于长度 大于等于 tgt_L 的 seqBox, regularize 到 tgt_L. 计算此时 tgt_L/2 到 tgt_L 的seqBox的总size B, 以及1到tgt_L的总size B_
    如果 B >= 2, 则执行下一步; 如果 B < 2 但是 B_ >=2, 则把所有 1到tgt_L/2 的都PAD到 tgt_L/2, 然后执行下一步后结束; 如果 B_ < 2, 结束

    对于长度 大于等于 tgt_L/2 但又小于 tgt_L 的seqBox, regularize 到 tgt_L/2 后 得到 [B, 3, tgt_L/2] 的 seqBox.
    拆分 dim0 pack 到 dim2, 得到 [B//2, 3, tgt_L]. 这里可能要pack 1条 [1, 3, tgt_L/2] 的 datapoint 到 residual

    对于长度 大于等于 tgt_L/4 但又小于 tgt_L/2 的seqBox, regularize 到 tgt_L/4 后 得到 [B, 3, tgt_L/4] 的 seqBox.
    拆分 dim0 pack 到 dim2, 得到 [B//4, 3, tgt_L]. 这里可能要pack 3条 [1, 3, tgt_L/4] 的 datapoint 到 residual.

    ...

    重复上述流程 t-1 次后, 检查所有长度小于 tgt_L/2^t 的 seqBoxes, 计算 min_L, 以及总size B. 若 min_L < tgt_L/2^t, B >= 2^t 成立,
    步骤一: 对于长度小于 tgt_L/2^t 的, 全部 PAD 到 长度 tgt_L/2^t.
    步骤二: 对于长度 大于等于 tgt_L/2^t 但又小于 tgt_L/2^(t-1) 的seqBox, regularize 到 tgt_L/2^t 后 得到 [B, 3, tgt_L/2^t] 的 seqBox.
    拆分 dim0 pack 到 dim2, 得到 [B//2^t, 3, tgt_L]. 这里可能要 pack 2^t-1 条 [1, 3, tgt_L/2^t] 的 datapoints 到 residual.
    '''
    min_L, max_L = min(data), max(data)
    T = int( math.log2(tgt_L/2) ) # what if T = 0 here?
    
    _regularize_batch(tgt_L, max_L+1, data) # what if tgt_L >= max_L?
    # 此时 data: min_L -- tgt_L

    residual = torch.empty(1, 3, 0, dtype=torch.long)

    for t in range(1, T+1): # t 最多到 T, 因为当 t > T 时, min_L > tgt_L/2^t, 而裁剪 min_L 实属没有必要.
        # 考虑子区间 tgt_L/2^t -- tgt_L/2^(t-1), 即:
        # t = 1: tgt_L/2 -- tgt_L
        # t = 2: tgt_L/4 -- tgt_L/2
        # ...
        # t = t: tgt_L/2^t -- tgt_L/2^(t-1)
        print(f'epoch {t}')
        L_t = tgt_L//2**t
        up_semi_size = _size_stat(L_t, tgt_L//2**(t-1), data) # size from tgt_L/2^t(incl) -- tgt_L/2^(t-1)(not-incl)
        down_semi_size = _size_stat(min_L, L_t, data) # size from min_L(incl) -- tgt_L/2^t(not-incl)

        if up_semi_size >= 2**t:
            _regularize_batch(L_t, tgt_L//2**(t-1), data)
            # data[L_t]: [up_semi_size, 3, tgt_L//2**t]
            # up_semi_size = 2**t * incre_size + r, 0 <= r < 2**t
            incre_size, r = up_semi_size//(2**t), up_semi_size%(2**t)
            splitted = data[L_t].split(incre_size, dim=0) # 2**t个 [incre_size, 3, tgt_L//2**t], 1个 [r, 3, tgt_L//2**t]
            if r != 0:
                increment = _pad( torch.cat(splitted[:-1], dim=-1), tgt_L ) # [incre_size, 3, tgt_L]
                # 补充 last [r, 3, tgt_L//2**t] 到 [2**t, 3, tgt_L//2**t]
                residual = torch.cat([splitted[-1], torch.zeros(2**t-r, 3, L_t, dtype=torch.long)], dim=0)
                residual = _pad( torch.cat(residual.split(1, dim=0), dim=-1), tgt_L ) # [2**t, 3, tgt_L//2**t] --> [1, 3, tgt_L]
                increment = torch.cat([increment, residual], dim=0) # [.., 3, tgt_L]
            else:
                increment = _pad( torch.cat(splitted, dim=-1), tgt_L ) # [.., 3, tgt_L]
            data[tgt_L] = torch.cat([data[tgt_L], increment], dim=0)
            del data[L_t]
        elif up_semi_size + down_semi_size >= 2**t:
            _size = up_semi_size + down_semi_size
            # 遍历所有 长度小于 tgt_L//2**t 的 seqBox, 将它们的长度全部 PAD 到 tgt_L//2**t, 然后全部 stack 到 长度为 tgt_L//2**t 的 seqBox 上
            if L_t not in data:
                data[L_t] = torch.empty(0, 3, L_t, dtype=torch.long) # [.., 3, tgt_L//2**t]
            
            for l in range(min_L, L_t):
                if l in data:
                    data[L_t] = torch.cat([data[L_t], _pad(data[l], L_t)], dim=0) # [..++, 3, tgt_L//2**t]
                    del data[l]
            _regularize_batch(L_t, tgt_L//2**(t-1), data)
            # data[L_t] [_size, 3, tgt_L//2**t]
            # _size = 2**t * incre_size + r, 0 <= r < 2**t
            incre_size, r = _size//(2**t), _size%(2**t)
            splitted = data[L_t].split(incre_size, dim=0) # 2**t个 [incre_size, 3, tgt_L//2**t], 1个 [r, 3, tgt_L//2**t]
            if r != 0:
                increment = _pad( torch.cat(splitted[:-1], dim=-1), tgt_L ) # [incre_size, 3, tgt_L]
                # 补充 last [r, 3, tgt_L//2**t] 到 [2**t, 3, tgt_L//2**t]
                residual = torch.cat([splitted[-1], torch.zeros(2**t-r, 3, L_t, dtype=torch.long)], dim=0)
                residual = _pad( torch.cat(residual.split(1, dim=0), dim=-1), tgt_L ) # [2**t, 3, tgt_L//2**t] --> [1, 3, tgt_L]
                increment = torch.cat([increment, residual], dim=0) # [.., 3, tgt_L]
            else:
                increment = _pad( torch.cat(splitted, dim=-1), tgt_L ) # [.., 3, tgt_L]
            data[tgt_L] = torch.cat([data[tgt_L], increment], dim=0)
            del data[L_t]
        else:
            break

    return data[tgt_L], data

if __name__ == "__main__":
    text = \
"""Everything seemed to be going perfectly.	Tout semblait se dérouler parfaitement.
The road is straight for over ten miles.	La route est en ligne droite sur une distance de plus de dix miles.
My apartment is on the fourth floor.	Mon appartement se situe au quatrième étage.
You have to see it to believe it.	Vous devez le voir pour le croire.
"Just go in and tell the boss you want a raise." "That's easier said than done."	«Entre seulement et dis au patron que tu veux une augmentation.» «Plus facile à dire qu'à faire.»
What made you ask that question?	Qu'est-ce qui t'a fait poser cette question ?
I don't mind it.	Ça ne me dérange pas.
Maybe we should cancel the meeting.	Peut-être devrions-nous annuler la réunion.
We'll do everything we can to help you.	Nous ferons tout ce que nous pouvons pour vous aider.
Could you fill me in?	Pourrais-tu me mettre au courant ?
At last, we got to the lake.	Enfin, nous sommes arrivés au lac.
I'd like you to read this book.	J'aimerais que tu lises ce livre.
The mail train lost most of its mail in the fire.	Le train postal a perdu une bonne partie de son courrier dans l'incendie.
I need paint.	Il me faut de la peinture.
Did you see a brown wallet around here?	Avez-vous vu un portefeuille marron dans les alentours ?
In a way you are right, but I still have doubts.	D'une certaine manière tu as raison, mais j'ai encore des doutes.
I'll act as a guide for you.	Je serai ton guide.
You must be cautious.	Vous devez être prudente.
Hey, wait a minute, are you thinking what I'm thinking?	Eh, minute ! Es-tu en train de penser à ce que je suis en train de penser ?
What did you make?	Qu'as-tu confectionné ?
Excuse me, which way is the station?	Excusez-moi, de quel côté est la gare ?
Don't forget we have to do our homework.	N’oubliez pas qu'il nous faut faire nos devoirs.
Give it to her.	Donne-le-lui.
Tell whoever comes that I'm out.	Dis à quiconque se présente que je suis sorti !
Look at this picture.	Regarde ce tableau.
Would that make you happy?	Cela vous rendrait-il heureuse ?
I'm not shy.	Je ne suis pas timide.
Do you want another one of these?	En voulez-vous encore une ?
It's getting worse and worse.	C'est de pire en pire.
Bring me the Kleenex.	Apporte-moi les Kleenex.
I know you aren't stupid enough to believe that.	Je sais que vous n'êtes pas stupides au point de croire cela.
I think I could handle that.	Je pense que je pourrais gérer ça.
I have some time.	J'ai du temps.
It has to be done.	Il faut le faire.
Did you iron all the shirts?	Avez-vous repassé toutes les chemises ?
We abhor violence.	Nous détestons la violence.
He was busy when I called him up.	Il était occupé lorsque je l'ai appelé.
Sometimes I hear things.	Quelques fois, j'entends des choses.
He writes letters to his mother.	Il écrit des lettres à sa mère."""

    tok = boostBBPETokenizer(name='.', buffer_dir='../cache/temp')
    tok.load('../artifact/gpt2/tokenizer/mt.tok')

    lines = text.split('\n')
    data = {}
    for i, line in enumerate(lines):
        line += ENDOFTEXT
        tokens = tok.encode(line, allowed_special=set([ENDOFTEXT]))
        L = len(tokens) - 1
        input = tokens[:-1]
        label = tokens[1:]
        segments = [i+1]*L
        datapoint = torch.tensor([input, label, segments], dtype=torch.long).unsqueeze(0) # [1, 3, L]
        data[L] = torch.cat([data.get(L, torch.empty(0, 3, L, dtype=torch.long)), datapoint], dim=0)

    # 检查 tokenizer 正确性
    assert tok.decode(data[13][0][0].tolist()) == lines[0]

    sorted_data = {k: data[k] for k in sorted(data)}
    data = sorted_data

    tgt_L = 20
    output, all_ = pack_text(tgt_L, data)
    _len = 0
    for l in all_.keys():
        if l == tgt_L:
            continue
        print(f'{l}: size {all_[l].size(0)}, len {all_[l].size(2)}')
        _len += all_[l].size(0)*all_[l].size(2)
    print(_len)