# test.py
import torch
import typing as t
import math

cache_play = '../cache/playground/'


from src.core.utils.text.tokenizer import boostBBPETokenizer, ENDOFTEXT
from src.core.utils.common.seq_operation import pack_seq_to_batch_pow2


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

    # output, all_ = pack_text(tgt_L, data)
    output, all_ = pack_seq_to_batch_pow2(data, tgt_L, min_L=2, pad_value=0)

    _len = 0
    for l in all_.keys():
        if l == tgt_L:
            continue
        print(f'{l}: size {all_[l].size(0)}, len {all_[l].size(2)}')
        _len += all_[l].size(0)*all_[l].size(2)
    print(_len)
    