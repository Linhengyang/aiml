# test.py


def get_counts_of_pair(tokens_lst:list):
    counts = {}
    for pair in zip(tokens_lst, tokens_lst[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    
    return counts

def merge_pair(tokens_lst:list, pair, new_token):
    new_tokens_lst = []
    i = 0
    while i < len(tokens_lst):
        # check the consecutive two in current position with the pair
        if i < len(tokens_lst) - 1 and tokens_lst[i] == pair[0] and tokens_lst[i+1] == pair[1]: # 当前 i, i+1 两个位置 组成 pair
            new_tokens_lst.append(new_token) # i 位置 记录 new_token
            i += 2 # i+1 位置被跳过, compress。下一轮 检查 i+2 位置
        else: # 当前 i, i+1 两个位置 不组成 pair, 或者 i 已经是 last token
            new_tokens_lst.append( tokens_lst[i] ) # i 位置 记录 原 i 位置 token
            i += 1 # 下一轮 检查 i+1位置
    
    return new_tokens_lst



# BPE train
vocab_size = 276
num_merges = vocab_size - 256 # 20 merges, add 20 new tokens


text = 'this is a test text for bpe algorithm in re-implement GPT2 for project linhengyang'
raw_tokens = list( map(int, text.encode('utf-8')) )
print(f'raw_tokens as {raw_tokens}')

corpus = list(raw_tokens) # deep copy
merges = {} # maintain a mapping for (token_Left, token_Right) --> token_merged, a recorder 记录 token 合并的过程
for i in range(num_merges):
    pair_counts = get_counts_of_pair(corpus) # dict of key: consecutive tokens pair tuple, value: key occurance as pair
    top_pair = max(pair_counts, key=pair_counts.get) # pair of tokens tuple. here we can improve it with more subtle selection
    if pair_counts[top_pair] < 2: # 如果 此个 top_pair 只出现了一次，就不存在合并的必要了
        break
    new_token = i + 256 # 以 rank 为 new token
    print(f'merging {top_pair} into a new token {new_token}: occurance {pair_counts[top_pair]}')
    corpus = merge_pair(corpus, top_pair, new_token) # tokens 更新, occur-most pair of tokens merged
    # 在 tokens list 中，merged pair 被一个 彻底全新的 new_token 替代，所以在下一轮 merge 中，能确保已经被 merge 过的 pair 绝对不会再在 tokens 中出现
    merges[top_pair] = new_token # 记录 merge 记录.

# merges:
# 按 insertion 的顺序，merges 记录了 各pair(as key) merge 的 target as value
# 遍历 tokens 和 遍历 merges，哪个在外？哪个在内？
# 遍历tokens 在外，即从 tokens 的 i 位置开始：检查tokens[i], tokens[i+1]，看它是否是 merges 匹配。若匹配，则 merge。在内侧遍历 merges。
#   遍历完之后，tokens 更新。要回到 i 位置，重新检测。直到 遍历 merges 都找不到匹配。那么 i+=1。
# 等价于 greedy

# 遍历 merges 在外，即从 merges[k] pair_k 开始：检查 pair_k 是否出现在 tokens 中。如果出现，替代 pair_k with its merge。在内侧遍历 tokens。
#   遍历完之后，tokens 更新。k+=1
# merge order 优先

# 采用第二种
# encode
def encode(text):
    tokens_bytes = text.encode('utf-8')
    tokens_ints = list(map(int, tokens_bytes))
    # [116, 32, 98, 101, 32, 97, 32, 114]
    # merges {(101, 32): 256, (256, 105):257, (114, 101):258}
    while True:
        counts = get_counts_of_pair(tokens_ints) # all {consecutive pair: occurance}
        # find the pair in counts with min rank in merge
        # 找到 counts 中，最早在 merges 中出现的 pair（意味着最早被 merge 的pair) 
        pair = min( counts, key=lambda p: merges.get(p, float('inf')) ) # min 函数作用在 iterable object 上，iterable object 会把 all elements 先输入 key fn, 再根据各个 return 选择最小
        # min 自动返回 自定义的最小，if all as float('inf') 这个时候其实是 counts 中没一个 pair 在 merges 里，也就是说 tokens_ints 里没有需要 merge 的了
        if pair not in merges:
            break

        tokens_ints = merge_pair(tokens_ints, pair, merges[pair]) # merge pair. tokens_ints 被完整更新一遍
    
    return tokens_ints

    


# tokens = list(raw_tokens)
tokens_enc = encode(raw_tokens)


if __name__ == "__main__":
    print(f'{len(raw_tokens)} raw tokens as\n {raw_tokens}')
    print(f'{len(corpus)} corpus as\n {corpus}')
    print(f'{len(tokens_enc)} tokens_enc as\n {tokens_enc}')



#  [116, 104, 105, 115, 32, 105, 115, 32, 97, 32, 116, 101, 115, 116, 32, 116, 101, 120, 116, 32, 102, 111, 114, 32, 98, 112, 101, 32, 97, 108, 103, 111, 114, 105, 116, 104, 109, 32, 105, 110, 32, 114, 101, 45, 105, 109, 112, 108, 101, 109, 101, 110, 116, 32, 71, 80, 84, 50, 32, 102, 111, 114, 32, 112, 114, 111, 106, 101, 99, 116, 32, 108, 105, 110, 104, 101, 110, 103, 121, 97, 110, 103]
#  [258, 104, 105, 115, 32, 105, 115, 32, 97, 32, 116, 101, 115, 116, 32, 116, 101, 120, 116, 32, 102, 111, 114, 32, 98, 112, 101, 32, 97, 108, 103, 111, 114, 105, 116, 104, 109, 32, 105, 110, 32, 114, 101, 45, 105, 109, 112, 108, 101, 109, 101, 110, 116, 32, 71, 80, 84, 50, 32, 102, 111, 114, 32, 112, 114, 111, 106, 101, 99, 116, 32, 108, 105, 110, 104, 101, 110, 103, 121, 97, 110, 103]