# Tokenizer.py
# new version of Tokenizer inspired by GPT tokenizer(tiktokenizer)

# 新版本的 Tokenizer 应该满足：
# 1. 一站式满足 encode（original string --> indices）, decode (indices --> original string），且保证是 round-trips
# 2. 可序列化
# 3. 可扩展
# 4. 高效
# 5. 