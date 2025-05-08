# run some simple scripts
from Code.Utils.File.TextSplit import split_textfile




full_text = "../../data/text_translator/fra-eng/eng2fra.txt"

output_dir = "../../data/text_translator/fra-eng"


split_textfile(full_text, [0.1, 0.05], output_dir)