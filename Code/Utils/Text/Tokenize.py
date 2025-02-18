import typing as t


def line_tokenize_simple(
        sentence:str,
        symbols: t.List[str] | t.Set[str] | None = None
        ) -> t.List[str]:
    
    return sentence.split(' ')