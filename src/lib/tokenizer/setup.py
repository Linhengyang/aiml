# setup.py

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        name="merge_pair",  # 输出模块名 (import 时用的名字)
        sources=["merge_pair.pyx", "merge_pair_api.cpp", "merge_pair_core.cpp"],  # 包含 .pyx 和 C++ 源文件
        language="c++",
        include_dirs=[np.get_include(), "../src/lib/share", "../src/lib/tokenizer"],  # 包含 numpy 的头文件路径
        extra_compile_args=["-O3"],       # 优化编译选项
    )
]

setup(
    name="merge_pair",
    ext_modules=cythonize(
        ext_modules,
        compiler_directives={
            "language_level": "3",
            "boundscheck": True,
            "wraparound": True
        },
    ),
)
