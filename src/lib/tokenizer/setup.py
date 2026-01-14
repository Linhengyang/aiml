# setup.py

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

tokenizer_dir = os.path.abspath(os.path.dirname(__file__))
share_dir = os.path.abspath("../share")
include_dirs = [np.get_include(), share_dir, tokenizer_dir]
print("Include dirs:", include_dirs)


ext_modules = [
    Extension(
        name="pair_count_merge",  # 输出模块名 (import 时用的名字)
        sources=[
            "mp_pair_count_merge.pyx",
            "mp_pair_count_merge_api.cpp",
            "core_merge_pair_mp.cpp",
            "core_count_pair_mp.cpp",
            os.path.join(share_dir,"memory_block.cpp"),
            os.path.join(share_dir,"memory_pool_singleton.cpp"),
            os.path.join(share_dir,"memory_pool.cpp"),
            ],  # 包含 .pyx 和所有涉及到的 C++ 源文件
        language="c++",
        include_dirs=include_dirs,  # 包含 numpy 的头文件路径
        extra_compile_args=["-O3", "-std=c++17"],       # 优化编译选项
    )
]

setup(
    name="pair_count_merge",
    ext_modules=cythonize(
        ext_modules,
        build_dir="build/cython_temp",
        compiler_directives={
            "language_level": "3",
            "boundscheck": True,
            "wraparound": True
        },
    ),
)
