# setup.py

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os



tokenizer_dir = os.path.abspath(os.path.dirname(__file__))
share_dir = os.path.abspath("../share")
include_dirs = [np.get_include(), share_dir, tokenizer_dir]
print("Include dirs:", include_dirs)



# 将 bin/ 目录添加到 build_ext, 使得 .so 文件直接输出到 bin/
_bin_dir = os.path.join(tokenizer_dir, '../../../bin')
_bin_dir = os.path.abspath(_bin_dir)
# 自定义 build_ext, 输出到 bin/
from setuptools.command.build_ext import build_ext
class BuildExtToBin(build_ext):
    def initialize_options(self):
        super().initialize_options()
        self.build_lib = _bin_dir


ext_modules = [
    Extension(
        name="mp_pair_count_merge",  # 输出模块名 (.so文件名, 也是 import .so文件时 时用的名字)
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
    name="bpeboost",
    ext_modules=cythonize(
        ext_modules,
        build_dir="build/cython_temp",
        compiler_directives={
            "language_level": "3",
            "boundscheck": True,
            "wraparound": True
        },
    ),
    cmdclass = {'build_ext': BuildExtToBin}
)
