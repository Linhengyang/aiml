# operations for folders
import os
import shutil # 引入 shutil 模块，虽然这里主要用 os.remove，但它是处理目录的常用工具
import typing as t


def clean_folder(folder_path: str, method:t.Literal['all', 'only_file', 'only_folder']='all'):
    """
    按方法清空指定文件夹中的所有文件
    Args:
        folder_path (str): 要清空的文件夹的路径
        method:
            all
            only_file
            only_folder
    """
    # 1. 检查路径是否存在 / 检查路径是否确实是一个目录
    assert os.path.exists(folder_path), f"error: folder '{folder_path}' not exists"
    assert os.path.isdir(folder_path), f"error: path '{folder_path}' not a folder"
    # os.listdir() 返回文件夹内所有文件和子文件夹的名称列表
    for item_name in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item_name) # 构建完整路径
        is_file = os.path.isfile(item_path)
        is_dir = os.path.isdir(item_path)
        del_file = method in ('all', 'only_file')
        del_dir = method in ('all', 'only_folder')

        try: # 只有当此文件的 性质 与 是否删除这个性质的文件 匹配，才删除此文件
            if is_file and del_file:
                os.remove(item_path)
            elif is_dir and del_dir:
                shutil.rmtree(item_path)
            else:
                pass
        except OSError as e:
            raise OSError(f"processing '{item_path}' errors: {e}")
        except Exception as e:
            raise Exception(f"errors: {e}")