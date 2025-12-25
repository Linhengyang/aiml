# operations for folders
import os
import shutil # 引入 shutil 模块，虽然这里主要用 os.remove，但它是处理目录的常用工具
import typing as t


def clean_folder(folder_path: str, method:t.Literal['all', 'only_file', 'only_folder']='all', keep=True):
    """
    按方法清空指定文件夹中的所有文件
    Args:
        folder_path (str): 要清空的文件夹的路径
        method:
            all
            only_file
            only_folder
    """
    # 检查路径是否存在 / 检查路径是否确实是一个目录
    assert os.path.exists(folder_path), f"error: folder '{folder_path}' not exists"
    assert os.path.isdir(folder_path), f"error: path '{folder_path}' not a folder"
    
    for item_name in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item_name) # 构建完整路径

        # 只有当此文件的 性质 与 是否删除这个性质的文件 匹配，才删除此文件
        del_file =  os.path.isfile(item_path) and (method in ('all', 'only_file'))
        del_dir = os.path.isdir(item_path) and (method in ('all', 'only_folder'))

        try:
            if del_file:
                os.remove(item_path)
            elif del_dir:
                shutil.rmtree(item_path)
            else:
                pass
        except OSError as e:
            raise OSError(f"processing '{item_path}' errors: {e}")
        except Exception as e:
            raise Exception(f"clean folder error: {e}")
    
    # keep = True --> 保存 folder_path; keep = False --> 不保存 folder_path
    if not keep:
        shutil.rmtree(folder_path)