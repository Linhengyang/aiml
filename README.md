# repo skeleton  
## Execute Note:  
* a `__init__.py` file shows that the package imports all from underlying modules.  
* always run `python -B xxx.py` file under `aiml` directory in case of relative importing  
* always check ideas in `experiment.ipynb`, test code chunk in `test.py`
* after successful tests, wrap code chunk to `_jobs.py` under `src/projs/xxxx`
* invoke function from `_jobs.py` to `main.py`, run `main.py` for official execution

## src:
* core:  
&nbsp;&nbsp;&nbsp;&nbsp;torch code including:  
    * layers:&nbsp;&nbsp;basic layers   
    * blocks:&nbsp;&nbsp;network blocks   
    * models:&nbsp;&nbsp;complete network   
    * loss:&nbsp;&nbsp;customized useful loss functions  
    * optim:&nbsp;&nbsp;customized optimizers 
    * data:&nbsp;&nbsp;data process  
    * evaluation:&nbsp;&nbsp;evaluation tools  
* kits:  
    * huggingface:  &nbsp;&nbsp;utilities for adapting huggingface-style to custom-style
* lib:  
    * share: &nbsp;&nbsp;shared cpp headers & files  
    * tokenizer:  &nbsp;&nbsp;cpp & cython files to boost tokenizer bpe train  
* utils:  
&nbsp;&nbsp;&nbsp;&nbsp;tools including  
    * text:&nbsp;&nbsp;tokenzier, preprocess, etc   
    * image:&nbsp;&nbsp;image related   
    * parquet:&nbsp;&nbsp;parquet format related   
* projs:  
&nbsp;&nbsp;&nbsp;&nbsp;code & config for specific projects  
* apps:  
&nbsp;&nbsp;&nbsp;&nbsp;code & config for specific application  


## Work Note:

the online working space must contain following directories:  
* `aiml`: `git clone https://github.com/Linhengyang/aiml.git`
* `model`: consisting directory named by the `proj_name` -- save trained params
* `log`: consisting directory named by the `proj_name`  -- save logs


also recommend to have:
* `artifact`: -- to save useful outputs
* `tmp`: -- temporary results to be deleted safely
* `cache`: -- to save files may be saved or not

set env:
* `PYTHONPATH`: -- add absolute path of `{path}/{to}/aiml/bin`, e.g.  
`source env.sh` where `env.sh` like following:
```
cd ./aiml
export PYTHONPATH=$(pwd)/bin:$PYTHONPATH
```
or use python(usually in `ext.{package}.__init__.py`) like following:
```
import sys
_bin_dir = os.path.join(os.path.dirname(__file__), '../../bin')
_bin_dir = os.path.abspath(_bin_dir)
if _bin_dir not in sys.path:
    sys.path.insert(0, _bin_dir)
```