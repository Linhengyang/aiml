# repo skeleton  
## Execute Note:  
* a `__init__.py` file shows that the package imports all from underlying modules.  
* always run `python -B xxx.py` file under `aiml` directory in case of relative importing  
* always check ideas in `experiment.ipynb`, test code chunk in `test.py`
* after successful tests, wrap code chunk to `_jobs.py` under `src/projs/xxxx`
* invoke function from `_jobs.py` to `main.py`, run `main.py` for official execution

## src:
* core:
    * base: 
        * compute:&nbsp;&nbsp;performance & tools & hardware-sensitive  
        * functions:&nbsp;&nbsp;fundamental functions  
    * loss:&nbsp;&nbsp;customized useful loss functions  
    * nn_components:  
        * meta_frames:&nbsp;&nbsp;frameworks  
        * root_layers:&nbsp;&nbsp;nn layers  
        * sub_modules:&nbsp;&nbsp;customized module blocks/bulks for projs
    * utils:&nbsp;&nbsp;data & algo & preprocess  
* projs:  
&nbsp;&nbsp;&nbsp;&nbsp;a complete proj needs to implement followings:
    * function
    * dataset
    * network
    * trainer
    * evaluator
    * predictor
    * optimizer
    * loss
    * configs
    * _jobs


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
* `PYTHONPATH`: -- add absolute path of `{path}/{to}/aiml/src/bin`, e.g.
`source env.sh` where `env.sh` like following:
```
cd ./aiml
export PYTHONPATH=$(pwd)/src/bin:$PYTHONPATH
```

---
    aiml
    ├── src
    │   ├── bin
    │   ├── core
    │   │   ├── base
    │   │   │   ├── compute
    │   │   │   └── functions
    │   │   ├── design
    │   │   ├── loss
    │   │   ├── nn_components
    │   │   │   ├── meta_frames
    │   │   │   ├── root_layers
    │   │   │   └── submodules
    │   │   └── utils
    │   │       ├── common
    │   │       ├── data
    │   │       ├── file
    │   │       ├── image
    │   │       ├── system
    │   │       └── text
    │   ├── lib
    │   │   ├── share
    │   │   └── tokenizer
    │   ├── apps
    │   │   ├── bpe_build
    │   │   ├── semantic_segmentation
    │   │   └── sentiment_analysis
    │   └── projs
    │       ├── bert
    │       ├── gpt
    │       ├── transformer
    │       └── vit
    ├── tests
    ├── notebooks
    ├── README.md
    ├── learn.py
    ├── main.py
    ├── test.py
    artifact
    ├── app
    cache
    ├── proj
    model
    ├── proj
    logs
    ├── app
    tmp
    ├── proj
    tool
    ├── database
    └── hf_download.py
---