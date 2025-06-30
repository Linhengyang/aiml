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
        * compute:  
        &nbsp;&nbsp;&nbsp;&nbsp;performance & tools & hardware-sensitive  
        * functions:  
        &nbsp;&nbsp;&nbsp;&nbsp;fundamental functions  
    * loss:  
    &nbsp;&nbsp;&nbsp;&nbsp;customized useful loss functions  
    * nn_components:  
        * meta_frames:  
        &nbsp;&nbsp;&nbsp;&nbsp;frameworks  
        * root_layers:  
        &nbsp;&nbsp;&nbsp;&nbsp;nn layers  
        * sub_modules:  
        &nbsp;&nbsp;&nbsp;&nbsp;customized module blocks/bulks for projs
    * utils:  
    &nbsp;&nbsp;&nbsp;&nbsp;data & algo & preprocess  
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

---
    artifact
    ├── app1
    cache
    ├── proj2
    model
    ├── proj3
    logs
    ├── proj1
    ├── app1
    tmp
    ├── proj1
    ├── app1
    tool
    ├── database
    ├── hf_download.py
    aiml
    ├── src
    │   ├── core
    │   │   ├── base
    │   │   │   ├── functions
    │   │   │   │   ├── mask.py
    │   │   │   │   └── patch_operation.py
    │   │   │   └── compute
    │   │   │   │   ├── evaluate_tools.py
    │   │   │   │   ├── predict_tools.py
    │   │   │   │   ├── sampling_tools.py
    │   │   │   │   ├── train_tools.py
    │   │   │   │   └── visualize_tools.py
    │   │   ├── loss
    │   │   │   ├── bp_ranking_loss.py
    │   │   │   ├── l2penalty_mse_loss.py
    │   │   │   └── mask_ce_loss.py
    │   │   ├── nn_components
    │   │   │   ├── meta_frames
    │   │   │   │   ├── __init__.py
    │   │   │   │   └── architectures.py
    │   │   │   ├── root_layers
    │   │   │   │   ├── add_layer_norm.py
    │   │   │   │   ├── attention_pools.py
    │   │   │   │   ├── mc_feat_emb.py
    │   │   │   │   ├── patchify.py
    │   │   │   │   └── positional_encodings.py
    │   │   │   └── modules
    │   │   │       ├── _recsys.py
    │   │   │       ├── _transformer.py
    │   │   │       └── _vit.py
    │   │   └── utils
    │   │       ├── common
    │   │       │   └── seq_operation.py
    │   │       ├── data
    │   │       │   ├── data_assemble.py
    │   │       │   ├── data_split.py
    │   │       │   └── data_transform.py
    │   │       ├── file
    │   │       │   └── text_split.py
    │   │       ├── image
    │   │       │   ├── display.py
    │   │       │   └── mnist.py
    │   │       ├── system
    │   │       │   ├── math.py
    │   │       │   └── statistics.py
    │   │       └── text
    │   │           ├── glossary.py
    │   │           ├── string_segment.py
    │   │           ├── text_preprocess.py
    │   │           ├── tokenizer.py
    │   │           └── vocabulize.py
    │   ├── apps
    │   │   ├── semantic_segmentation
    │   │   └── sentiment_analysis
    │   └── projs
    │       ├── bert
    │       ├── gpt
    │       ├── transformer
    │       └── vit
    ├── tests
    ├── notebooks
    │   ├── speedup.ipynb
    │   └── tokenizer.ipynb
    ├── README.md
    ├── learn.py
    ├── main.py
    └── test.py
---