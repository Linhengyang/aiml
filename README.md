# repo skeleton  
## Execute Note:  
* a `__init__.py` file shows that the package imports all from underlying modules.  
* always run `python -B xxx.py` file under `aiml` directory in case of relative importing  
* always check ideas in `experiment.ipynb`, test code chunk in `test.py`
* after successful tests, wrap code chunk to `_jobs.py` under `Code/projs/xxxx`
* invoke function from `_jobs.py` to `main.py`, run `main.py` for official execution

## Code Note:
* Base:  
&nbsp;&nbsp;&nbsp;&nbsp;model & layer & frames
* Compute:  
&nbsp;&nbsp;&nbsp;&nbsp;performance & tools & hardware-sensitive  
* Utils:  
&nbsp;&nbsp;&nbsp;&nbsp;data & algo & preprocess  
* Modules:  
&nbsp;&nbsp;&nbsp;&nbsp;customized module blocks/bulks for projs
* Loss:  
&nbsp;&nbsp;&nbsp;&nbsp;customized loss functions for projs
* projs:  
&nbsp;&nbsp;&nbsp;&nbsp;a complete proj needs to implement followings:
    * Dataset
    * Network
    * Trainer
    * Evaluator
    * Predictor
    * configs
    * _jobs

## Work Note:

the online working space must contain following directories:  
* `model`: consisting directory named by the `proj_name` -- save trained params
* `logs`: consisting directory named by the `proj_name`  -- save logs
* `aiml`: `git clone https://github.com/Linhengyang/aiml.git`

also recommend to have:
* `tmp`: -- temporary results to be deleted safely
* `cache`: -- to save files may be saved or not

---
    model
    ├── mlp
    ├── transformer
    ├── vit
    logs
    ├── mlp
    ├── transformer
    ├── vit
    aiml
    ├── Code
    │   ├── Base
    │   │   ├── MetaFrames
    │   │   │   ├── Architectures.py
    │   │   │   └── __init__.py
    │   │   ├── RootLayers
    │   │   │   ├── AttentionPools.py
    │   │       ├── MultiCategFeatEmbedding.py
    │   │   │   └── PositionalEncodings.py
    │   │   └── SubModules
    │   │       ├── AddLNorm.py
    │   │       └── Patchify.py
    │   ├── Compute
    │   │   ├── EvaluateTools.py
    │   │   ├── PredictTools.py
    │   │   ├── SamplingTools.py
    │   │   ├── TrainTools.py
    │   │   └── VisualizeTools.py
    │   ├── Loss
    │   │   ├── BPRankingLoss.py
    │   │   ├── L2PenaltyMSELoss.py
    │   │   └── MaskedCELoss.py
    │   ├── Modules
    │   │   ├── _recsys.py
    │   │   ├── _transformer.py
    │   │   └── _vit.py
    │   ├── Utils
    │   │   ├── Common
    │   │   │   ├── DataAssemble.py
    │   │   │   ├── DataTransform.py
    │   │   │   ├── Mask.py
    │   │   │   └── SeqOperation.py
    │   │   ├── Text
    │   │   │   ├── BytePairEncoding.py
    │   │   │   ├── TextPreprocess.py
    │   │   │   ├── Tokenize.py
    │   │       └── Vocabulize.py
    │   │   └── image
    │   │       └── PatchOperation.py
    │   ├── apps
    │   │   ├── semantic_segmentation
    │   │   └── sentiment_analysis
    │   └── projs
    │       ├── _demo
    │       ├── bert
    │       ├── gan
    │       ├── recsys
    │       ├── transformer
    │       ├── vit
    │       └── word2vec
    ├── README.md
    ├── experiment.ipynb
    ├── learn.py
    ├── main.py
    └── test.py
---