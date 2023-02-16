# repo skeleton  
## Execute Note:  
* a `__init__.py` file shows that the package imports all from underlying modules.  
* always run `python -B xxx.py` file under `autodl` directory in case of relative importing  

## Code Note:
* Base:  
&nbsp;&nbsp;&nbsp;&nbsp;model & tool  
* Compute:  
&nbsp;&nbsp;&nbsp;&nbsp;performance & hardware-sensitive  
* Utils:  
&nbsp;&nbsp;&nbsp;&nbsp;data & algo  
* Modules:  
&nbsp;&nbsp;&nbsp;&nbsp;customized module blocks/bulks for project network  
* Loss:  
&nbsp;&nbsp;&nbsp;&nbsp;customized loss functions for project loss  
* proj:  
&nbsp;&nbsp;&nbsp;&nbsp;designed networks/datasets/trainers for projects  

## Work Note:

the online working space must contain following directories:  
* model: consisting directory named by the proj_name which will save trained params of the proj
* logs: consisting directory named by the proj_name which will save logs of the proj
* autodl: git clone https://github.com/Linhengyang/autodl.git
---
    model
    ├── transformer
    logs
    ├── transformer
    autodl
    ├── Code
    │   ├── Base
    │   │   ├── MetaFrames
    │   │   │   ├── Architectures.py
    │   │   │   └── __init__.py
    │   │   ├── RootLayers
    │   │   │   ├── AttentionPools.py
    │   │   │   └── PositionalEncodings.py
    │   │   ├── SubModules
    │   │   │   └── AddLNorm.py
    │   │   └── Tools
    │   │       ├── DataTools.py
    │   │       ├── EvaluateTools.py
    │   │       └── VisualizeTools.py
    │   ├── Compute
    │   │   └── Trainers.py
    │   ├── Loss
    │   │   └── MaskedCELoss.py
    │   ├── Modules
    │   │   └── _transformer.py
    │   ├── Utils
    │   │   ├── Common
    │   │   │   └── SeqOperations.py
    │   │   └── Text
    │   │       └── TextPreprocess.py
    │   └── projs
    │       ├── mlp
    │       └── transformer
    │           ├── Dataset.py
    │           ├── Network.py
    │           ├── Predictor.py
    │           ├── Trainer.py
    │           ├── settings.py
    │           └── note.txt
    ├── Config
    │   ├── __init__.py
    │   └── params.py
    ├── README.md
    ├── main.py
    ├── nettester.py
    └── test.py
---