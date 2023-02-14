# repo skeleton  
NOTE:  
* a `__init__.py` file shows that the package imports all from underlying modules.  
* always run .py files under `autodl` directory in case of relative importing  
* run `python -B test.py` to test code from modules  

in Base:  (general codes relying only official packages. avoid unnecessary modification)  
model & data related  
&nbsp;&nbsp;&nbsp;&nbsp;RootLayers --> SubModules  
&nbsp;&nbsp;&nbsp;&nbsp;MetaFrames  

in Utils:  (general codes relying only official packages. avoid unnecessary modification)  
logic & utility related  

in Modules:  (customized module blocks/units for project networks)  
&nbsp;&nbsp;&nbsp;&nbsp;invoke components from RootLayers, SubModules --> Modules  

in Loss:  (customized loss functions for project train)  

in Optimizer:  (customized optimizer for project train)  

in Utils:  (general codes relying only official packages. avoid unnecessary modification)  

in proj:  (designed functions and networks for projects)  
&nbsp;&nbsp;&nbsp;&nbsp;invoke components from RootLayers(Base), SubModules(Base) --> Blocks (Modules)  
&nbsp;&nbsp;&nbsp;&nbsp;invoke frameworks from MetaFrames(Base)                   --> Architecture  
&nbsp;&nbsp;&nbsp;&nbsp;Block + Architecture = Network  

---
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
    │   │   ├── Trainers.py
    │   │   └── __init__.py
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
    │       ├── bert
    │       │   └── __init__.py
    │       └── transformer
    │           ├── Dataset.py
    │           ├── Network.py
    │           ├── Predictor.py
    │           ├── Trainer.py
    │           └── note.txt
    ├── Config
    │   ├── __init__.py
    │   └── params.py
    ├── README.md
    ├── main.py
    ├── nettester.py
    └── test.py
---