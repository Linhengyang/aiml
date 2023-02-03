# repo skeleton  
NOTE:  
* a `__init__.py` file shows that the package imports all from underlying modules.  
* always run .py files under `autodl` directory in case of relative importing  
* run `python -B test.py` to test code from modules  

in Base:  (avoid unnecessary modification)  
&nbsp;&nbsp;&nbsp;&nbsp;RootLayers --> SubModules  
&nbsp;&nbsp;&nbsp;&nbsp;MetaFrames  

in Modules:  (module blocks/units for project networks)  
&nbsp;&nbsp;&nbsp;&nbsp;invoke components from RootLayers, SubModules --> Modules  

in proj:  (desigend functions and networks for projects)  
&nbsp;&nbsp;&nbsp;&nbsp;invoke components from RootLayers, SubModules and Modules --> Block  
&nbsp;&nbsp;&nbsp;&nbsp;invoke frameworks from MetaFrames                         --> Architecture  
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
    │   │       └── VisualizeTools.py
    │   ├── Loss
    │   ├── Modules
    │   │   └── _transformer.py
    │   ├── Optimizer
    │   ├── Utils
    │   │   ├── Common
    │   │   │   └── SeqOperations.py
    │   │   └── Text
    │   │       └── TextPreprocess.py
    │   └── projs
    │       └── bert
    │       └── transformer
    │           ├── DataLoad_seq2seq.py
    │           └── network.py
    ├── Config
    │   ├── __init__.py
    │   └── params.py
    ├── README.md
    ├── main.py
    └── test.py
---