# repo skeleton  
NOTE:  
* when a `__init__.py` file shows, that means the package imports all from underlying modules.  
* please always execute .py files under **autodl** directory in case of relative importing  
* run test.py to test module code  
* run main.py to run formal code

in Base: 
    RootLayers --> SubModules
    MetaFrames
in Modules:
    invoke components from RootLayers, SubModules --> Modules
in proj: 
    invoke components from RootLayers, SubModules and Modules --> Block
    invoke frameworks from MetaFrames                         --> Architecture
    Block + Architecture = Network

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