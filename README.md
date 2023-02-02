# repo skeleton  
NOTE:  
* when a `__init__.py` file shows, that means the package imports all from underlying modules.  
* please always execute .py files under **autodl** directory in case of relative importing  
* run test.py to test module code  
* run main.py to run formal code

---
    autodl  
    ├── Code  
    │   ├── Base  
    │   │   ├── Layers  
    │   │   │   ├── AttentionPools.py  
    │   │   │   └── PositionalEncodings.py  
    │   │   ├── Metaframes  
    │   │   │   ├── __init__.py  
    │   │   │   └── Architectures.py  
    │   │   └── Tools  
    │   │       ├── DataTools.py  
    │   │       └── VisualizeTools.py  
    │   ├── Utils  
    │   │   ├── Text  
    │   │   │   └── TextPreprocess.py  
    │   │   └── image  
    │   └── projs  
    │       ├── bert  
    │       └── transformer  
    │           └── DataLoad_seq2seq.py  
    ├── Config  
    │   ├── __init__.py  
    │   └── params.py  
    ├── README.md  
    ├── main.py  
    └── test.py  
---