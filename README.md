repo skeleton  
note:  
    when a __init__.py file shows, tha means the package imports all from underlying modules.  
    please always run .py files under autodl directory incase of relative import  
    test.py to test module code  
    main.py to run code  
  
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