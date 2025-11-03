# ACKE

Our project is base on [easyedit](https://github.com/zjunlp/EasyEdit).

You can run our code by simply add the script in [easyedit](https://github.com/zjunlp/EasyEdit).

```
EasyEdit-main/
│
├── easyeditor/
│   ├── editors/
│   │   └── editor.py
│   │
│   └── models
│       └── ACKE/...
├── hparams/
│   └── ACKE/
│       └── llama-7b.yaml
└── run_acke.py
```

## Setup
You can create a virtual environment and install the dependencies via [Anaconda](https://www.anaconda.com).
```shell
conda create -n ACKE python=3.9.7
conda activate ACKE
pip install -r requirements.txt
```

## running
You can run the code by running the following command:
```shell
run_acke.py
```
