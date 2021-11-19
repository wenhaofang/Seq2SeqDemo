## Seq2Seq Demo

This repository includes some demo Seq2Seq models.

Note: The project refers to <https://github.com/bentrevett/pytorch-seq2seq>

### Process Data

```shell
PYTHONPATH=. python dataprocess/process.py
```

### Unit Test

* For loader

```shell
PYTHONPATH=. python loaders/loader.py
```

* For module

```shell
PYTHONPATH=. python modules/module1.py
```

### Main Process

```shell
python main.py
```

You can change the config either in the command line or in the file `utils/parser.py`
