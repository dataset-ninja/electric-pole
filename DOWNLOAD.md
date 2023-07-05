Dataset **Electric Pole** can be downloaded in Supervisely format:

 [Download](https://assets.supervisely.com/supervisely-supervisely-assets-public/teams_storage/a/K/68/dPMCbsboaaRoHmONhQBrXnhgynPjXakizlKOhJdjIUCyJAktLfnDMP8NNGunWKtTx7utMmS8IGoqIcxb5I1Z2hlAEjTYNWnMKGAiPuLMeizuMUyX7paR1eE5uD0s.tar)

As an alternative, it can be downloaded with *dataset-tools* package:
``` bash
pip install --upgrade dataset-tools
```

... using following python code:
``` python
import dataset_tools as dtools

dtools.download(dataset='Electric Pole', dst_path='~/dtools/datasets/Electric Pole.tar')
```
The data in original format can be 🔗[downloaded here](https://universe.roboflow.com/ritsumeikan-university/electric-pole/dataset/1/download)