# ⚠️ Conflito de import `datasets` (CARMEN × LeRobot)

## Sintoma

Ao rodar qualquer comando do LeRobot (`lerobot-info`, `lerobot-record`, etc), ocorre erro como: 

`ModuleNotFoundError: No module named 'datasets.utils'`

### No ~/.bashrc na linha do NOD_Tracker, você precisa comentar essa linha que da export para não puxar o aquivo errado em vez do que o lerobot precisa.

```
export PYTHONPATH=$CARMEN_HOME/src/neural_object_detector3/pedestrian_tracker:$PYTHONPATH
```