# Pickup Experiments - Trajetórias e Testes

Pasta com todos os scripts de trajetória e testes para pegar o copo com o G1+Dex3.

## 📋 Scripts de Geração de Trajetória

`generate_trajectory_ik.py` - Gera trajetória usando cinemática inversa com Jacobiana do MuJoCo

## 🧪 Scripts de Teste

`test_simple_elbow.py` - Teste simples: move só cotovelo (motor 25) de 0.0 → 1.5
- Status: ✅ Funciona (prova que o pipeline ZMQ → ActionReceiver → Simulator funciona)

`test_hardcoded_trajectory.py` - Trajetória hardcoded com 5 waypoints
- Waypoints: inicial → frente → descer → levantar → volta
- Status: ⚠️ Executa mas braço não faz o movimento esperado

`test_pick_cup_complete.py` - Trajetória completa 6 waypoints com controle de mão
- Waypoints: inicial → abaixar → descer → fechar mão → levantar → volta
- Dedos: índex=0.8, middle=0.8
- Status: ⚠️ Executa mas não pega o copo

`test_pick_cup_aggressive.py` - Versão mais agressiva
- shoulder_pitch = 1.5 (em vez de 1.0)
- todos_dedos = 1.0 (máximo)
- tempo_por_waypoint = 0.5s
- Status: ⚠️ Executa mas não pega o copo

## 🔄 Scripts de Teste Automático

`run_automated_pickup_test.py` - Loop automático: roda simulator + teste + análise de logs
- Roda 10 iterações
- Analisa logs para validar se pegou e levantou
- Status: ⚠️ Encontra logs mas valores inconsistentes

`run_aggressive_test.py` - Loop só com trajetória agressiva
- Roda 5 iterações com test_pick_cup_aggressive.py
- Análise mais rigorosa (dedos > 0.8)
- Status: ⚠️ Não gera logs novos

## 📖 Documentação

`TRAJECTORY_PIPELINE_EXPLANATION.md` - Explicação técnica completa do pipeline

## 🚀 Como Rodar

Todos os scripts precisam de:
- Conda environment `g1` ativado
- MuJoCo simulator rodando em background: `python run_sim.py`
- ZMQ funcionando na porta 6001

Exemplo:
```bash
conda activate g1
python unitree-g1-mujoco/run_sim.py &
sleep 3
python pickup_experiments/test_pick_cup_complete.py
```

## 🔗 Scripts de Suporte (fora desta pasta)

`unitree-g1-mujoco/run_sim.py` - Inicia simulator
`unitree-g1-mujoco/sim/action_receiver.py` - Recebe comandos ZMQ
`unitree-g1-mujoco/sim/action_logger.py` - Registra ações
`analyze_action_log.py` - Analisa logs JSONL
