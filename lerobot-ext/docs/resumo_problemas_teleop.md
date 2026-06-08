# Histórico de Problemas e Soluções (Teleoperação G1)

Este documento resume as principais reclamações, bugs enfrentados e as soluções aplicadas durante a sessão de debug e otimização do sistema de teleoperação VR (Vuer) com o robô Unitree G1.

## 1. Problema de Teleoperação "Travada" / Não-Fluida
* **Reclamação:** A teleoperação estava extremamente "truncada" e a culpa estava sendo atribuída injustamente à rede ou à carga de CPU.
* **Solução (pelo Usuário):** O problema era puramente de configuração embarcada do Unitree. A adição e configuração da flag `G1_RAMP_ONBOARD=0` desativou o "soft-start" embarcado, permitindo que a teleoperação fluísse em tempo real sem resistência do robô.

## 2. Lentidão Extrema e Delay nas Câmeras (Até 3s de atraso)
* **Reclamação:** A câmera começava a atrasar severamente, especialmente após iniciar a contagem de "frames no buffer", chegando a 1 ou até 3 segundos de atraso no Quest.
* **Diagnóstico e Soluções aplicadas:**
  1. **Saturação de Rede Gigabit:** O sistema original abria três conexões diretas ao robô (`LeRobot dataset`, `OmniView` e `Vuer VR`), puxando 3x60MB/s = 180MB/s e saturando o cabo Gigabit (limite de ~125MB/s).
     * *Correção:* Criado um **ZMQ Proxy Local (Forwarder)** (`127.0.0.1:5555`) no `init_lerobot_record_v2.py` para puxar apenas 1 stream (60MB/s) e distribuir localmente.
  2. **Hardcode de IP no VR:** O script `xr_g1_arm.py` ignorava o proxy local pois tinha a linha `self.config.img_server_ip = "192.168.123.164"` "chumbada" no código. 
     * *Correção:* O hardcode foi removido para forçar o VR a ler do Proxy local.
  3. **ZMQ Buffering (Efeito Cebola):** O proxy ZMQ e os sockets TCP do Linux empilhavam frames antigos na memória se houvesse qualquer micro-engasgo, gerando o atraso contínuo.
     * *Correção:* Adicionada a flag `zmq.CONFLATE = 1` aos sockets `SUB` e `PUB` do proxy para dropar mensagens velhas e garantir 0 buffers.
  4. **Processos Zumbis:** Havia múltiplas instâncias ocultas (zumbis) do `init_lerobot_record_v2.py` rodando em background após fechamentos forçados (Ctrl+C), competindo pela rede.
     * *Correção:* Limpeza via `kill -9` e recomendação de usar `pgrep -af init_lerobot` para checagem.

## 3. Episódios Curtos (1.3s) e Corrompidos
* **Reclamação:** Episódios gravados apareciam com apenas 1.3s, e ao abrir não continham os dados esperados.
* **Causa:** O encerramento do script (sair da gravação) matava o processo de forma bruta antes que as filas (queues) assíncronas do LeRobot terminassem de codificar os vídeos (mp4) e salvar os arquivos parquet.
* **Solução:** Foi aplicado um patch no método `finalize()` do `LeRobotDataset` para escutar e esperar ativamente (`_drain_save_queue`) o término das tarefas em background antes de permitir que o programa encerre.

## 4. O OmniView Não Mostrava as Imagens (Tela Preta) ou Mostrava Só Uma Câmera
* **Reclamação:** O dataset salvo não exibia imagens RGB no web viewer, e na versão "Ao Vivo" apenas uma das câmeras aparecia, ignorando a outra.
* **Soluções:**
  * **No visualizador de Datasets:** O OmniView original tentava criar "symlinks" em sistemas/arquiteturas incompatíveis, resultando em links quebrados. Foi substituído pelo `shutil.copy` para mover os assets com segurança.
  * **No Live OmniView:** O dicionário que guardava o frame das câmeras estava sendo sobrescrito (`LATEST["img"] = out`) a cada nova mensagem (alternando entre RGB e Depth muito rápido). Corrigido para `LATEST["img"].update(out)` para manter as duas câmeras na tela simultaneamente e resolvido um bug de `NoneType` na inicialização do dashboard.

---
*Este documento foi gerado a pedido do usuário para fins de registro histórico das batalhas travadas (e vencidas) durante o desenvolvimento.*
