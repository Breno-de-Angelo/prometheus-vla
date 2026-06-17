2026-06-17 (criado) · **PROPOSTA — a discutir com o Luiz, NÃO implementado**

> **O que é:** o porquê e o escopo de um **eval de loop FECHADO** + o teste complementar **"copo deslocado sintético"** pro pi05/G1-grasp. É documento de alinhamento — apresenta opções, **não decide** design (critérios/métrica/sim são pra fechar junto antes de codar).

---

# Eval de loop fechado + copo deslocado sintético

## 1. Por que isto existe (o diagnóstico que levou aqui)

Rodamos probes **estáticos** (offline, frames de treino memorizados) da MÃO e do BRAÇO no `best` do run4a (state-dropout). Achados:

- **Mão (squeeze):** aterrada na imagem (sensib. à imagem **0,72**) e crava a decisão. **Mas** o fallback sem-visão é **FECHAR** (tampar a imagem empurra o squeeze pra fechado) → o "age cego / acha que segura" do robô.
- **Braço (reach):** o **ALVO** do reach (último passo do chunk) **responde à imagem** (sensib. **0,645** no run4a, 0,446 no armstate7) e **TRACKA** o copo — **mas FRACO**: correlação(deslocamento-do-copo, mudança-do-braço) ≈ **+0,26**; copo trocado mais longe faz o braço mudar ~**1,3×** mais que perto. O 1º passo (commit) é quase cego (0,12) porque a ação imediata é dominada por continuidade; a influência da imagem aparece **downstream**, no alvo.

**Conclusão:** o braço **não ignora** o copo — ele **tracka fraco e ruidoso**. Logo o **"desce em cima do copo"** e o **"não re-reach"** observados no robô são mais consistentes com **distribution-shift de loop FECHADO** do que com ausência de grounding:

> Em **open-loop** (os probes) o state é sempre o ground-truth do dataset → o braço lê a imagem e mira ~no copo. No **robô (closed-loop)** os erros por passo **compõem**, o state deriva pra **OOD**, e o tracking **fraco demais (corr 0,2)** não consegue puxar a trajetória de volta → aterrissa fora, em cima.

Os probes estáticos **não capturam** essa degradação (porque lá o state nunca deriva). Daí a necessidade de medir em loop fechado.

## 2. O que o eval de loop fechado mediria

Rodar a política num **SIM que REAGE às próprias ações do modelo** (não replay do dataset) e medir a **deriva** ao longo do episódio — a parte que o open-loop esconde.

Métricas candidatas (**a discutir qual/quais**):
- **Distância cartesiana garra↔copo** ao longo do tempo (precisa FK do braço + pose do copo no sim).
- **Taxa de "aterrissa em cima"** / colisão por cima do copo.
- **Deriva acumulada do state** vs a distribuição das demos (quando o state sai do envelope de treino).
- **Sucesso de pega** end-to-end como âncora.

## 3. Teste complementar "copo deslocado sintético" (o item **b**)

O probe de braço atual troca a **imagem inteira** (cup position **+** fundo **+** braço-no-quadro mudam juntos) → por isso a correlação do tracking é **ruidosa** (0,2). O teste limpo:

> **Mover só o copo** na imagem — recortar+colar o copo numa nova posição (2D), ou re-renderizar a cena no sim com o copo deslocado — e medir se o **alvo do reach acompanha** o deslocamento. Isola a **posição do copo** das outras mudanças de imagem → mede "o braço **servoa** pro copo?" sem o ruído.

É o probe `(b)` que ficou pendente (junto do `(c)` distância garra↔copo no fechamento). Pode rodar **offline** (não precisa do sim de loop fechado), então é o passo mais barato pra **afiar** o número de tracking.

## 4. Como cada peça se conecta ao conserto (OPÇÕES — não decidir aqui)

| Peça | O que faz | Prós | Contras |
|---|---|---|---|
| **Copo deslocado sintético** (offline) | mede o tracking limpo (corr real) | barato, sem treino, sem sim novo | colar 2D é aproximado; render no sim é mais fiel porém mais trabalho |
| **Eval de loop fechado** (sim) | **mede** a deriva que quebra no robô | prova a hipótese do compounding-drift | precisa de sim fiel + FK + pose do copo |
| **DAgger** (treinar nos próprios rollouts) | **ataca** o compounding-drift na raiz | conserto principiado do closed-loop | caro (loop de coleta+retreino); precisa do sim |
| **Grounding loss / aug de posição** | reforça o tracking (corr 0,2→↑) | direto no que o probe mede | mexe no treino; efeito no closed-loop incerto |

Ordem natural (**a confirmar**): copo-sintético (mede limpo) → eval loop-fechado (mede a deriva) → escolher conserto (DAgger e/ou grounding) com base no que os dois mostrarem.

## 5. Estado / pendências (tudo **a discutir**)

- **Infra que já temos:** sim no laptop; `offline_sim_host.py` (replay de episódio por ZMQ, sem robô); recorder/probes na Atena. **Falta definir com o Luiz:** qual sim usar pro loop fechado (fidelidade câmera/depth vs custo), a métrica exata, e o escopo (1 posição modal vs varrer posições).
- **Decisões de design em aberto (NÃO inventar):** método do copo-sintético (colagem 2D vs render no sim), critério de "sucesso/deriva", e se vale ir pra DAgger agora ou só medir primeiro.
- **Relacionado:** probes da mão (`docs/diagnostico_treino_grasp/PROBE_RUN4a.html`) e do braço (`docs/diagnostico_treino_grasp/PROBE_BRACO_REACH.html`), diagnóstico do grasp (`docs/diagnostico_treino_grasp/`).

> **Próximo passo:** discutir com o Luiz qual peça atacar primeiro e fechar as decisões de design acima **antes** de escrever código.
