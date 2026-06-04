# Critérios de Aceitação (CA) — Experimento de pegar o copo (G1, replay do dataset real)

Objetivo: reproduzir no simulador (`run_sim_visible.py` + ZMQ) a trajetória de teleoperação
real gravada no dataset HF (`Mrwlker/pick_up_the_cup_2026-04-30`, mão DIREITA, ep18) e
fazer o robô **pegar o copo**, com o mundo simulado fiel ao real (mesa, copo, pose, nuvem de pontos).

Legenda de status: ✅ atende | ⚠️ parcial | ❌ não atende | ❓ não medido

---

## CA1 — Spawn limpo (definido pelo usuário)
Os braços do robô **NÃO** podem nascer dentro nem embaixo da mesa na inicialização.
A mão deve começar acima/à frente da mesa, como no frame inicial do episódio real.
- Causa real da falha observada: o sim **não tem compensação de gravidade**, então o braço
  **cai (sag)** abaixo do alvo. Se a mesa estiver alta (ex.: tampo 0.94), o braço caído
  afunda nela no spawn. Não é a postura — é (mesa alta) + (sag, ver CA7).
- A FK (rastreamento perfeito) engana: diz braço em Z≈0.98–1.02 (acima), mas o braço real
  cai bem mais por causa do sag → entra na mesa.
- Mitigação atual: mesa no quadril (tampo **0.86**, `mesa pos z=0`), robô reto. A validar
  VISUALMENTE no sim real (a FK não serve aqui).
- Status: ✅ com mesa no quadril (0.86) + robô reto: só um overlap mínimo transitório no
  spawn, que se assenta na posição esperada. Resolver CA7 (sag) torna robusto a qualquer altura.

## CA2 — Postura natural do robô
O robô fica **reto** (de pé), sem inclinação artificial de tronco nem deslocamento forçado
da pélvis. Pernas embaixo do corpo, pélvis na vertical.
- Status: ✅ após reverter o lean de tronco e `ROOT_X`. (torso=0, ROOT_X=0)

## CA3 — Mesa fiel ao real
Mesa com dimensões reais **1.80m (comprimento) × 0.75m (largura/profundidade)**, borda de
trás encostando no quadril do robô. Altura do tampo coerente (quadril).
- Status: ✅ dimensões/borda no quadril; ⚠️ altura do tampo ainda em ajuste (0.86–0.94).

## CA4 — Nuvem de pontos da mesa bate com a mesa simulada
A nuvem reconstruída do depth (frame do episódio certo) forma um plano **horizontal** que
coincide com o tampo da mesa simulada (em XY e altura), sem entrar no corpo do robô.
- Status: ⚠️ depende de calibração de escala/ângulo da câmera; com robô reto a nuvem da
  mesa não fica perfeitamente plana (depth 8-bit ruidoso). Câmera é fixa no torso (sem pescoço).

## CA5 — Posição do copo fiel ao real
A posição XY do copo no sim corresponde à posição real, obtida do depth do **frame correto
do episódio** (mapeado por `from_timestamp × 30`, não por offset). Raio do pixel da caneca
→ plano da mesa (a caneca branca é inválida no depth).
- Status: ⚠️ ep18 → caneca em X≈0.28–0.34, Y≈-0.02 (corroborado pelo alcance da mão).

## CA6 — Tamanho do copo compatível
O copo tem tamanho coerente com o real (Ø~13cm) OU ajustado para caber na abertura da mão
Dex3 (~12.4cm). Decidir: reduzir o copo ou aceitar o real e mudar a pega.
- Status: ❓ em aberto (real 13cm > abertura da mão 12.4cm).

## CA7 — A mão vai EM DIREÇÃO ao copo
Ao reproduzir a trajetória do dataset, a mão se move na direção do copo (não erra o alvo
por causa do "sag" do braço — o sim não tem compensação de gravidade, o robô real tinha).
- Status: ✅ (resolvido por geometria, não por kp) — com o robô **aterrado** (BAND_Z=0.847,
  pélvis ~0.79) e a mesa no quadril (0.78), a mão (que cai por gravidade) pousa naturalmente
  no nível do copo. kp alto PIORA (mão fica no alvo alto da FK, acima do copo). O sag + robô
  aterrado faz a mão ir ao copo. min_dist chegou a 4.8cm.

## CA8 — Contato com o copo
A mão encosta no copo durante a tentativa (nº de contatos > 0).
- Status: ✅ com robô aterrado + copo em [0.26, 0.01]: 2 contatos consistentes.

## CA9 — Fechamento envolvente
Os dedos fecham EM TORNO do copo (distância mínima dedo-centro < raio do copo, não tangencial).
- Status: ⚠️ melhor min_dist **4.8cm** (robô aterrado, copo [0.26,0.01]) — perto do raio, mas
  a mão empurra o copo pra frente em vez de envolver. Falta afinar posição/timing do fechamento.

## CA10 — Levanta e MANTÉM
O robô levanta o copo e o mantém levantado até o fim (elevação líquida > 4cm, copo não
escorrega de volta).
- Status: ❌ — sweep FIEL (sim real, 12 configs de escala×atrito×massa) → **0 seguraram**.
  Melhor: pico +5.0cm, líquido +2.8cm, min 3.3cm, até 5 contatos (atrito alto). A mão
  encosta (5 contatos), levanta ~5cm no pico, mas o copo ESCORREGA. Física do copo NÃO
  resolve. Causa raiz: trajetória replicada VARRE o copo (pose cintura/pélvis do episódio
  real não foi gravada). Para fechar CA10: (a) obter a pose real do corpo (fonte do dataset),
  (b) ajustar/autorar a fase final de pega (deixa de ser replay 100% fiel), ou
  (c) aceitar "encosta + levanta parcial" como demonstração.

## CA11 — Validação no SIM REAL
A avaliação final é no sim real (`run_sim_visible.py`), não em réplica/headless offline.
- Status: ✅ critério aceito. (Sweep headless serve só pra triagem; a física diverge — a
  config "vencedora" do headless deu 0 contatos no sim real.)

## CA12 — Consistência episódio ↔ depth
O frame do depth usado para localizar copo/mesa/mãos corresponde ao MESMO episódio do replay.
- Status: ✅ entendido (ep18 = frame 6023 via timestamp); cada episódio tem o copo em local diferente.

---

### Resumo do que falta (gargalos)
1. **CA7** (sag do braço) é o gargalo principal — resolver faz a mão chegar no copo.
2. **CA9/CA10** (fechamento envolvente + manter) dependem de CA6 (tamanho) + CA7.
3. **CA4** (nuvem plana) é validação visual, secundário ao grasp.
</content>
