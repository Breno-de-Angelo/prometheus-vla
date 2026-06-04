#!/usr/bin/env python3
import mujoco
import numpy as np
from pathlib import Path

def diagnosticar_simulacao():
    # 1. Carregar exatamente a mesma cena que seu run_sim.py usa
    xml_path = Path(__file__).parent / "unitree-g1-mujoco/assets/scene_43dof.xml"
    if not xml_path.exists():
        print(f"❌ XML não encontrado em: {xml_path}")
        return

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print("="*60)
    print("        📊 AUDITORIA GEOMÉTRICA E DE ATUADORES - G1 📊")
    print("="*60)

    # ----------------------------------------------------------------
    # AUDITORIA 1: Onde está o copo de verdade na simulação?
    # ----------------------------------------------------------------
    try:
        copo_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "objeto_customizado")
        pos_copo = data.xpos[copo_id]
        print(f"\n[COPO] ID: {copo_id} | Posição Real Inicial XYZ: ({pos_copo[0]:.4f}, {pos_copo[1]:.4f}, {pos_copo[2]:.4f})")
    except Exception as e:
        print(f"\n[COPO] ❌ Erro ao encontrar copo: {e}")
        copo_id = None

    # ----------------------------------------------------------------
    # AUDITORIA 2: Rastreamento dos Links dos Dedos (Onde eles estão?)
    # ----------------------------------------------------------------
    print("\n[DEDOS] Mapeamento de Posição Global (mjData.xpos):")
    palavras_chave_dedos = ["thumb", "index", "middle", "finger", "palm", "hand"]
    dedos_encontrados = []

    for i in range(model.nbody):
        nome_corpo = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if nome_corpo and any(p in nome_corpo.lower() for p in palavras_chave_dedos):
            pos_corpo = data.xpos[i]
            dedos_encontrados.append((nome_corpo, pos_corpo.copy(), i))
            print(f"  • Body [{i:02d}] '{nome_corpo}': XYZ = ({pos_corpo[0]:7.4f}, {pos_corpo[1]:7.4f}, {pos_corpo[2]:7.4f})")

    # ----------------------------------------------------------------
    # AUDITORIA 3: Validação dos Atuadores (Motores correspondentes)
    # ----------------------------------------------------------------
    print("\n[ACTUATORS] Verificação dos Índices de Controle da Mão:")
    print("  Verifique se os IDs que você envia via ZMQ batem com esses índices internos:")

    actuadores_mao = {}
    for i in range(model.nu):
        nome_actuator = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if nome_actuator and any(p in nome_actuator.lower() for p in ["hand", "finger", "thumb", "index", "middle"]):
            joint_id = model.actuator_trnid[i, 0]
            nome_joint = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            actuadores_mao[i] = (nome_actuator, nome_joint)
            print(f"  • Atuador ID [{i:02d}] '{nome_actuator}' -> Move junta: '{nome_joint}'")

    # ----------------------------------------------------------------
    # AUDITORIA 4: Teste Dinâmico de Transmissão (Os dedos se movem?)
    # ----------------------------------------------------------------
    print("\n[TESTE DINÂMICO] Aplicando q=1.0 virtualmente nos motores da mão...")

    # Salvar posições iniciais de todos os dedos
    pos_iniciais = {}
    for nome, pos, id_corpo in dedos_encontrados:
        pos_iniciais[nome] = (id_corpo, pos.copy())

    # Injetar comando máximo em todos os atuadores da mão detectados
    print(f"  Setando ctrl=1.0 para {len(actuadores_mao)} atuadores de mão...")
    for i in actuadores_mao.keys():
        data.ctrl[i] = 1.0

    # Avançar a física por 0.5 segundos de simulação
    for step in range(250):
        mujoco.mj_step(model, data)

    # Verificar se mudou de lugar
    print("\n  Resultado do movimento após 0.5s de simulação com cmd=1.0:")
    algum_dedo_moveu = False
    movimentos_dedos = []

    for nome_corpo, pos_inicial, id_corpo in dedos_encontrados:
        pos_final = data.xpos[id_corpo]
        dist_movida = np.linalg.norm(pos_final - pos_inicial)
        movimentos_dedos.append((nome_corpo, dist_movida, pos_inicial, pos_final))

        if dist_movida > 0.001:
            print(f"  ✅ '{nome_corpo}' MOVEU: {dist_movida*100:.2f} cm")
            print(f"     Inicial: ({pos_inicial[0]:7.4f}, {pos_inicial[1]:7.4f}, {pos_inicial[2]:7.4f})")
            print(f"     Final:   ({pos_final[0]:7.4f}, {pos_final[1]:7.4f}, {pos_final[2]:7.4f})")
            algum_dedo_moveu = True
        else:
            print(f"  ❌ '{nome_corpo}' FICOU PARADO (Delta < 1mm)")

    if not algum_dedo_moveu:
        print("\n⚠️ ALERTA CRÍTICO: Nenhum geom da mão se moveu fisicamente!")
        print("   O modelo XML pode estar sem juntas dinâmicas (fixed) ou quebrado.")

    # ----------------------------------------------------------------
    # AUDITORIA 5: Distância Euclidiana Mínima no Alvo do IK
    # ----------------------------------------------------------------
    print("\n[DISTÂNCIA] Simulando posição máxima de descida (Shoulder Pitch = 1.0):")

    # Resetar estado
    mujoco.mj_resetData(model, data)

    # Forçar o ombro no ponto máximo (shoulder_pitch=1.0, elbow=0.8)
    try:
        # Aplicar comandos conforme no teste
        data.ctrl[22] = 1.0   # shoulder_pitch máximo
        data.ctrl[25] = 0.8   # elbow estendido

        # Aplicar comando de fechamento nas mãos também
        for i in actuadores_mao.keys():
            data.ctrl[i] = 1.0

        # Sincronizar física por 1 segundo (500 steps de 0.002s)
        print("  Sincronizando posições com shoulder_pitch=1.0, elbow=0.8...")
        for _ in range(500):
            mujoco.mj_step(model, data)

        # Calcular menor distância entre dedos e o copo
        if copo_id is not None:
            pos_copo_atual = data.xpos[copo_id]
            print(f"\n  Posição do copo neste estado: ({pos_copo_atual[0]:.4f}, {pos_copo_atual[1]:.4f}, {pos_copo_atual[2]:.4f})")
            print(f"\n  Distâncias dos dedos até o copo:")

            distancias_dedos = []
            for nome_corpo, pos_inicial, id_corpo in dedos_encontrados:
                pos_dedo = data.xpos[id_corpo]
                dist = np.linalg.norm(pos_dedo - pos_copo_atual)
                distancias_dedos.append((nome_corpo, dist, pos_dedo))

                icon = "❌" if dist > 0.05 else "⚠️ " if dist > 0.01 else "✅"
                print(f"    {icon} '{nome_corpo}': {dist:.4f}m ({dist*100:.2f} cm) | Pos: ({pos_dedo[0]:7.4f}, {pos_dedo[1]:7.4f}, {pos_dedo[2]:7.4f})")

            # Encontrar dedo mais próximo
            distancia_minima = min(d[1] for d in distancias_dedos)
            dedo_mais_proximo = [d[0] for d in distancias_dedos if d[1] == distancia_minima][0]

            print(f"\n  🎯 DEDO MAIS PRÓXIMO DO COPO: '{dedo_mais_proximo}' a {distancia_minima*100:.2f} cm")

            if distancia_minima > 0.1:
                print(f"  ⚠️ PROBLEMA IDENTIFICADO: Distância mínima > 10cm!")
                print(f"     Isto sugere HIPÓTESE A/D: Mão no lugar errado geometricamente")
            elif distancia_minima > 0.01:
                print(f"  ⚠️ Distância marginal (1-10cm): Pode haver leve desalinhamento")
            else:
                print(f"  ✅ Dedos muito próximos do copo!")

    except Exception as e:
        print(f"  ❌ Falha ao computar distâncias dinâmicas: {e}")

    # ----------------------------------------------------------------
    # AUDITORIA 6: Resumo de Diagnóstico
    # ----------------------------------------------------------------
    print("\n" + "="*60)
    print("                    📋 RESUMO DE DIAGNÓSTICO")
    print("="*60)

    print(f"\n✅ Dados Coletados:")
    print(f"   • XML carregado com sucesso: {xml_path}")
    print(f"   • Bodies totais: {model.nbody}")
    print(f"   • Atuadores totais: {model.nu}")
    print(f"   • Dedos/Hand links encontrados: {len(dedos_encontrados)}")
    print(f"   • Atuadores de mão mapeados: {len(actuadores_mao)}")

    print(f"\n⚠️ Próximas Ações Recomendadas:")
    if not algum_dedo_moveu:
        print(f"   1. Abrir g1_29dof_with_hand.xml")
        print(f"   2. Verificar se as juntas da mão são do tipo 'hinge' (não 'fixed')")
        print(f"   3. Verificar se os <actuator> apontam para juntas válidas")
        print(f"   4. Hipótese C pode estar correta: modelo quebrado")
    else:
        print(f"   1. Dedos estão se movendo ✅")
        if distancia_minima > 0.05:
            print(f"   2. MAS estão longe do copo: Ajustar posição target do copo")
            print(f"   3. Hipóteses A ou D: Erro geométrico, não de controle")
        else:
            print(f"   2. Dedos perto do copo - problema pode ser de contato/fricção")

    print("\n" + "="*60)

if __name__ == "__main__":
    diagnosticar_simulacao()
