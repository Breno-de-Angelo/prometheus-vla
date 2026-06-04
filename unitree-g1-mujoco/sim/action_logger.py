"""
Logger para debugar fluxo de ações distribuído.
Salva em arquivo: comandos recebidos, posições das juntas, etc.
"""
import json
import time
from pathlib import Path
from datetime import datetime


class ActionLogger:
    def __init__(self, log_dir: str = "/tmp", verbose: bool = False):
        """
        Args:
            log_dir: diretório para salvar logs
            verbose: se True, também printa no console
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Timestamp do início da sessão
        self.session_start = datetime.now().isoformat()
        self.log_file = self.log_dir / f"action_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

        self.verbose = verbose
        self.step_count = 0

        self._log_event("session_start", {"timestamp": self.session_start})

        if verbose:
            print(f"[ActionLogger] Logging para: {self.log_file}")

    def _log_event(self, event_type: str, data: dict):
        """Log um evento em formato JSONL (um JSON por linha)."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "step": self.step_count,
            **data
        }

        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            print(f"[ActionLogger] Erro ao salvar log: {e}")

    def log_action_received(self, payload: dict):
        """Log uma ação recebida do Atenas."""
        self.step_count += 1

        body_motors = payload.get("body_motors", [])
        left_hand = payload.get("left_hand", [])
        right_hand = payload.get("right_hand", [])

        data = {
            "motor_count": len(body_motors),
            "body_motors": body_motors[:3],  # Primeiros 3 apenas
            "has_left_hand": len(left_hand) > 0,
            "has_right_hand": len(right_hand) > 0,
        }

        self._log_event("action_received", data)

        if self.verbose:
            print(f"[Step {self.step_count}] Ação: {len(body_motors)} motores")

    def log_joint_state(self, joint_positions: dict):
        """Log posição atual de todas as juntas."""
        self._log_event("joint_state", {
            "positions": joint_positions,
            "num_joints": len(joint_positions)
        })

    def log_motor_command(self, motor_idx: int, q_target: float, q_actual: float, kp: float, kd: float):
        """Log comando de um motor individual."""
        self._log_event("motor_cmd", {
            "idx": motor_idx,
            "q_target": float(q_target),
            "q_actual": float(q_actual),
            "kp": float(kp),
            "kd": float(kd),
            "error": float(abs(q_actual - q_target))
        })

    def log_bridge_state(self, bridge):
        """Log estado completo do unitree_bridge."""
        if bridge is None or bridge.low_cmd is None:
            return

        body_motors = []
        hand_motors = []

        # Body motors
        for i in range(min(bridge.num_body_motor, 29)):
            body_motors.append({
                "idx": i,
                "q": float(bridge.low_cmd.motor_cmd[i].q),
                "kp": float(bridge.low_cmd.motor_cmd[i].kp),
                "kd": float(bridge.low_cmd.motor_cmd[i].kd),
            })

        # Right hand motors
        if hasattr(bridge, 'right_hand_cmd') and bridge.right_hand_cmd:
            for i in range(min(7, len(bridge.right_hand_cmd.motor_cmd))):
                hand_motors.append({
                    "idx": i,
                    "q": float(bridge.right_hand_cmd.motor_cmd[i].q),
                    "kp": float(bridge.right_hand_cmd.motor_cmd[i].kp),
                    "kd": float(bridge.right_hand_cmd.motor_cmd[i].kd),
                })

        self._log_event("motor_cmd", {
            "body_motors": body_motors,
            "hand_motors": hand_motors
        })

    def log_physics_state(self, simulator):
        """Log estado da física (posição do copo, dedos, contatos)."""
        if simulator is None:
            return

        try:
            import mujoco

            # BaseSimulator has sim_env attribute containing DefaultEnv
            env = simulator.sim_env if hasattr(simulator, 'sim_env') else simulator

            if not hasattr(env, 'mj_data') or not hasattr(env, 'cup_qpos_adr'):
                return

            mj_data = env.mj_data
            mj_model = env.mj_model

            # Cup position (free joint has 7 DOF: x,y,z position + w,x,y,z quaternion)
            cup_pos = mj_data.qpos[env.cup_qpos_adr:env.cup_qpos_adr+3]
            cup_quat = mj_data.qpos[env.cup_qpos_adr+3:env.cup_qpos_adr+7]

            # Cup linear velocity
            cup_lvel = mj_data.qvel[env.cup_dof_adr:env.cup_dof_adr+3]

            # Right hand finger positions
            finger_positions = {}
            finger_names = [
                "right_hand_thumb_2_link",
                "right_hand_middle_1_link",
                "right_hand_index_1_link",
                "left_hand_thumb_2_link",
                "left_hand_middle_1_link",
                "left_hand_index_1_link"
            ]

            for fname in finger_names:
                try:
                    body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, fname)
                    if body_id >= 0:
                        pos = mj_data.xpos[body_id]
                        finger_positions[fname] = [float(p) for p in pos]
                except:
                    pass

            # Detect contacts between hand and cup
            contacts = []
            cup_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "objeto_customizado")

            for i in range(mj_data.ncon):
                contact = mj_data.contact[i]
                # Check if contact involves cup and hand
                if (contact.geom1 >= 0 and contact.geom2 >= 0):
                    g1 = mj_model.geom(contact.geom1)
                    g2 = mj_model.geom(contact.geom2)

                    # Rough check: if one geom is on cup body
                    if "right_hand" in (mj_model.body(g1.bodyid).name if g1.bodyid >= 0 else "") or \
                       "right_hand" in (mj_model.body(g2.bodyid).name if g2.bodyid >= 0 else ""):
                        if "objeto" in (mj_model.body(g1.bodyid).name if g1.bodyid >= 0 else "") or \
                           "objeto" in (mj_model.body(g2.bodyid).name if g2.bodyid >= 0 else ""):
                            contacts.append({
                                "geom1": mj_model.geom(contact.geom1).name if contact.geom1 >= 0 else "unknown",
                                "geom2": mj_model.geom(contact.geom2).name if contact.geom2 >= 0 else "unknown",
                                "distance": float(contact.dist),
                                "force": float(contact.solref[0]) if contact.solref[0] > 0 else 0
                            })

            # DEBUG: root pos + qpos real do braco direito (qpos 36-42)
            root_pos = [float(x) for x in mj_data.qpos[:3]]
            arm_qpos = [float(mj_data.qpos[36 + k]) for k in range(7)]

            data = {
                "cup_position": [float(x) for x in cup_pos],
                "cup_quaternion": [float(x) for x in cup_quat],
                "cup_linear_velocity": [float(x) for x in cup_lvel],
                "cup_height": float(cup_pos[2]),
                "finger_positions": finger_positions,
                "hand_cup_contacts": contacts,
                "num_contacts": len(contacts),
                "root_pos": root_pos,
                "arm_qpos": arm_qpos,
            }

            self._log_event("physics_state", data)
        except Exception as e:
            pass  # Silent fail - physics state is optional

    def print_summary(self):
        """Printa resumo do log."""
        print(f"\n[ActionLogger] Log salvo em: {self.log_file}")
        print(f"[ActionLogger] Total de eventos: {self.step_count}")
        print("\nPara analisar o log:")
        print(f"  cat {self.log_file} | jq .")
        print(f"  tail -20 {self.log_file} | jq .")
