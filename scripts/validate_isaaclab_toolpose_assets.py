#!/usr/bin/env python3
"""Isaac Lab asset/state probe for the SimToolReal ToolPose stack.

This is a scoped Blackwell migration validator, not a training environment. It
launches Isaac Lab, converts the actual SimToolReal URDF assets, instantiates
the SHARPA robot plus a procedurally generated handle-head tool, steps physics,
and verifies that the state tensors needed by the ToolPose policy are available
on GPU.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from validate_blackwell_isaaclab import (
    SmiSampler,
    git_info,
    import_path,
    nvidia_smi_gpu,
    pip_check,
    repo_root,
    torch_probe,
    utc_now,
    write_report,
)


FINGERTIP_BODY_NAMES = [
    "left_index_DP",
    "left_middle_DP",
    "left_ring_DP",
    "left_thumb_DP",
    "left_pinky_DP",
]
PALM_BODY_NAME = "iiwa14_link_7"
OBJECT_KEYPOINT_OFFSETS = [
    [1.0, 1.0, 1.0],
    [1.0, 1.0, -1.0],
    [-1.0, -1.0, 1.0],
    [-1.0, -1.0, -1.0],
]
OBJECT_SCALE = [0.20, 0.08, 0.04]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--gpu-id", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--report-path",
        default="reports/isaaclab_toolpose_asset_probe_metrics.json",
        help="Path for the structured validation report.",
    )
    parser.add_argument(
        "--generated-asset-dir",
        default="/tmp/simtoolreal_isaaclab_assets",
        help="Directory for generated temporary URDF assets. This is outside the repo by default.",
    )
    return parser


def ast_constant(path: Path, name: str) -> Any:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == name:
                return ast.literal_eval(node.value)
    raise KeyError(f"Could not find literal constant {name!r} in {path}")


def expected_joint_names(root: Path) -> List[str]:
    utils_path = root / "isaacgymenvs/utils/observation_action_utils_sharpa.py"
    value = ast_constant(utils_path, "JOINT_NAMES_ISAACGYM")
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"Unexpected JOINT_NAMES_ISAACGYM value in {utils_path}")
    return value


def generate_tool_urdf(root: Path, generated_asset_dir: Path) -> Path:
    generated_asset_dir.mkdir(parents=True, exist_ok=True)
    generator_path = root / "isaacgymenvs/tasks/simtoolreal/generate_objects.py"
    spec = importlib.util.spec_from_file_location("simtoolreal_generate_objects", generator_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load generator module from {generator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    tool_path = generated_asset_dir / "toolpose_handle_head_blackwell_probe.urdf"
    module.generate_handle_head_urdf_constant_density(
        filepath=tool_path,
        handle_scale=(0.20, 0.025, 0.025),
        head_scale=(0.06, 0.08, 0.04),
        density=400,
    )
    return tool_path


def _quat_apply_wxyz(quat, vec):
    import torch

    q_w = quat[:, 0:1]
    q_xyz = quat[:, 1:4]
    uv = torch.cross(q_xyz, vec, dim=-1)
    uuv = torch.cross(q_xyz, uv, dim=-1)
    return vec + 2.0 * (q_w * uv + uuv)


def _build_policy_observation(robot, tool, scene, palm_ids, fingertip_ids, action_targets, object_scale):
    import torch

    num_envs = robot.num_instances
    joint_pos = robot.data.joint_pos
    joint_vel = robot.data.joint_vel
    palm_pos = robot.data.body_pos_w[:, palm_ids[0], :]
    palm_rot_wxyz = robot.data.body_quat_w[:, palm_ids[0], :]
    fingertip_pos = robot.data.body_pos_w[:, fingertip_ids, :]
    fingertip_rel = (fingertip_pos - palm_pos[:, None, :]).reshape(num_envs, -1)

    object_state = tool.data.root_state_w
    object_pos = object_state[:, :3]
    object_rot_wxyz = object_state[:, 3:7]
    goal_pos = object_pos + torch.tensor([0.0, 0.0, 0.12], device=object_pos.device)
    goal_rot_wxyz = object_rot_wxyz

    offsets = torch.tensor(OBJECT_KEYPOINT_OFFSETS, dtype=torch.float32, device=object_pos.device)
    scale = torch.tensor(object_scale, dtype=torch.float32, device=object_pos.device) * 0.5
    offsets = offsets * scale
    offsets = offsets.unsqueeze(0).expand(num_envs, -1, -1)
    obj_rotated = _quat_apply_wxyz(
        object_rot_wxyz[:, None, :].expand(num_envs, offsets.shape[1], 4).reshape(-1, 4),
        offsets.reshape(-1, 3),
    ).reshape(num_envs, offsets.shape[1], 3)
    goal_rotated = _quat_apply_wxyz(
        goal_rot_wxyz[:, None, :].expand(num_envs, offsets.shape[1], 4).reshape(-1, 4),
        offsets.reshape(-1, 3),
    ).reshape(num_envs, offsets.shape[1], 3)
    keypoints_rel_palm = (object_pos[:, None, :] + obj_rotated - palm_pos[:, None, :]).reshape(num_envs, -1)
    keypoints_rel_goal = (
        object_pos[:, None, :] + obj_rotated - (goal_pos[:, None, :] + goal_rotated)
    ).reshape(num_envs, -1)

    palm_rot_xyzw = torch.cat([palm_rot_wxyz[:, 1:4], palm_rot_wxyz[:, 0:1]], dim=-1)
    object_rot_xyzw = torch.cat([object_rot_wxyz[:, 1:4], object_rot_wxyz[:, 0:1]], dim=-1)
    object_scales = torch.tensor(object_scale, dtype=torch.float32, device=object_pos.device).repeat(num_envs, 1)
    obs = torch.cat(
        [
            joint_pos,
            joint_vel,
            action_targets,
            palm_pos - scene.env_origins,
            palm_rot_xyzw,
            object_rot_xyzw,
            keypoints_rel_palm,
            keypoints_rel_goal,
            fingertip_rel,
            object_scales,
        ],
        dim=-1,
    )
    return obs


def run_asset_probe(args: argparse.Namespace) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "success": False,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "seed": args.seed,
        "device": args.device,
    }
    root = repo_root()
    try:
        import torch

        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.sim import SimulationContext
        from isaaclab.utils import configclass

        torch.manual_seed(args.seed)

        joint_names_expected = expected_joint_names(root)
        robot_urdf = root / "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
        table_urdf = root / "assets/urdf/table_narrow.urdf"
        tool_urdf = generate_tool_urdf(root, Path(args.generated_asset_dir))
        for path in [robot_urdf, table_urdf, tool_urdf]:
            if not path.exists():
                raise FileNotFoundError(str(path))

        kuka_default = [-1.571, 1.571, -0.000, 1.376, -0.000, 1.485, 1.308]
        init_joint_pos = {name: 0.0 for name in joint_names_expected}
        for name, value in zip(joint_names_expected[:7], kuka_default):
            init_joint_pos[name] = value

        @configclass
        class SimToolRealAssetSceneCfg(InteractiveSceneCfg):
            ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
            light = AssetBaseCfg(
                prim_path="/World/Light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75)),
            )
            robot: ArticulationCfg = ArticulationCfg(
                prim_path="{ENV_REGEX_NS}/Robot",
                spawn=sim_utils.UrdfFileCfg(
                    asset_path=str(robot_urdf),
                    fix_base=True,
                    merge_fixed_joints=True,
                    force_usd_conversion=True,
                    self_collision=False,
                    collision_from_visuals=False,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=True,
                        retain_accelerations=True,
                        linear_damping=0.01,
                        angular_damping=0.01,
                        max_depenetration_velocity=1000.0,
                    ),
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                        enabled_self_collisions=False,
                        solver_position_iteration_count=8,
                        solver_velocity_iteration_count=1,
                    ),
                    joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
                        drive_type="force",
                        target_type="position",
                        gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(
                            stiffness={".*": 50.0},
                            damping={".*": 2.0},
                        ),
                    ),
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=(0.0, 0.8, 0.0),
                    rot=(1.0, 0.0, 0.0, 0.0),
                    joint_pos=init_joint_pos,
                    joint_vel={".*": 0.0},
                ),
                actuators={
                    "toolpose_position_targets": ImplicitActuatorCfg(
                        joint_names_expr=[".*"],
                        effort_limit_sim={".*": 300.0},
                        stiffness={".*": 50.0},
                        damping={".*": 2.0},
                    )
                },
                soft_joint_pos_limit_factor=1.0,
            )
            table = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Table",
                spawn=sim_utils.UrdfFileCfg(
                    asset_path=str(table_urdf),
                    fix_base=True,
                    merge_fixed_joints=True,
                    force_usd_conversion=True,
                    joint_drive=None,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.38)),
            )
            tool: RigidObjectCfg = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tool",
                spawn=sim_utils.UrdfFileCfg(
                    asset_path=str(tool_urdf),
                    fix_base=False,
                    merge_fixed_joints=True,
                    force_usd_conversion=True,
                    joint_drive=None,
                    collision_from_visuals=False,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=False,
                        linear_damping=0.01,
                        angular_damping=0.01,
                        max_depenetration_velocity=5.0,
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.63), rot=(1.0, 0.0, 0.0, 0.0)),
            )

        sim_cfg = sim_utils.SimulationCfg(dt=1 / 120, device=args.device)
        sim = SimulationContext(sim_cfg)
        sim.set_camera_view([1.6, -1.0, 1.2], [0.0, 0.2, 0.5])
        scene_cfg = SimToolRealAssetSceneCfg(
            num_envs=args.num_envs,
            env_spacing=1.2,
            replicate_physics=True,
            clone_in_fabric=True,
        )
        scene = InteractiveScene(scene_cfg)
        sim.reset()

        robot = scene["robot"]
        tool = scene["tool"]
        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] += scene.env_origins
        robot.write_root_pose_to_sim(root_state[:, :7])
        robot.write_root_velocity_to_sim(root_state[:, 7:])
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        tool_state = tool.data.default_root_state.clone()
        tool_state[:, :3] += scene.env_origins
        tool.write_root_pose_to_sim(tool_state[:, :7])
        tool.write_root_velocity_to_sim(tool_state[:, 7:])
        scene.reset()

        palm_ids, palm_names = robot.find_bodies(PALM_BODY_NAME)
        fingertip_ids, fingertip_names = robot.find_bodies(FINGERTIP_BODY_NAMES, preserve_order=True)
        action_targets = joint_pos.clone()

        start = time.perf_counter()
        sim_dt = sim.get_physics_dt()
        obs = None
        for _ in range(args.steps):
            robot.set_joint_position_target(action_targets)
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)
            obs = _build_policy_observation(
                robot,
                tool,
                scene,
                palm_ids,
                fingertip_ids,
                action_targets,
                OBJECT_SCALE,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        missing_joints = [name for name in joint_names_expected if name not in robot.joint_names]
        extra_joints = [name for name in robot.joint_names if name not in joint_names_expected]
        joint_order_matches = robot.joint_names == joint_names_expected
        missing_bodies = [
            name
            for name in [PALM_BODY_NAME] + FINGERTIP_BODY_NAMES
            if name not in robot.body_names
        ]
        obs_shape = list(obs.shape) if obs is not None else None
        action_shape = list(action_targets.shape)
        checks = {
            "robot_has_29_joints": robot.num_joints == 29,
            "all_expected_joints_present": len(missing_joints) == 0,
            "all_required_bodies_present": len(missing_bodies) == 0,
            "observation_shape_is_140": obs_shape == [args.num_envs, 140],
            "action_shape_is_29": action_shape == [args.num_envs, 29],
            "tool_state_shape_ok": list(tool.data.root_state_w.shape) == [args.num_envs, 13],
        }
        result.update(
            {
                "success": all(checks.values()),
                "elapsed_sec": elapsed,
                "throughput_env_steps_per_sec": (args.num_envs * args.steps / elapsed) if elapsed > 0 else None,
                "sim_dt": sim_dt,
                "assets": {
                    "robot_urdf": str(robot_urdf),
                    "table_urdf": str(table_urdf),
                    "generated_tool_urdf": str(tool_urdf),
                },
                "robot": {
                    "num_instances": robot.num_instances,
                    "num_joints": robot.num_joints,
                    "num_bodies": robot.num_bodies,
                    "joint_names": robot.joint_names,
                    "body_names": robot.body_names,
                    "expected_joint_names": joint_names_expected,
                    "missing_expected_joints": missing_joints,
                    "extra_joints": extra_joints,
                    "joint_order_matches_isaacgym": joint_order_matches,
                    "palm_body_ids": [int(item) for item in palm_ids],
                    "palm_body_names": palm_names,
                    "fingertip_body_ids": [int(item) for item in fingertip_ids],
                    "fingertip_body_names": fingertip_names,
                    "missing_required_bodies": missing_bodies,
                },
                "state_tensors": {
                    "joint_pos_shape": list(robot.data.joint_pos.shape),
                    "joint_vel_shape": list(robot.data.joint_vel.shape),
                    "body_pos_shape": list(robot.data.body_pos_w.shape),
                    "tool_root_state_shape": list(tool.data.root_state_w.shape),
                    "action_target_shape": action_shape,
                    "policy_observation_shape": obs_shape,
                    "policy_observation_first_row_finite": bool(torch.isfinite(obs[0]).all().item()) if obs is not None else False,
                },
                "checks": checks,
            }
        )
    except Exception as exc:
        result.update(
            {
                "success": False,
                "error": "{}: {}".format(type(exc).__name__, exc),
                "traceback": traceback.format_exc(),
            }
        )
    return result


def main() -> int:
    root = repo_root()
    os.chdir(str(root))

    parser = build_parser()
    try:
        from isaaclab.app import AppLauncher
    except Exception as exc:
        args, _ = parser.parse_known_args()
        report = {
            "status": "failed",
            "exit_code": 1,
            "message": "Isaac Lab AppLauncher import failed.",
            "error": "{}: {}".format(type(exc).__name__, exc),
            "started_at_utc": utc_now(),
            "ended_at_utc": utc_now(),
            "system": {
                "python_executable": sys.executable,
                "python_version": platform.python_version(),
            },
            "imports": {"isaaclab": import_path("isaaclab")},
        }
        write_report(root / args.report_path, report)
        print(json.dumps(report, sort_keys=True))
        return 1

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True

    started = utc_now()
    simulation_app = None
    app_error: Optional[str] = None
    probe_result: Dict[str, Any] = {}
    sampler = SmiSampler(str(args.gpu_id))
    sampler.start()
    try:
        launcher = AppLauncher(args)
        simulation_app = launcher.app
        probe_result = run_asset_probe(args)
    except Exception as exc:
        app_error = "{}: {}".format(type(exc).__name__, exc)
        probe_result = {
            "success": False,
            "error": app_error,
            "traceback": traceback.format_exc(),
        }
    finally:
        sampler.stop()

    torch_info = torch_probe(args.device)
    imports = {
        "isaaclab": import_path("isaaclab"),
        "isaacsim": import_path("isaacsim"),
        "trimesh": import_path("trimesh"),
    }
    pip = pip_check()
    smi = sampler.summary()
    probe_ok = bool(probe_result.get("success"))
    torch_ok = bool(torch_info.get("kernel_smoke_ok"))
    status = "success"
    message = "SimToolReal ToolPose assets and GPU state tensors loaded in Isaac Lab."
    exit_code = 0
    if not probe_ok or not torch_ok:
        status = "failed"
        message = "SimToolReal ToolPose Isaac Lab asset/state probe failed."
        exit_code = 1
    elif not pip.get("ok"):
        status = "success_with_dependency_conflicts"
        message = "ToolPose asset/state probe succeeded, but pip check reports dependency conflicts."

    payload: Dict[str, Any] = {
        "status": status,
        "exit_code": exit_code,
        "message": message,
        "started_at_utc": started,
        "ended_at_utc": utc_now(),
        "command": " ".join([sys.executable] + sys.argv),
        "mode": "isaaclab_toolpose_asset_probe",
        "smoke_test": True,
        "gpu_id": str(args.gpu_id),
        "num_envs": args.num_envs,
        "seed": args.seed,
        "steps": args.steps,
        "device": args.device,
        "system": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "environment_name": os.environ.get("CONDA_DEFAULT_ENV")
            or Path(os.environ.get("VIRTUAL_ENV", "")).name
            or "system",
            "git": git_info(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
        "torch": torch_info,
        "imports": imports,
        "pip_check": pip,
        "nvidia_smi_after": nvidia_smi_gpu(str(args.gpu_id)),
        "asset_probe": probe_result,
        "app_error": app_error,
        "metrics": {
            "peak_vram_mib": smi.get("peak_vram_mib"),
            "avg_gpu_util_percent": smi.get("avg_gpu_util_percent"),
            "peak_gpu_util_percent": smi.get("peak_gpu_util_percent"),
            "throughput_fps": probe_result.get("throughput_env_steps_per_sec"),
            "final_reward": None,
        },
        "gpu_samples": smi,
    }
    report_path = root / args.report_path
    write_report(report_path, payload)
    print(json.dumps(payload, sort_keys=True))
    sys.stdout.flush()
    sys.stderr.flush()
    if simulation_app is not None:
        close_mode = os.environ.get("SIMTOOLREAL_ISAACSIM_CLOSE_MODE", "force_exit")
        if close_mode == "graceful":
            try:
                simulation_app.close()
            except Exception:
                pass
        else:
            # Isaac Sim 5.1 can hang during shutdown after converting the large
            # SHARPA URDF. The report has already been written; process exit
            # lets the OS reclaim GPU resources and keeps smoke scripts bounded.
            os._exit(exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
