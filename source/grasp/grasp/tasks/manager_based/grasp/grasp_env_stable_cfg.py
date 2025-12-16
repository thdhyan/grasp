# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Spot Grasp Environment Configuration (Stable Variant)

Complete standalone environment configuration based on env.yaml with stabilization improvements.
"""

import math
from pathlib import Path

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, CameraCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.actuators import DelayedPDActuatorCfg, RemotizedPDActuatorCfg

import isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp as spot_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

import isaaclab.envs.mdp as base_mdp
import grasp.tasks.manager_based.grasp.mdp as local_mdp

##
# Pre-defined paths
##
HERE = Path(__file__).parent
# Go from tasks/manager_based/grasp -> tasks/manager_based -> tasks -> grasp -> assets/robots
ASSETS_DIR = HERE.parent.parent.parent / "assets" / "robots"
POLICY_DIR = ASSETS_DIR / "spot_policy"


##
# Terrain Configuration
##
COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=9,
    num_cols=21,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.05), noise_step=0.02, border_width=0.25
        ),
    },
)


##
# Robot Configuration with Cameras
##
SPOT_ARM_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(ASSETS_DIR / "spot_arm.usda"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.7),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "[fh]l_hx": 0.1,
            "[fh]r_hx": -0.1,
            "f[rl]_hy": 0.9,
            "h[rl]_hy": 1.1,
            ".*_kn": -1.5,
            "arm0_sh1": -3.13,
            "arm0_el0": 3.13,
            "arm0_el1": 0.0,
            "arm0_sh0": 0.0,
            "arm0_wr0": 0.0,
            "arm0_wr1": 0.0,
            "arm0_f1x": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "spot_hip": DelayedPDActuatorCfg(
            joint_names_expr=[".*_h[xy]"],
            effort_limit=45.0,
            stiffness=60.0,
            damping=5.0, # Increased damping
            friction=0.05,
            min_delay=0,
            max_delay=4,
        ),
        "spot_knee": DelayedPDActuatorCfg(
            joint_names_expr=[".*_kn"],
            effort_limit=45.0,
            stiffness=60.0,
            damping=5.0, # Increased damping
            friction=0.05,
            min_delay=0,
            max_delay=4,
        ),
        "spot_sh1": DelayedPDActuatorCfg(
            joint_names_expr=["arm0_sh1"],
            effort_limit=181.8,
            stiffness=60.0,
            damping=5.0, # Increased damping
            friction=0.05,
            min_delay=0,
            max_delay=4,
        ),
        "spot_el0": DelayedPDActuatorCfg(
            joint_names_expr=["arm0_el0"],
            effort_limit=90.9,
            stiffness=60.0,
            damping=5.0, # Increased damping
            friction=0.05,
            min_delay=0,
            max_delay=4,
        ),
        "spot_el1": DelayedPDActuatorCfg(
            joint_names_expr=["arm0_el1"],
            effort_limit=30.3,
            stiffness=60.0,
            damping=5.0, # Increased damping
            friction=0.05,
            min_delay=0,
            max_delay=4,
        ),
        "spot_sh0": DelayedPDActuatorCfg(
            joint_names_expr=["arm0_sh0"],
            effort_limit=90.3,
            stiffness=60.0,
            damping=5.0, # Increased damping
            friction=0.05,
            min_delay=0,
            max_delay=4,
        ),
        "spot_wr0": DelayedPDActuatorCfg(
            joint_names_expr=["arm0_wr0"],
            effort_limit=30.3,
            stiffness=60.0,
            damping=5.0, # Increased damping
            friction=0.05,
            min_delay=0,
            max_delay=4,
        ),
        "spot_wr1": DelayedPDActuatorCfg(
            joint_names_expr=["arm0_wr1"],
            effort_limit=30.3,
            stiffness=60.0,
            damping=5.0, # Increased damping
            friction=0.05,
            min_delay=0,
            max_delay=4,
        ),
        "spot_f1x": DelayedPDActuatorCfg(
            joint_names_expr=["arm0_f1x"],
            effort_limit=15.32,
            stiffness=60.0,
            damping=5.0, # Increased damping
            friction=0.05,
            min_delay=0,
            max_delay=4,
        ),
    },
)


##
# Scene Configuration
##
@configclass
class SpotGraspStableSceneCfg(InteractiveSceneCfg):
    """Scene configuration for Spot Grasp Stable environment with cameras."""

    # Ground/Terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=COBBLESTONE_ROAD_CFG,
        max_init_terrain_level=COBBLESTONE_ROAD_CFG.num_rows - 1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=True,
    )

    # Robot
    robot: ArticulationCfg = SPOT_ARM_CFG

    # Contact forces sensor
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.002,
        history_length=3,
        debug_vis=False,
        track_air_time=True,
        force_threshold=1.0,
    )

    # Sky light
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAACLAB_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# Action Configuration
##
@configclass
class SpotActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.2,
        use_default_offset=True,
    )


##
# Command Configuration
##
@configclass
class SpotCommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
        ),
    )


##
# Observation Configuration
##
@configclass
class SpotObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.5, n_max=0.5),
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


##
# Event Configuration
##
@configclass
class SpotEventCfg:
    """Configuration for randomization events."""

    # Startup events
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="body"),
            "mass_distribution_params": (-2.5, 2.5),
            "operation": "add",
        },
    )

    # Reset events
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="body"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=spot_mdp.reset_joints_around_default,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    reset_arm_joints = EventTerm(
        func=base_mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5), 
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names="arm.*"),
        },
    )

    # Apply force to end effector (arm0_f1x is the finger link/end effector area)
    apply_ee_force = EventTerm(
        func=base_mdp.apply_external_force_torque,
        mode="step",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="arm0_link_wr1"),
            "force_range": (0.0, 0.0), 
            "torque_range": (0.0, 0.0),
        },
    )

    # Interval events
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # Reduced push velocity
        },
    )


##
# Reward Configuration
##
@configclass
class SpotRewardsCfg:
    """Reward terms for the MDP."""

    # Task rewards
    air_time = RewardTermCfg(
        func=spot_mdp.air_time_reward,
        weight=5.0,
        params={
            "mode_time": 0.3,
            "velocity_threshold": 0.5,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )
    base_angular_velocity = RewardTermCfg(
        func=spot_mdp.base_angular_velocity_reward,
        weight=5.0,
        params={"std": 2.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    base_linear_velocity = RewardTermCfg(
        func=spot_mdp.base_linear_velocity_reward,
        weight=5.0,
        params={
            "std": 1.0,
            "ramp_rate": 0.5,
            "ramp_at_vel": 1.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    foot_clearance = RewardTermCfg(
        func=spot_mdp.foot_clearance_reward,
        weight=0.75,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )
    gait = RewardTermCfg(
        func=spot_mdp.GaitReward,
        weight=15.0,
        params={
            "std": 0.1,
            "max_err": 0.2,
            "velocity_threshold": 0.5,
            "synced_feet_pair_names": (("fl_foot", "hr_foot"), ("fr_foot", "hl_foot")),
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )

    # Penalties
    joint_deviation_hip = RewardTermCfg(
        func=mdp.joint_deviation_l1,
        weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "arm0_sh1", "arm0_el0", "arm0_el1", "arm0_sh0",
                    "arm0_wr0", "arm0_wr1", "arm0_f1x",
                ],
            )
        },
    )
    action_smoothness = RewardTermCfg(
        func=spot_mdp.action_smoothness_penalty,
        weight=-0.1, # Reduced from -1.0 per user request
    )
    air_time_variance = RewardTermCfg(
        func=spot_mdp.air_time_variance_penalty,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    base_motion = RewardTermCfg(
        func=spot_mdp.base_motion_penalty,
        weight=-3.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    base_orientation = RewardTermCfg(
        func=spot_mdp.base_orientation_penalty,
        weight=-1.0, # Adjusted per user request
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    base_height_l2 = RewardTermCfg(
        func=base_mdp.base_height_l2,
        weight=-1.0, # Reduced penalty
        params={
            "target_height": 0.75,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    foot_slip = RewardTermCfg(
        func=spot_mdp.foot_slip_penalty,
        weight=-0.75,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 1.0,
        },
    )
    joint_acc = RewardTermCfg(
        func=spot_mdp.joint_acceleration_penalty,
        weight=-1.0e-4,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_h[xy]", "arm0_sh1", "arm0_el0", "arm0_el1",
                    "arm0_sh0", "arm0_wr0", "arm0_wr1", "arm0_f1x",
                ],
            )
        },
    )
    joint_pos = RewardTermCfg(
        func=spot_mdp.joint_position_penalty,
        weight=-0.7,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.5,
        },
    )
    joint_torques = RewardTermCfg(
        func=spot_mdp.joint_torques_penalty,
        weight=-5.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    joint_vel = RewardTermCfg(
        func=spot_mdp.joint_velocity_penalty,
        weight=-1.0e-2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_h[xy]", "arm0_sh1", "arm0_el0", "arm0_el1",
                    "arm0_sh0", "arm0_wr0", "arm0_wr1", "arm0_f1x",
                ],
            )
        },
    )
    
    # Survival reward to encourage longer episodes
    is_alive = RewardTermCfg(
        func=mdp.is_alive,
        weight=100.0, # Heavily boosted for stability
    )

    illegal_contacts_penalty = RewardTermCfg(
        func=mdp.undesired_contacts,
        weight=-100.0, # Increased penalty
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["body", ".*leg"]),
            "threshold": 5.0,
        },
    )

    termination_penalty = RewardTermCfg(
        func=mdp.is_terminated,
        weight=-200.0, # Increased penalty
    )  


##
# Termination Configuration
##
@configclass
class SpotTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    body_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["body", ".*leg"]),
            "threshold": 10.0,
        },
    )
    
    terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )


##
# Curriculum Configuration
##
@configclass
class SpotCurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    
    # Curriculum terms for arm noise and force commented out until validated functions found
    # arm_joint_noise = CurrTerm(...)
    # ee_force_curriculum = CurrTerm(...)
    # arm_pos_curriculum = CurrTerm(...)


##
# Main Environment Configuration
##
@configclass
class SpotGraspStableEnvCfg(ManagerBasedRLEnvCfg):
    """
    Complete standalone configuration for Spot Grasp Environment (Stable Variant).
    
    Based on env.yaml with cameras added for body and wrist.
    Pre-trained policy can be loaded from: {POLICY_DIR}/spot_arm_policy.pt
    """

    # Scene
    scene: SpotGraspStableSceneCfg = SpotGraspStableSceneCfg(num_envs=4096, env_spacing=2.5)

    # Basic settings
    observations: SpotObservationsCfg = SpotObservationsCfg()
    actions: SpotActionsCfg = SpotActionsCfg()
    commands: SpotCommandsCfg = SpotCommandsCfg()

    # MDP settings
    rewards: SpotRewardsCfg = SpotRewardsCfg()
    terminations: SpotTerminationsCfg = SpotTerminationsCfg()
    events: SpotEventCfg = SpotEventCfg()
    curriculum: SpotCurriculumCfg = SpotCurriculumCfg()

    # Viewer
    viewer = ViewerCfg(
        eye=(10.5, 10.5, 0.3),
        lookat=(0.0, 0.0, 0.0),
        origin_type="world",
        env_index=0,
        asset_name="robot",
    )

    def __post_init__(self):
        """Post initialization."""
        # General settings
        self.decimation = 10  # 50 Hz control
        self.episode_length_s = 20.0
        self.seed = 42

        # Simulation settings
        self.sim.dt = 0.002  # 500 Hz physics
        self.sim.render_interval = self.decimation
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "multiply"
        self.sim.physics_material.restitution_combine_mode = "multiply"

        # Update sensor period to match physics dt
        self.scene.contact_forces.update_period = self.sim.dt

        # No height scanner for this environment
        self.scene.height_scanner = None


##
# Flat Terrain Variant
##
@configclass
class SpotGraspStableEnvFlatCfg(SpotGraspStableEnvCfg):
    """Configuration for Spot on flat terrain."""

    def __post_init__(self):
        super().__post_init__()

        # Change terrain to plane
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # Disable terrain curriculum
        self.curriculum.terrain_levels = None


##
# Play/Evaluation Configuration
##
@configclass
class SpotGraspStableEnvCfg_PLAY(SpotGraspStableEnvFlatCfg):
    """Configuration for evaluation/play mode."""

    def __post_init__(self):
        super().__post_init__()

        # Smaller scene for evaluation
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
