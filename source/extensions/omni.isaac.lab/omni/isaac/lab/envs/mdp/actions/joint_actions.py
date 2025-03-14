# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

import omni.isaac.lab.utils.string as string_utils
from omni.isaac.lab.assets import Articulation, DeformableObject, RigidObject
from omni.isaac.lab.managers.action_manager import ActionTerm
from omni.isaac.lab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv

    from . import actions_cfg


class JointAction(ActionTerm):
    r"""Base class for joint actions.

    This action term performs pre-processing of the raw actions using affine transformations (scale and offset).
    These transformations can be configured to be applied to a subset of the articulation's joints.

    Mathematically, the action term is defined as:

    .. math::

       \text{action} = \text{offset} + \text{scaling} \times \text{input action}

    where :math:`\text{action}` is the action that is sent to the articulation's actuated joints, :math:`\text{offset}`
    is the offset applied to the input action, :math:`\text{scaling}` is the scaling applied to the input
    action, and :math:`\text{input action}` is the input action from the user.

    Based on above, this kind of action transformation ensures that the input and output actions are in the same
    units and dimensions. The child classes of this action term can then map the output action to a specific
    desired command of the articulation's joints (e.g. position, velocity, etc.).
    """

    cfg: actions_cfg.JointActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor | float
    """The scaling factor applied to the input action."""
    _offset: torch.Tensor | float
    """The offset applied to the input action."""

    def __init__(self, cfg: actions_cfg.JointActionCfg, env: ManagerBasedEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names, preserve_order=self.cfg.preserve_order)
        self._num_joints = len(self._joint_ids)
        # log the resolved joint names for debugging
        omni.log.info(f"Resolved joint names for the action term {self.__class__.__name__}:" f" {self._joint_names} [{self._joint_ids}]")

        # Avoid indexing across all joints for efficiency
        if self._num_joints == self._asset.num_joints:
            self._joint_ids = slice(None)

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

        # parse scale
        if isinstance(cfg.scale, (float, int)):
            self._scale = float(cfg.scale)
        elif isinstance(cfg.scale, dict):
            self._scale = torch.ones(self.num_envs, self.action_dim, device=self.device)
            # resolve the dictionary config
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.scale, self._joint_names)
            self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported scale type: {type(cfg.scale)}. Supported types are float and dict.")
        # parse offset
        if isinstance(cfg.offset, (float, int)):
            self._offset = float(cfg.offset)
        elif isinstance(cfg.offset, dict):
            self._offset = torch.zeros_like(self._raw_actions)
            # resolve the dictionary config
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.offset, self._joint_names)
            self._offset[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported offset type: {type(cfg.offset)}. Supported types are float and dict.")

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # apply the affine transformations
        self._processed_actions = self._raw_actions * self._scale + self._offset

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0


class JointPositionAction(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    cfg: actions_cfg.JointPositionActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.JointPositionActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use default joint positions as offset
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

    def apply_actions(self):
        # set position targets
        self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)


class JointPositionBaseForceTorqueAction(JointAction):
    """joint tar dof pos and base external force and torque"""

    def __init__(self, cfg: actions_cfg.JointPositionBaseForceTorqueCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use default joint positions as offset
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

        self._force_limit = cfg.force_limit
        self._torque_limit = cfg.torque_limit
        self._force_scale = cfg.force_scale
        self._torque_scale = cfg.torque_scale

        self._body: RigidObject | Articulation = env.scene[cfg.body_cfg.name]
        self._body_ids = cfg.body_cfg.body_ids

        if not isinstance(self._body, (RigidObject, Articulation)):
            raise ValueError(f"Unsupported body type: {type(self._body)}. Supported types are RigidObject and Articulation.")

    @property
    def action_dim(self) -> int:
        return self._num_joints + 6

    def process_actions(self, actions):
        # store the raw actions
        self._raw_actions[:] = actions
        # apply the affine transformations
        forces = self._raw_actions[:, self._num_joints : self._num_joints + 3] * self._force_scale
        forces = torch.clamp(forces, -self._force_limit, self._force_limit)
        torques = self._raw_actions[:, self._num_joints + 3 :] * self._torque_scale
        torques = torch.clamp(torques, -self._torque_limit, self._torque_limit)

        self._processed_actions = torch.cat(
            [
                self._raw_actions[:, : self._num_joints] * self._scale + self._offset,
                forces,
                torques,
            ],
        )

    def reduce_limits(self, scale):
        self._force_limit *= scale
        self._torque_limit *= scale
        if self._force_limit < 1:
            self._force_limit = 0.0
        if self._torque_limit < 1:
            self._torque_limit = 0.0

    def zero_limits(self):
        self._force_limit = 0.0
        self._torque_limit = 0.0

    def apply_actions(self):
        # set position targets
        self._asset.set_joint_position_target(self.processed_actions[:, : self._num_joints], joint_ids=self._joint_ids)
        # set base external force and torque
        env_ids = torch.arange(self._env.num_envs, device=self.device)
        forces = self.processed_actions[env_ids, self._num_joints : self._num_joints + 3]
        torques = self.processed_actions[env_ids, self._num_joints + 3 :]
        self._body.set_external_force_and_torque(forces, torques, body_ids=self._body_ids, env_ids=env_ids)


class BaseForceTorqueAction(ActionTerm):
    """Base external force and torque action term."""

    cfg: actions_cfg.BaseForceTorqueActionCfg
    """The configuration of the action term."""
    _body: RigidObject | Articulation
    """The body to which the external force and torque are applied."""
    _body_ids: Sequence[int]
    """The body IDs of the body to which the external force and torque are applied."""
    _force_limit: torch.Tensor
    """The limit of the external force."""
    _torque_limit: torch.Tensor
    """The limit of the external torque."""
    _force_scale: float
    """The scaling factor of the external force."""
    _torque_scale: float
    """The scaling factor of the external torque."""
    _raw_actions: torch.Tensor
    """The raw actions."""
    _processed_actions: torch.Tensor
    """The processed actions."""

    def __init__(self, cfg: actions_cfg.BaseForceTorqueActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._force_limit = cfg.force_limit * torch.ones(self.num_envs, device=self.device)
        self._torque_limit = cfg.torque_limit * torch.ones(self.num_envs, device=self.device)
        self._max_force_limit = cfg.force_limit
        self._max_torque_limit = cfg.torque_limit
        self._force_scale = cfg.force_scale
        self._torque_scale = cfg.torque_scale

        cfg.body_cfg._resolve_body_names(env.scene)  # resolve the body names
        self._body = env.scene[cfg.body_cfg.name]
        self._body_ids = cfg.body_cfg.body_ids

        if not isinstance(self._body, (RigidObject, Articulation)):
            raise ValueError(f"Unsupported body type: {type(self._body)}. Supported types are RigidObject and Articulation.")

        self._raw_actions = torch.zeros(self.num_envs, 6, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

    @property
    def action_dim(self) -> int:
        return 6

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def force_limit(self) -> torch.Tensor:
        return self._force_limit

    @property
    def torque_limit(self) -> torch.Tensor:
        return self._torque_limit

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        _force_limit = self._force_limit.unsqueeze(1)
        _torque_limit = self._torque_limit.unsqueeze(1)
        forces = self._raw_actions[:, :3] * self._force_scale
        forces = torch.clamp(forces, -_force_limit, _force_limit)
        torques = self._raw_actions[:, 3:] * self._torque_scale
        torques = torch.clamp(torques, -_torque_limit, _torque_limit)

        self._processed_actions = torch.cat([forces, torques], dim=1)

    def apply_actions(self):
        env_ids = torch.arange(self.num_envs, device=self.device)
        forces = self.processed_actions[env_ids, :3]  # N x 3
        torques = self.processed_actions[env_ids, 3:]  # N x 3
        # N x body_length x 3, body_length = 1, TEMP
        assert len(self._body_ids) == 1, "Only one body is supported for now."
        # N x 3 -> N x 1 x 3
        forces = forces.unsqueeze(1)
        torques = torques.unsqueeze(1)

        self._body.set_external_force_and_torque(forces, torques, body_ids=self._body_ids, env_ids=env_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0

    def update_force_limit(self, env_ids: Sequence[int], scale: float) -> None:
        _delta_force_limit = self._force_limit[env_ids] * (scale - 1)
        # abs(delta) must > 5, if scale > 1, then delta > 5, if scale < 1, then delta < -5
        _delta_force_limit = torch.clamp(_delta_force_limit, 5, float('inf')) if scale > 1 else torch.clamp(_delta_force_limit, float('-inf'), -5)
        self._force_limit[env_ids] += _delta_force_limit

        self._force_limit[self._force_limit < 5] = 0.0

    def update_torque_limit(self, env_ids: Sequence[int], scale: float) -> None:
        _delta_torque_limit = self._torque_limit[env_ids] * (scale - 1)
        # abs(delta) must > 5, if scale > 1, then delta > 5, if scale < 1, then delta < -5
        _delta_torque_limit = torch.clamp(_delta_torque_limit, 5, float('inf')) if scale > 1 else torch.clamp(_delta_torque_limit, float('-inf'), -5)
        self._torque_limit[env_ids] += _delta_torque_limit

        self._torque_limit[self._torque_limit < 5] = 0.0

    def add_force_limit(self, env_ids: Sequence[int], add: float) -> None:
        self._force_limit[env_ids] += add
        self._force_limit[env_ids] = torch.clamp(self._force_limit[env_ids], 0, self._max_force_limit)

    def add_torque_limit(self, env_ids: Sequence[int], add: float) -> None:
        self._torque_limit[env_ids] += add
        self._torque_limit[env_ids] = torch.clamp(self._torque_limit[env_ids], 0, self._max_torque_limit)


class RelativeJointPositionAction(JointAction):
    r"""Joint action term that applies the processed actions to the articulation's joints as relative position commands.

    Unlike :class:`JointPositionAction`, this action term applies the processed actions as relative position commands.
    This means that the processed actions are added to the current joint positions of the articulation's joints
    before being sent as position commands.

    This means that the action applied at every step is:

    .. math::

         \text{applied action} = \text{current joint positions} + \text{processed actions}

    where :math:`\text{current joint positions}` are the current joint positions of the articulation's joints.
    """

    cfg: actions_cfg.RelativeJointPositionActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.RelativeJointPositionActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use zero offset for relative position
        if cfg.use_zero_offset:
            self._offset = 0.0

    def apply_actions(self):
        # add current joint positions to the processed actions
        current_actions = self.processed_actions + self._asset.data.joint_pos[:, self._joint_ids]
        # set position targets
        self._asset.set_joint_position_target(current_actions, joint_ids=self._joint_ids)


class JointVelocityAction(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as velocity commands."""

    cfg: actions_cfg.JointVelocityActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.JointVelocityActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use default joint velocity as offset
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_vel[:, self._joint_ids].clone()

    def apply_actions(self):
        # set joint velocity targets
        self._asset.set_joint_velocity_target(self.processed_actions, joint_ids=self._joint_ids)


class JointEffortAction(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as effort commands."""

    cfg: actions_cfg.JointEffortActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.JointEffortActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

    def apply_actions(self):
        # set joint effort targets
        self._asset.set_joint_effort_target(self.processed_actions, joint_ids=self._joint_ids)
