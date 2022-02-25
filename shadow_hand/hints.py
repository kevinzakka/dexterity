"""Common type hints for shadow_hand."""

from dm_control import mjcf
from dm_control.mujoco.wrapper.core import MjData, MjModel
from typing_extensions import TypeAlias


MjcfElement: TypeAlias = mjcf.element._ElementImpl
MjcfAttachmentFrame: TypeAlias = mjcf.element._AttachmentFrame
MujocoModel: TypeAlias = MjModel
MujocoData: TypeAlias = MjData
MujocoObjectType: TypeAlias = int
