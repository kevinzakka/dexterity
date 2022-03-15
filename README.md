# Shadow Hand

A suite of tools for simulating the [Shadow Hand](https://www.shadowrobot.com/) in [MuJoCo](https://mujoco.org/) and constructing [dm_control](https://github.com/deepmind/dm_control) environments for training policies with reinforcement learning.

## Todos

**Core**

- [x] Upgrade to native MuJoCo python bindings.
- [x] Add torque sensors and fingertip sites.
- [x] Implement inverse kinematics.
    - [x] Implement basic pseudoinverse `CartesianVelocitytoJointVelocityMapper`.
    - [ ] Implement LSQP `CartesianVelocitytoJointVelocityMapper`.
    - [ ] Add support for disabling wrist pitch joint (as in OpenAI's [Learning Dexterity](https://arxiv.org/abs/1808.00177)).
    - [ ] Consider speeding up  with C++ implementation if too slow.
- [x] Optimize scene light settings in XML file.
- [x] `Reach` task
    - [x] Implement sparse and dense reward.
    - [x] Unit test task.
- [x] `ReOrient` task
    - [x] Add shaped reward.
    - [x] Add episode termination criteria.
    - [x] Figure out time limit vs max time steps.
    - [ ] Unit test task.
- [x] Effectors
    - [x] Hand effector with underlying mujoco actuation.
    - [ ] Figure out clean API for relative vs absolute control.
    - [ ] Merge IK effector into main.

**Misc.**

- [ ] Figure out joint torque and velocity limits. I submitted an [issue](https://github.com/shadow-robot/sr_core/issues/206).

## Notes

See [NOTES.md](NOTES.md) for more information.

## Logbook

See [LOG.md](LOG.md) for more information.

## Questions

> Is it better to specify damping at the `<joint>` level or have an explicit velocity actuator with a damping gain?

> How do I tune the `<visual>`, `<option>` and `<size>` properties?

> How do I tune `<joint>` properties such as `<damping>`, `<armature>`, `<margin>` and `<frictionloss>`?

> How do I generate an XML file from a dynamically create MJCF model? `to_xml_string()` does not resolve asset paths.

Looks like I can use [export_with_assets.py](https://github.com/deepmind/dm_control/blob/master/dm_control/mjcf/export_with_assets.py) from `dm_control.mjcf`.
