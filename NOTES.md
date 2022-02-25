# Notes

Just a bunch of notes as I play around with MuJoCo and learn more about the Shadow Hand.

## Calibration

The simulation parameters OpenAI calibrated in their Rubik's cube paper:

* dof_damping
* jnt_range
* geom_size
* tendon_range
* tendon_lengthspring
* tendon_stiffness
* actuator_forcerange
* actuator_gainprm

## Code Architecture

Trying to figure out how to create the base task abstraction and then how to make specific tasks that inherit from it. For example, we'd want a `InHandManipulationTask` that inherits from the `BaseTask`.

Also have to figure out how we're gonna set things up to easily swap out action spaces. For example, one policy might want to directly control the joint positions, maybe another might want to control relative joint positions and then another might want to command fingertip positions, in which case there needs to be an IK solver under the hood.

```python
from dm_control import composer

env = composer.Environment(
    task,
    time_limit,
)

timestep = env.reset()

# Policy returns relative joint positions.
# Under the hood, these relative joint positions need to get converted to relative joint
# positions using the IK solver.
action = policy(timestep.observation)
timestep = env.step(action)
```

`composer.Task` has very granular control over the physics simulation.

* before_step: A callback which is executed before an agent control step.
* before_substep: A callback which is executed before a simulation step.
* after_substep: A callback which is executed after a simulation step.
* after_step: A callback which is executed after an agent control step.

By default, the control signal for the actuators is set in `before_step`.

```python
def before_step(self, physics, action, random_state):
    del random_state  # Unused.
    physics.set_control(action)
```
