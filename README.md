# Shadow Hand Mujoco

## Changelog

* Changed the worldbody joint ranges to the ones in the spec sheet, page 7.
    * Specifically, WRJ1, THJ1 differred from the spec sheet, presumably from an older model of the hand?
    * I also increased significant digits on all joint ranges to 6.

## Todos

- [ ] Figure out joint torque and velocity limits. I submitted an [issue](https://github.com/shadow-robot/sr_core/issues/206).
