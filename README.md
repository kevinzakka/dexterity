# Shadow Hand Mujoco

## Changelog

* Changed the worldbody joint ranges to the ones in the spec sheet, page 7.
    * Specifically, WRJ1, THJ1 differred from the spec sheet, presumably from an older model of the hand?
    * I also increased significant digits on all joint ranges to 6.

## Notes

* Joint ranges
* Actuator ranges

Spec sheet page 7 specifies joint ranges
However, mujoco model joint ranges differs from spec sheet

Mujoco Model in `<worldbody>`:

```
WRJ1 -> -0.523599 0.174533 | -30 10
WRJ0 -> -0.698132 0.488692 | -40 28

FFJ3 -> -0.349066 0.349066 | -20 20
FFJ2 -> 0 1.5708           | 0 90
FFJ1 -> 0 1.5708           | 0 90
FFJ0 -> 0 1.5708           | 0 90

MFJ3 -> -0.349066 0.349066 | -20 20
MFJ2 -> 0 1.5708           | 0 90
MFJ1 -> 0 1.5708           | 0 90
MFJ0 -> 0 1.5708           | 0 90

RFJ3 -> -0.349066 0.349066 | -20 20
RFJ2 -> 0 1.5708           | 0 90
RFJ1 -> 0 1.5708           | 0 90
RFJ0 -> 0 1.5708           | 0 90

LFJ4 -> 0 0.785398         | 0 45
LFJ3 -> -0.349066 0.349066 | -20 20
LFJ2 -> 0 1.5708           | 0 90
LFJ1 -> 0 1.5708           | 0 90
LFJ0 -> 0 1.5708           | 0 90

THJ4 -> -1.0472 1.0472     | -60 60
THJ3 -> 0 1.22173          | 0 70
THJ2 -> -0.20944 0.20944   | -12 12
THJ1 -> -0.698132 0.698132 | -40 40
THJ0 -> 0 1.5708           | 0 90
```

Spec sheet, page 7:

```
WRJ1 -> -0.523599 0.174533 | -28 8
WRJ0 -> -0.698132 0.488692 | -40 28

FFJ3 -> -0.349066 0.349066 | -20 20
FFJ2 -> 0 1.5708           | 0 90
FFJ1 -> 0 1.5708           | 0 90
FFJ0 -> 0 1.5708           | 0 90

MFJ3 -> -0.349066 0.349066 | -20 20
MFJ2 -> 0 1.5708           | 0 90
MFJ1 -> 0 1.5708           | 0 90
MFJ0 -> 0 1.5708           | 0 90

RFJ3 -> -0.349066 0.349066 | -20 20
RFJ2 -> 0 1.5708           | 0 90
RFJ1 -> 0 1.5708           | 0 90
RFJ0 -> 0 1.5708           | 0 90

LFJ4 -> 0 0.785398         | 0 45
LFJ3 -> -0.349066 0.349066 | -20 20
LFJ2 -> 0 1.5708           | 0 90
LFJ1 -> 0 1.5708           | 0 90
LFJ0 -> 0 1.5708           | 0 90

THJ4 -> -1.0472 1.0472     | -60 60
THJ3 -> 0 1.22173          | 0 70
THJ2 -> -0.20944 0.20944   | -12 12
THJ1 -> -0.698132 0.698132 | -30 30
THJ0 -> 0 1.5708           | 0 90
```

So it basically differs in WRJ1, THJ1.
