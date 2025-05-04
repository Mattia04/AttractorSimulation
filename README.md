# Attractors Simulation

This repository contains a Python project for simulating and visualizing various 3D and 2D strange attractors using numerical integration. It includes animation and static plotting utilities, as well as a collection of pre-implemented attractor models.

### List of 3D attractors implemented

- Lorentz
- Lorentz-83
- Thomas
- Langford
- Dadras
- Chen-Lee
- Rössler
- Halvorsen
- Rabinovich-Fabrikant (Not working for `make_animation()`)
- Three scroll
- Sprott
- Sprott-Linz
- Four wing

### List of 2D attractors implemented

None implemented yet.

## Usage

In `Simulation2.py` uncomment the blocks you want to generate and then run the simulation to save the animations or static images:

```bash
python3 Simulation2.py
```

Output files will be saved in the `videos/` and `images/` directories.

## Installation

Download or clone this repository on your machine.

### Using UV (recommended)
In the project directory run:
```commandline
uv run Simulation2.py
```

### Using pip
In the project directory run:
```commandline
python3 -m venv .venv
```
On macs/linux:
```commandline
source .venv/bin/activate
```
On windows
```commandline
.venv\Scripts\activate
```
Then install the packages
```commandline
pip install requirements.txt
```


## References

### Websites

- [Dinamic Mathematics](https://www.dynamicmath.xyz/strange-attractors/)

<!--
### Books

- Steven H. Strogatz, Nonlinear Dynamics and Chaos (Westview Press, second edition)
-->

## Other attractors to implement in the future:

#### 2D Attractors (Discrete Maps)
- Lozi Map
- Hénon Map
- Ikeda Map
- Tinkerbell Map
- Gingerbreadman Map
- Sine Map
- Tent Map
- Logistic Map (not chaotic for all parameters)
- Arnold’s Cat Map
- Kaplan–Yorke Map
- Clifford Attractor
- Peter de Jong Attractor
- Zaslavsky Map

#### 3D Attractors (ODEs and Flow Systems)
- [Chua's circuit](https://en.wikipedia.org/wiki/Chua%27s_circuit)
* Nose–Hoover Attractor
* Dequan–Li Attractor
* Newton–Leipnik Attractor

#### 3. Attractors from Physical Systems
- Duffing Oscillator
- Van der Pol Oscillator (in forced chaotic regime)
- FitzHugh–Nagumo (limit cycles, can have chaos)
- Predator–Prey Models (chaotic under certain conditions)