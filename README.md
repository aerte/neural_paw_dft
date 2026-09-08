# Neural Electronic Initialization

This repository contains all the code used to perform the experiments described in
"Complete Neural Electronic Initialization Accelerates Materials DFT".

The code is split into three parts:

- **`vasp_runner`** – a Python-based VASP submission package used to perform the DFT experiments.
- **`spin_electrafi`** – an adaptation of [ELECTRAFI](https://github.com/Jotels/ELECTRAFI) that supports spin-difference density training.
- **`augnet`** – the augmentation occupancy prediction model AugNet described in the paper.
