# perceptual-kalman-filters
Implementation of demo from "PKF: online state estimation under a perfect perceptual quality constraint" (2023)

to execute run: python ./run_osc_demo.py or ./run_demo.sh

# Motivation for perceptual filters
**Example**: real time video streaming

![percp_filtering](https://github.com/ML-group-il/perceptual-kalman-filters/assets/147659286/b903bfd9-7c10-4165-810d-1ad24bd3c8d6)


Sensory data from the scene might be compressed/corrupted/missing.

Reconstruction to the viewer must be done in **real-time**.

reconstructed scene must look **natural**.

![dilemma_new](https://github.com/ML-group-il/perceptual-kalman-filters/assets/147659286/325f675b-406f-484a-96bf-749395ae3a75)

**The Temporal consistency dilemma**: Estimation cannot suddenly change the motion in
the output video, because such an abrupt change
would deviate from natural video statistics. Thus,
although the method is aware of its mistake, it may
have to stick to its past decisions.

# What's in the paper?
