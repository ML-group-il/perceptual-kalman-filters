# perceptual-kalman-filters
Implementation of demo from "PKF: online state estimation under a perfect perceptual quality constraint" (2023) (https://arxiv.org/abs/2306.02400)

to execute run: python ./run_osc_demo.py or ./run_demo.sh

# Motivation for perceptual filtering
**Example**: real time video streaming

![percp_filtering](https://github.com/ML-group-il/perceptual-kalman-filters/assets/147659286/b903bfd9-7c10-4165-810d-1ad24bd3c8d6)


Sensory data from the scene might be compressed/corrupted/missing.

Reconstruction to the viewer must be done in **real-time**.

reconstructed scene must look **natural**.

The latter **realism** demand means that not only each frame should look as a natural image, but motion should look natural too.
That makes us face the **Temporal Consistency dilemma**: Estimation cannot suddenly change the motion in
the output video, because such an abrupt change
would deviate from natural video statistics. Thus,
although the method is aware of its mistake, it may
have to stick to its past decisions.

<img width="774" alt="dilemma" src="https://github.com/ML-group-il/perceptual-kalman-filters/assets/147659286/96e10427-06af-42df-b7e5-28cdadf5d9a6">


# What's in the paper?

 We study the \emp{Gauss-Markov} setting with linear observations:

$x_{k}=A_{k}x_{k-1}+q_{k}, k=1,...,T$, $y_{k}=C_{k}x_{k}+r_{k}, k=0,...,T$.
$x_0,q_k,r_k$ are zero-mean Gaussian r.v.'s with covariance matrices $P_0,Q_k,R_k$.

Our goal is to minimize the **Quadratic** cost 

![sketch](https://github.com/ML-group-il/perceptual-kalman-filters/assets/147659286/b4ee51de-a809-4afd-b96c-6fcce4eed4d8)
