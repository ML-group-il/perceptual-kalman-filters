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


## What's in the paper?

 We study the _Gauss-Markov_ setting with linear observations:

$x_{k}=A_{k}x_{k-1}+q_{k}, k=1,...,T$, 

$y_{k}=C_{k}x_{k}+r_{k}, k=0,...,T$.

$x_0,q_k,r_k$ are zero-mean Gaussian r.v.'s with covariance matrices $P_0,Q_k,R_k$.


The problem is to minimize the **Quadratic** cost under the **Temporal causality** and **Perfect perceptual quality** constraints:.
```math
\mathcal C (\hat{x}_{0},\ldots,\hat{x}_{T}) = 
\sum_{k=0}^{T}\alpha_{k}E{\|{x}_{k}-\hat{x}_{k}\|^{2}}
```
```math
\hat{x}_{k}\sim p_{\hat{x}_{k}}(\cdot|y_{0},\ldots,y_{k},\hat{x}_{0},\ldots,\hat{x}_{k-1})
```
```math
p_{\hat{X}_{0}^{T}}=p_{X_{0}^{T}}
```

Without the perceptual constraint, the **Kalman state** is a MSE-optimal causal estimator.
 ```math
  \hat{x}^*_k = A_k \hat{x}^*_{k-1} + K_k \mathcal{I}_k
```
$K_k$ is the Kalman gain 
and the **innovation process** describes the new information carried by the observation $y_k$.

```math
\mathcal{I}_k=y_{k}-C_{k}\hat{x}^*_{k|k-1}
```

We suggest a formalism for linear perceptual filters
```math
\hat{x}_{k}=A_{k}\hat{x}_{k-1}+J_k
```
```math
J_{k}=\varPhi_{k}A_{k}\Upsilon_{k}+\Pi_{k}K_{k}\mathcal{I}_{k}+w_{k},\quad w_{k}\sim\mathcal{N}\left(0,\Sigma_{w_{k}}\right)
```
```math
\Sigma_{w_{k}}=Q_{k}-\varPhi_{k}A_{k}\Sigma_{\Upsilon_{k}}A_{k}^{\top}\varPhi_{k}^{\top}-\Pi_{k}M_{k}\Pi_{k}^{\top}\succeq0
```

![sketch](https://github.com/ML-group-il/perceptual-kalman-filters/assets/147659286/b4ee51de-a809-4afd-b96c-6fcce4eed4d8)
