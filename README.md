# perceptual-kalman-filters
Implementation of demo from "PKF: online state estimation under a perfect perceptual quality constraint" (2023) (https://arxiv.org/abs/2306.02400)

to execute run: python ./run_osc_demo.py or ./run_demo.sh

<img width="325" alt="PKF" src="https://github.com/ML-group-il/perceptual-kalman-filters/assets/147659286/ce0f3b78-2dfc-416e-8030-99c6b3146916">

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

We present the following formalism for linear **perceptual filters** with coefficients $\Pi_{k},\varPhi_{k}$
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

$\Upsilon_{k}$ is the **unutilized information** state
```math
\Upsilon_{k}\triangleq\hat{x}_{k-1}^{*}-\mathbb{{E}}\left[\hat{x}_{k-1}^{*}|\hat{x}_{0},\ldots,\hat{x}_{k-1}\right]
```

given by the recursion
```math
\varUpsilon_{k+1} = A_k\varUpsilon_{k} +K_k \mathcal{I}_k -  \varPsi_k Q_k^\dagger J_k,
```
```math
\Sigma_{\varUpsilon_{k+1}} 
= A_k \Sigma_{\varUpsilon_k} A_k^\top +M_k -\varPsi_k      Q_k^\dagger \varPsi_k^\T,
```
```math
\varPsi_k = M_k \Pi_k^\top +A_k \Sigma_{\varUpsilon_k} A_k^\top \varPhi_k^\top. 
```

# The Perceptual Kalman Filter (PKF)

We suggest  the simplified form
```math
J_{k}=\Pi_{k}K_{k}\mathcal{I}_{k}+w_{k},\quad w_{k}\sim\mathcal{N}\left(0,\Sigma_{w_{k}}\right)
```

Now, the optimal coefficients are given explicitly (see full details in our paper)
```math
\Pi_{k}^* = Q_{k}M_{B}^{1/2}\left(M_{B}^{1/2}Q_{k}M_{B}^{1/2}\right)^{1/2 \dagger}M_{B}^{\dagger 1/2}B_{k}M_{k}M_{k}^{\dagger}
```

**Algorithm - Perceptual Kalman Filter (PKF):**

FOR k=1 **to** $T$:

**calculate:** $M_{k}=K_{k}S_{k}K_{k}^{\top}$, $B_{k}= \sum_{t=k}^{T} \alpha_{t}(A^{t-k})^{\top}A^{t-k}$,
$M_{B}=B_{k}M_{k}B_{k}$.

**compute optimal gain:** 
$\Pi_{k} = Q_{k}M_{B}^{1/2}\left(M_{B}^{1/2}Q_{k}M_{B}^{1/2}\right)^{1/2 \dagger}M_{B}^{\dagger 1/2}B_{k}M_{k}M_{k}^{\dagger}
$.

**sample:** $w_{k}\sim \mathcal{N}\left(0,Q_{k}-\Pi_{k}M_{k}\Pi_{k}^{\top}\right).$

**update state:** 
```math
\hat{x}_{k} = A\hat{x}_{k-1} + \Pi_{k}K_{k}\mathcal{I}_{k} + w_{k}
```

# Visual demo


Users
Companies
COLLECTIVES
Explore Collectives
LABS
Discussions
TEAMS
Stack Overflow for Teams – Start collaborating and sharing organizational knowledge. 
How to embed a video into GitHub README.md?
Asked 12 years, 10 months ago
Modified 23 days ago
Viewed 403k times
430

Is it possible to embed a flash video into README.md on GitHub? It isn't showing up: https://github.com/mattdipasquale/PicSciP

flashgithubembedmarkdown
Share
Improve this question
Follow
edited Dec 9, 2011 at 18:03
Joel Coehoorn's user avatar
Joel Coehoorn
401k113113 gold badges570570 silver badges797797 bronze badges
asked Nov 25, 2010 at 17:32
ma11hew28's user avatar
ma11hew28
122k117117 gold badges451451 silver badges651651 bronze badges
Add a comment
16 Answers
Sorted by:

Highest score (default)
360

Add a url to video file in your ReadMe.
Github now supports videos, see more detailed answer here: https://stackoverflow.com/a/4279746/1092815

You can see a live example here (See at the end of the readme):
https://github.com/alelievr/Mixture/blob/0.4.0/README.md

Github Pages
I strongly recommend placing the video in a project website created with GitHub Pages instead of the readme, like described in VonC's answer; it will be a lot better than any of these ideas. But if you need a quick fix just like I needed, here are some suggestions.

Use a gif
See aloisdg's answer, result is awesome, gifs are rendered on github's readme ;)

Use a video player picture
You could trick the user into thinking the video is on the readme page with a picture. It sounds like an ad trick, it's not perfect, but it works and it's funny ;).

Example:

Watch the video


[![Watch the video](g)][](https://drive.google.com/file/d/1rVtd3QMNMq86NnzXfX0AQjMJ4IUbVcjz/view)


