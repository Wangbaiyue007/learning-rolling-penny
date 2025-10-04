# learning-rolling-penny
Code for paper [Learning the Rolling Penny Dynamics](https://arxiv.org/abs/2410.15201).

## Abstract
We  consider learning the dynamics of a typical nonholonomic system -- the rolling penny. A nonholonomic system is a system subject to 
nonholonomic constraints. Unlike a holonomic constraints, a nonholonomic constraint does not define a submanifold on the configuration 
space. Therefore, the inverse problem of finding the constraints has to involve the tangent space. This paper discusses how to learn 
the dynamics, as well as the constraints for such a system, given the data set of discrete trajectories on the tangent bundle $TQ$

## An Illustration
<img src=figs/rolling-penny.png width="800" />

Here $G$ is a Lie group, $\xi^q$ is a Lie algebra element, and $\xi^q_Q$ is the corresponding vector field.
$\mathcal{D_q}$ is the constraint distribution at $q$. We learned the vector field that lies in the constraint distribution,
which is the so called horizontal vector field.

## To cite the paper, use
```
@misc{wang2024learningrollingpennydynamics,
      title={Learning the Rolling Penny Dynamics}, 
      author={Baiyue Wang and Anthony Bloch},
      year={2024},
      eprint={2410.15201},
      archivePrefix={arXiv},
      primaryClass={math.DS},
      url={https://arxiv.org/abs/2410.15201}, 
}
```
