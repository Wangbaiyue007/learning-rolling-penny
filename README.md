# learning-rolling-penny
Code for paper [Learning Nonholonomic Dynamics with Constraint Discovery](https://arxiv.org/abs/2410.15201).

## Abstract
We consider learning nonholonomic dynamical systems while discovering the constraints, and describe in detail the case of the rolling disk. A nonholonomic system is a system subject to nonholonomic constraints. Unlike holonomic constraints, nonholonomic constraints do not define a sub-manifold on the configuration space. Therefore, the inverse problem of finding the constraints has to involve the tangent bundle. This paper discusses a general procedure to learn the dynamics of a nonholonomic system through Hamel's formalism, while discovering the system constraint by parameterizing it, given the data set of discrete trajectories on the tangent bundle $TQ$. We prove that there is a local minimum for convergence of the network. We also preserve symmetry of the system by reducing the Lagrangian to the Lie algebra of the selected group.

## An Illustration
<img src=png/2000.png width="800" />

## To cite the paper, use
```
@misc{wang2025learningnonholonomicdynamicsconstraint,
      title={Learning Nonholonomic Dynamics with Constraint Discovery}, 
      author={Baiyue Wang and Anthony Bloch},
      year={2025},
      eprint={2410.15201},
      archivePrefix={arXiv},
      primaryClass={math.DS},
      url={https://arxiv.org/abs/2410.15201}, 
}
```
