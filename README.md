<meta http-equiv="Content-Type" content="text/html; charset=utf-8">

# AIStructDynSolve

### Author:
- **Ke Du(&#x675C;&#x8F72;)**  
  Email: duke@iem.ac.cn

AIStructDynSolve is an artificial intelligence (AI) powered framework designed to solve both forward and inverse problems in structural dynamics. 
It leverages advanced artificial intelligence methods - particularly physics-informed neural networks (PINNs), Physics-Informed Kolmogorov-Arnold Network(PIKANs) and their extensions - to model, predict, and analyze dynamic structural responses under various loading scenarios, such as seismic excitations.

### The framework solves the following ODE of MDOF:

- M\*U_dotdot+C\*U_dot+K*U=Pt

- Initial Conditions:
   U(t=0)=InitialU
   U_dot(t=0)=InitialU_dot

### The framework aims to:
- Accurately simulate time-dependent structural behavior (forward problems).
- Identify structural parameters or input forces from measured responses (inverse problems).
- Incorporate domain knowledge and physical laws for improved generalization and interpretability.
- Address challenges in multi-frequency, multi-scale dynamics, especially in earthquake engineering applications.


### Date:
- Initial: 2023/12/26
- Latest: 2025/08/13

### Update Log

- Version 1.14  -  Aug 13, 2025
   Improved accuracy and stability for inverse problems.

- Version 1.13  -  Aug 5, 2025
   Added support for user-defined trainable_params and improved accuracy and stability in inverse problems.

- Version 1.12  -  Jul 25, 2025
   Fixed minor bugs related to trainable_params handling for inverse problems involving "eq", "eqX", "eqY", or "eqZ".

- Version 1.11  -  Jul 23, 2025
   Added support for trainable_params ("mass" or "stiffness" or "damping"or "pt" or "eq" or "eqX" or "eqY" or "eqZ") for inverse problems.

- Version 1.10  -  May 11, 2025
  Improved accuracy and stability of PIKANs

- Version 1.9  -  Apr 30, 2025
  Added support for EQ2D and  EQ3D of structural dynamic simulations, enabling users to analyze 2D and 3D earthquake loading scenarios.

- Version 1.8  -  Apr 12, 2025
  Improved accuracy of high-frequency response prediction 

- Version 1.7  -  Apr 11, 2025
  Minor bug fixes 

- Version 1.6  - Apr 7, 2025
  Minor bug fixes 

- Version 1.5  -  Jan 13, 2025
  Minor bug fixes 

- Version 1.4  -  Dec 31, 2024
  Minor bug fixes 

- Version 1.3  -  Dec 19, 2024
  Minor bug fixes 

- Version 1.2  -  Dec 18, 2024
  Minor bug fixes 

- Version 1.1  -  Dec 7, 2024
  Minor bug fixes 