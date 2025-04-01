# **HEM**
Project devoted to high entropy materials (HEM)

## HECN
A DeepMD model for high-entropy carbides and carbonitrides (HECN) [[1](https://www.nature.com/articles/s41598-024-78377-4),[2](https://www.sciencedirect.com/science/article/pii/S0925838824037666)].

We consider here HECN structures in the crystalline and liquid states, and for each structure we perform AIMD simulations at a constant temperature of 3500 K during the 200 steps with a timestep of 1.5 fs. The general procedure of training is presented in the following figure, and details can be found in the Ref.[1](https://www.nature.com/articles/s41598-024-78377-4)

![image](https://github.com/user-attachments/assets/3da01ba7-6576-42ac-9679-e4ecd2b567ba)

### Training

The DP-PBE model is constructed using a smooth edition of the Deep Potential model [3]. The DeePMD-kit package is used for training. In each iteration, two models are trained simultaneously using the same data set, with the only difference being the random seeds employed to initialize the model parameters. The sizes of the embedding and fitting nets are set to (30, 60, 120) and (120, 120, 120), respectively. 
The cut-off radius is set to 6 Å. The Adam stochastic gradient descent method with the default hyperparameter settings provided by the TensorFlow package is used to train the DP models.

The starting and final learning rate was 1×10^−4 and 5×10^−8, respectively.

In each DP-GEN [4,5] iteration the DP model is trained with 5.0×10^4 steps.

### References

[1] V. S. Baidyshev, C. Tantardini, and A. G. Kvashnin, Melting simulations of high-entropy carbonitrides by deep learning potentials, Sci Rep 14, 28678 (2024).

[2] D. S. Nikitin, I. I. Shanenkov, A. R. Nassyrbayev, A. A. Sivkov, V. S. Baidyshev, Y. A. Kvashnina, N. A. Matsokin, A. Ya. Pak, and A. G. Kvashnin, Synthesis of high-entropy Ti-Zr-Nb-Hf-Ta carbides and carbonitrides in high-speed arc discharge plasma jet, Journal of Alloys and Compounds 1010, 177178 (2025).

[3] L. Zhang, J. Han, H. Wang, W. Saidi, R. Car, and W. E, End-to-End Symmetry Preserving Inter-Atomic Potential Energy Model for Finite and Extended Systems, in Advances in Neural Information Processing Systems, Vol. 31 (Curran Associates, Inc., 2018).

[4] L. Zhang, D.-Y. Lin, H. Wang, R. Car, and W. E, Active learning of uniformly accurate interatomic potentials for materials simulation, Phys. Rev. Mater. 3, 023804 (2019).

[5] Y. Zhang, H. Wang, W. Chen, J. Zeng, L. Zhang, H. Wang, and W. E, DP-GEN: A concurrent learning platform for the generation of reliable deep learning based potential energy models, Computer Physics Communications 253, 107206 (2020).
