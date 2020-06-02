Running the file Run_me_NMF.m allows you to run the experiment on a low-rank synthetic data set (namely, the input matrix X = rank(m,r)*rand(r,n) with m=n=200 and r=20). 

It compares with three state-of-the-art NMF methods, namely, 
1) A-HALS from N. Gillis and F. Glineur, "Accelerated Multiplicative Updates and Hierarchical ALS Algorithms for Nonnegative Matrix Factorization", Neural Computation 24 (4), pp. 1085-1105, 2012.
2) E-A-HALS from A.M.S. Ang and N. Gillis, "Accelerating Nonnegative Matrix Factorization Algorithms using Extrapolation", Neural Computation 31 (2), pp. 417-439, 2019.
3) APGC from Xu, Y. and Yin, W., "A block coordinate descent method for regularized multiconvex optimization with applications to nonnegative tensor factorization and completion", SIAM Journal on imaging sciences, 6(3), 1758-1789, 2013. 
4) iPALM from Pock, T. and Sabach, S., "Inertial proximal alternating linearized minimization (iPALM) for nonconvex and nonsmooth problems", SIAM Journal on Imaging Sciences, 9(4):1756–1787, 2016.
