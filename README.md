cmtip is a Python implementation of cartesian MTIP (multitiered iterative phasing), and in some sense a next-generation spinifel. The three main algorithmic components are:
1. alignment, in which image orientations are deduced by comparison to references slices through the diffraction volume, which are computed from the autocorrelation volume using the forward nonuniform FFT (NUFFT).
2. solving for the autocorrelation, by maximizing the consistency of the oriented images with the intensity model. This component makes use of the adjoint NUFFT.
3. phasing, in which rounds of hybrid-input output (HIO), error reduction (ER), and shrink-wrap are used to compute the density from the oversampled autocorrelation. 

This repository was structured with modular testing in mind. Alignment and phasing accuracy can both be checked by supplying the ideal autocorrelation, while the autocorrelation solver can be checked by assuming ground truth orientations. At each iteration of the full pipeline, the estimated autocorrelation and density maps are saved as MRC files and the predicted orientations are saved to a numpy array. Parallelization, expansion to non-square detectors, and quantitative metrics tracking accuracy and convergence are currently in progress.

Current dependencies are skopi, finufft, cufinufft, and mrcfile, all of which are available through pip install.
