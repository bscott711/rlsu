# Richardson–Lucy Spectral Unmixing

These files provide the Richardson–Lucy Spectral Unmixing (RLSU) algorithm described in the manuscript [_Multispectral imaging with uncompromised spatiotemporal resolution_](https://doi.org/10.1038/s41566-025-01745-7) (``rlsu.py``), along with demo files for their use, as well as a demonstration of joint unmixing-deconvolution (as described in Supplementary Note 2).

After installing the necessary Python modules (`numpy`, `cupy`, `pandas`, `scipy` and `tifffile`), spectral unmixing can be performed with:
```
python rlsu.py --input demo_raw_data.tif --output demo_unmixed_rlsu.tif --matrix demo_PRISM_mixing_matrix_EGFP_Qdot_525_Qdot_565_Qdot_585_Qdot_605_Qdot_655_Qdot_705_TagBFP.csv --inverse demo_unmixed_inverse.tif
```
This takes the mixed, noisy data in `demo_raw_data.tif` and unmixes it using the mixing matrix specified in `demo_PRISM_mixing_matrix_EGFP_Qdot_525_Qdot_565_Qdot_585_Qdot_605_Qdot_655_Qdot_705_TagBFP.csv`.
This mixing matrix was generated via the web application associated with the manuscript (settings as in [this](https://beryl.mrc-lmb.cam.ac.uk/calculators/spectral_unmixing/?_inputs_&f1n=%22EGFP%22&f4n=%22Qdot%20585%22&primary=%22Spectrally%20flat%20(e.g.%2080%3A20%20beamsplitter)%22&brightness=100&f3n=%22Qdot%20565%22&f2n=%22Qdot%20525%22&f6n=%22Qdot%20655%22&f7n=%22Qdot%20705%22&brightnessScale=false&f5n=%22Qdot%20605%22&dset=%22PRISM%20491%2F514%2F532%2F561%2F594%2F633%2F670%22&f8n=%22TagBFP%22)).
The RLSU result is saved to `demo_unmixed_rlsu.tif`, while the conventional linear inverse unmixing result is saved to `demo_unmixed_inverse.tif`.
Ground truth objects are included in `demo_ground_truth.tif`.

A demonstration of joint unmixing-deconvolution can be run via
```
python unmixing_deconvolution_demo.py
```
with results saved to `demo_unmixed_deconvolution.tif`.
The progress of the algorithm over iterations can be seen in `demo_unmixed_deconvolution_iterations.tif`.
