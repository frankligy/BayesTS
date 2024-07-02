## How to incorporate new modalities?

Users just need to configure `custom.py` file to define the new modality they would like to introduce. In the demo, we chose to incorporate LFC values indicating how overexpressed a gene is between melanoma and skin tissues to further guide the selection of melanoma specific antigens. We will talk about how to construct the `custom.py`, but assuming you finish this step, to run BayesTS it is as easy as follow:

```bash
# below "c" means custom
# and make sure the custom.py is placed to the same level as BayesTS script
./BayesTS_rev.py --input gtex_gene_subsample.h5ad --weight weights.txt --mode XYc --outdir output_xyc
```

## How to construct the custom.py script?

I hope the provided template can be self-explainable and users who are familiar with python and bayesian modeling should be able to adapt. But just to explai, you should define a function called `generate_and_configure`, which takes a list of `ENSG ID`, and return four variables:

1. `CUSTOM`: a tensor of the shape `n_feature * n_common_gene`
2. `common`: a list of comon gene/ENSG ID
3. `order`: the indices of these `common` genes in original `ENSG_ID` list
4. `device`: either `cpu` or `cuda`

Other than that, you can randomly write this function, as long as you keep the above signatures.

Then you need to define `model_custom` and `guide_custom` function, it is important that pyro requires the `model` and `guide` functions, and it defines how you would connect the underlying `BayesTS` score to the observation in your new modalities. Defining this single model is to rescale the later combined run.

Finally you need to define the combine model `model_X_Y_custom` and `guide_X_Y_custom`, you should use the template in the current file and only replace the custom parts. With that, you should be all good.

## Even more flexible

So the resultant BayesTS can serve as the prior for other bayesian model, and you can also redefine the X and Y just like we demonstrated in logic-gate and splicing junction example. More clarification, please shoot me an email below.

## Contact

Guangyuan(Frank) Li

Email: li2g2@mail.uc.edu

PhD student, Biomedical Informatics

Cincinnati Children’s Hospital Medical Center(CCHMC)

University of Cincinnati, College of Medicine


