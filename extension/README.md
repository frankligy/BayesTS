## How to incorporate new modalities?

There are three ways users can extend BayesTS model:

* Construct New AnnData [see here](https://github.com/frankligy/BayesTS/tree/main/extension#construct-new-anndata)
* Incorporate New Modality through `custom.py` [see here](https://github.com/frankligy/BayesTS/tree/main/extension#incorporate-new-modalities-through-custompy)
* Using posterior as the new prior for any other model [see here](https://github.com/frankligy/BayesTS/tree/main/extension#using-posterior-as-new-prior)

### Construct New AnnData

Construct a new `AnnData` with other modalities such as splicing junction or any features. You `AnnData` should be uniformly like following:

```
                 tisse_type      tissue_type    ...      tissue_type       
               observation_1    observation_2   ...     observation_n
feature_1         value_           value_       ...        value_    
feature_2         value_           value_       ...        value_
...               ...              ...          ...        ...
feature_n         value_           value_       ...        value_
```

When you print your `AnnData`, your `AnnData` should be following:

```
# here n_obs is the features, n_vars is the observations
# tissue is the var of the anndata
AnnData object with n_obs × n_vars = 1430522 × 155
    var: 'tissue'
```

With that, you can just run BayesTS like you do in the base model, our manuscript shows the performance for splicing junction and logic-gate target pair:

```bash
python BayesTS.py --input new_adata.h5ad 
                  --weight weights.txt 
                  --mode XY
```

### Incorporate New Modalities through custom.py

Users just need to configure `custom.py` file to define the new modality (either point estimate or distributions) they would like to introduce. In the demo, we chose to incorporate LFC value indicating how overexpressed a gene is between melanoma and skin tissues to further guide the selection of melanoma specific antigens. We will talk about how to construct the `custom.py` in the following section, but assuming you have finished all these steps, to run BayesTS it is as easy as follow:

```bash
# below "c" means custom
# and make sure the custom.py is placed to the same level as BayesTS script
./BayesTS.py --input gtex_gene_subsample.h5ad --weight weights.txt --mode XYc --outdir output_xyc
```

I hope the provided template can be self-explainable and users who are familiar with python and bayesian modeling should be able to adapt. But just to explain, you should define a function called `generate_and_configure`, which takes a list of `ENSG/feature ID`, and return four variables:

1. `CUSTOM`: a tensor of the shape `n_observations * n_common_features`
2. `common`: a list of comon gene/ENSG ID/features
3. `order`: the indices of these `common` genes in original `ENSG/feature ID` list
4. `device`: either `cpu` or `cuda`

Other than that, you can randomly write this function, as long as you keep the above function signatures.

Then you need to define `model_custom` and `guide_custom` function, it is important that pyro requires the `model` and `guide` functions, and it defines how you would connect the underlying `BayesTS` score to the observation in your new modality. Defining this single model is to rescale the final combined run.

Finally you need to define the combine model `model_X_Y_custom` and `guide_X_Y_custom`, you should use the template in the current script and only replace the custom parts. With that, you should be all set.

### Using Posterior as new Prior

In the `full_results.txt`, user will find the posterior alpha and beta, which can serve as informative prior for next round of Bayesian inference, this opens other possibilities for even more flexible tuning of current model. If you are interseted in exploring that angle, please don't hesitate to reach out to me!

## Contact

Guangyuan(Frank) Li

Email: li2g2@mail.uc.edu

PhD student, Biomedical Informatics

Cincinnati Children’s Hospital Medical Center(CCHMC)

University of Cincinnati, College of Medicine


