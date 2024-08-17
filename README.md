# README

## Introduction

Chronic back pain (CBP) is a significant global health issue, affecting many people and incurring substantial healthcare costs. In the United States, CBP treatment expenses reached \$365 billion in 2010. Despite extensive research, the root cause of CBP remains unclear, and the transition from subacute to chronic back pain is common. CBP often coexists with other health issues, making early identification and prevention crucial.

Existing prognostic models for CBP rely mainly on subjective pain reports and have limited accuracy. Recent neuroimaging studies have identified brain characteristics that could serve as reliable CBP prognosis predictors. However, large-scale, multisite validation of these predictors is lacking. This study aims to develop models trained on structural connectome data from three large neuroimaging datasets to identify individuals with CBP using various machine learning models.

The report is organized into three main sections: 1. Data acquisition from three laboratories. 2. Methods, including the research pipeline. 3. Findings and conclusions.

## Data

Data for this study were collected from three laboratories: New Haven, Chicago, and Mannheim, using diffusion tensor imaging (DTI). DTI measures water molecule diffusion in white matter tracts, providing information on the brain's structural connectivity. We used the population-based structural connectome (PSC) method, as published by Prof. Zhengwu Zhang in 2018 (Available: [PMC5910206](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5910206/)), to analyze the DTI data. PSC generates connectivity matrices based on fiber count, fiber length, and fiber volume shared between regions of interest (ROIs), using the Desikan or Destrieux atlas.

We collected data from 28 subjects in New Haven, 56 in Chicago, and 40 in Mannheim. After applying exclusion criteria based on head translation and rotation, we used data from 21 subjects in New Haven, 50 in Chicago, and 39 in Mannheim, resulting in 59 subjects in the persistent group and 51 in the recovered group.

In addition to DTI data, we also have Labels and Demographics data, which are provided in 'GroupLabelsPerSite.xlsx'.

## Methodology

The research framework involves several steps: dimension reduction, harmonization, correction for confounders, and building/testing machine learning models. The following methods were used:

1.  **TN-PCA (Tensor Network Principal Components Analysis):**
    -   Reduces high-dimensional tensor network data from a structural connectome to a lower-dimensional space.
    -   Used 18 types of connection matrices per atlas, selecting fiber count and total connected surface area features.
    -   Applied TN-PCA with hyper-parameter values to generate principal component scores.
    -   For the TN-PCA step, we've utilized Matlab code developed by Prof. Zhengwu Zhang in 2019 (Available: [PMC5910206](https://doi.org/10.1016/j.neuroimage.2019.04.027)). Note that the TN-PCA code is a separate step provided in Matlab, so you need to run TN-PCA for all the dataset options before running this R code. The example TN-PCA result is provided in the "derived_TNPCA" folder.

Below steps are included in the `Brain_CBP_ML.R` script. When running the `Brain_CBP_ML.R` script to execute the above steps, you can choose the main dataset between Chicago and New Haven. Additionally, if you set `mixYes = TRUE`, the code will use a mixed dataset consisting of one main dataset (either Chicago or New Haven) combined with 40% of data from the other site.

2.  **Combat Harmonization:**
    -   Reduces unwanted variability in data collected from different laboratories.
    -   Adjusts data to minimize variation unrelated to the variables of interest, enhancing effect detection.
    -   Function in `Brain_CBP_ML.R`: `harmonFun()`
3.  **Linear Regression:**
    -   Conducted to account for confounding effects of age, sex, head translation, and rotation.
    -   Used residuals in downstream analysis to ensure observed effects are not due to these factors.
4.  **Machine Learning:**
    -   Applied logistic regression, linear support vector classification (SVC), and random forest models with leave-one-out cross-validation (LOOCV).
    -   Used data combinations from different laboratories for training and testing.
    -   Employed LOOCV due to the limited number of subjects to ensure robust results.
    -   Function in `Brain_CBP_ML.R`: `multi2()` which includes `logisticLOOCV()`, `svcLOOCV()`, and `rfLOOCV()`

The final output is a CSV file with AUC (Area Under Curve) values for the train set, validation set, test set1 (full or 60% rest of the dataset of one of the main data not used in the train set), and test set2 (Mannheim dataset). This results in four columns for one simulation, and the number of columns will be multiplied by four as you increase the number of trials. An example output is provided in the result folder with the file name "ChicagoMix_desikan_count.csv". This file name indicates that the main data is from Chicago, mixed with 40% of New Haven data, and the TN-PCA input was a matrix of Desikan count. For this file, we performed 20 simulations, resulting in 4\*20 = 80 AUC values for the modeling.

## Results

The best performing structural connectivity model combined 40% of the New Haven data with 100% of the Chicago data for initial training and validation, then tested on the remaining 60% of the New Haven data or on the Mannheim data. The average AUC was 0.67 ± 0.03 using a support vector classifier (SVC) for the New Haven sample and 0.53 ± 0.03 for the Mannheim data.

## Files and Structure

-   **Brain_CBP_ML.R:** Main R script for running machine learning validation.
-   **Data:** `labels` contains label data; `derived_TNPCA` includes matrices after TNPCA.
-   **Result:** `result` folder contains results with the mean AUC for each dataset (train, validation, test).
-   **README.md:** This file.

## Usage

1.  **Set Directory:** Ensure the correct directory is set before running the code.

2.  **Prepare Data:**

    -   Estimate structural connectivity using population-based structural connectomes (PSC).
    -   Run TNPCA for dimensionality reduction.
    -   Verify data files are correctly named and placed as required by the script.

3.  **Run Script:** Execute the `Brain_CBP_ML.R` script, which includes steps for reading TNPCA results, applying combat harmonization, removing confounding factors, and training/testing models.

## Contact

For questions or additional information, please contact Kyungjin Sohn.
