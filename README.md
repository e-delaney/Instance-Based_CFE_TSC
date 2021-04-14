# Instance-based Counterfactual Explanation for Time Series Classification

This supplementary repository provides details of the experiments in the paper **Instance-based Counterfactual Explanations for Time Series Classification**. Please read the paper for full details of this research.

The central idea behind the Native Guide technique is that existing counterfactual instances can be retrieved for any query and used to guide the counterfactual generation process. By integratrating feature attribution information/deep features into the counterfactual generation process, Native Guide can generate sparse counterfactuals that provide information about discriminative and often meaningful areas of the time series. An explanation weight vector **w**, is often available from the base classifier (e.g. from the activations of DNNs) and in the case where such information is not available, model agnostic techniques such as SHAP could be implemented.

![Image of FCN](https://github.com/e-delaney/Instance-based_CFE_TSC/blob/main/Method_BIG.PNG)

### Datasets
The datasets used are from the UCR archive (http://www.timeseriesclassification.com/) [1]. We can import these datasets directly in the code using tslearn functionality. 

### Black-box Classifier
We have trained a fully convolutional neural network (FCN) originally proposed by Wang et al. [2] for time series classification, closely following the implementation by Fawaz et al. [3] (https://github.com/hfawaz/dl-4-tsc). This can be found in the FCN file. Weights are saved for future use where we want to call this pre-trained model. 

![Image of FCN](https://github.com/e-delaney/Instance-based_CFE_TSC/blob/main/FCN_compressed.PNG)

### Class Activation Mapping 
Once the FCN has been trained we can use class activation mapping [2,3,4] to locate discriminative areas for classification and retrieve an explanation weight vector.
Moreover, the pre-trained model can be conveniently loaded using keras functionality. The resulting CAM is produced and the explanation weight vector is extracted and saved. See the CAM file and corresponding notebooks. 

### Native-Guide
Native guide uses the explanation weight vector (from the CAM) and the in-sample counterfactual (NUN) to generate a new counterfactual solution. If we are using simple 1-NN DTW classifiers we can use dynamic barycentre averaging to generate a counterfactual. The process can be found in the corresponding Native Guide notebooks.

### Benchmark Models and Comparison
We implement the NUN-CF (just the retrieved NUN without adaptation) and the Wachter-CF [5] methods as comparison models. We evaluate our counterfactuals based on proximity, sparsity, plausibility, and diversity. Unlike tabular data, we can also visualize the explanations! 


### References

[1] Dau,  H.A.,  Bagnall,  A.,  Kamgar,  K.,  Yeh,  C.C.M.,  Zhu,  Y.,  Gharghabi,  S.,Ratanamahatana, C.A., Keogh, E.: The ucr time series archive. IEEE/CAA Jour-nal of Automatica Sinica6(6), 1293–1305 (2019)

[2] Wang, Z., Yan, W., Oates, T.: Time series classification from scratch with deepneural networks: A strong baseline. In: IJCNN. pp. 1578–1585. IEEE (2017)

[3] Fawaz, H.I., Forestier, G., Weber, J., Idoumghar, L., Muller, P.A.: Deep learning fortime series classification: a review. Data Mining and Knowledge Discovery33(4),917–963 (2019)

[4]  Zhou, B., Khosla, A., Lapedriza, A., Oliva, A., Torralba, A.: Learning deep fea-tures for discriminative localization. In: IEEE Conference on Computer Vision and Pattern Recognition. pp. 2921–2929 (2016)

[5] Wachter,  S.,  Mittelstadt,  B.,  Russell,  C.:  Counterfactual  explanations  withoutopening the black box: automated decisions and the gdpr. Harv.J.Law Tech.31,841 (2017)
