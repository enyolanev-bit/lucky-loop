# Related Work Context

Research question: Can nonlinear models robustly outperform logistic regression on public sensor classification data?

## Search Protocol

- API: official arXiv Atom API (`https://export.arxiv.org/api/query`).
- Rate limit: one request every 3.1s.
- Curated fallback: known core papers are included and deduplicated against arXiv metadata.
- Citation stability: arXiv IDs and versions are preserved when available.

## Search Queries

- Can nonlinear models robustly outperform logistic regression on public sensor classification data?
- nonlinear models robustly outperform logistic regression sensor classification data machine learning classification
- nonlinear models robustly outperform logistic regression sensor classification data logistic regression random forest svm gradient boosting
- nonlinear models robustly outperform logistic regression sensor classification data robustness repeated seeds cross validation
- "human activity recognition" sensor classification machine learning
- "EEG eye state" classification machine learning
- wearable sensor classification logistic regression random forest svm

## Source -> Gap -> Metric -> Experiment

| Gap | Sources | Metric | Experiment |
|---|---|---|---|
| The literature context must establish whether nonlinear models are actually expected to improve sensor classification over simple baselines. | [arxiv_2604_25452_2026], [arxiv_2209_11750_2022], [arxiv_2008_00235_2020], [arxiv_1511_06663_2015], [arxiv_2303_00064_2023] | `domain_source_coverage` | Compare baseline and nonlinear model families on a dataset selected for the user question. |
| Claims about sensor classification performance require robustness checks across splits or seeds, not a single score. | [arxiv_2604_25452_2026], [arxiv_2209_11750_2022], [arxiv_2008_00235_2020], [arxiv_1511_06663_2015], [arxiv_2303_00064_2023] | `effect_to_noise_ratio` | Run matched repeated-seed comparisons and verify effect size against seed noise. |

## Included Sources

### [arxiv_2604_25452_2026] Benchmarking Logistic Regression, SVM, and LightGBM Against BiLSTM with Attention for Sentiment Analysis on Indonesian Product Reviews

- Authors: Razin Hafid Hamdi, Ivana Margareth Hutabarat, Hanna Gresia Sinaga, Luluk Muthoharoh
- Year: 2026
- URL: https://arxiv.org/abs/2604.25452v1
- arXiv: 2604.25452v1
- Categories: cs.CL
- Source: arxiv
- Used for: domain_background
- Relevance score: 8.0

Sentiment analysis of product reviews on e-commerce platforms plays a critical role in automatically understanding customer satisfaction and providing actionable insights for sellers seeking to improve product quality. This paper presents a comprehensive benchmarking study comparing a Machine Learning (ML) approach via the PyCaret AutoML framework against a Deep Learning (DL) approach based on a Bidirectional Long Short-Term Memory (BiLSTM) architecture with an Attention mechanism for binary sentiment classification on Indonesian product reviews. The dataset comprises 19,728 samples balanced equally between positive and negative reviews. For the ML approach, three prominent algorithms were evaluated via 10-fold stratified cross-validation: Logistic Regression (LR), Support Vector Machine (SVM) with a linear kernel, and Light Gradient Boosting Machine (LightGBM). Logistic Regression achieved the best ML performance with an accuracy of 97.26\% and an F1-score of 97.26\%. The BiLSTM with Attention model, evaluated on 3,946 held-out test samples, achieved an accuracy of 97.24\% and an F1-score of 97.24\%. These comparative results demonstrate that traditional ML algorithms with proper preprocessing and feature extraction can compete closely with, and even marginally outperform, more complex sequential DL architectures on high-dimensional datasets, while simultaneously offering greater computational efficiency.

### [arxiv_2209_11750_2022] Transformer-based Models to Deal with Heterogeneous Environments in Human Activity Recognition

- Authors: Sannara EK, François Portet, Philippe Lalanda
- Year: 2022
- URL: https://arxiv.org/abs/2209.11750v2
- arXiv: 2209.11750v2
- Categories: cs.CV, cs.AI, cs.LG
- Source: arxiv
- Used for: domain_background
- Relevance score: 7.0

Human Activity Recognition (HAR) on mobile devices has been demonstrated to be possible using neural models trained on data collected from the device's inertial measurement units. These models have used Convolutional Neural Networks (CNNs), Long Short-Term Memory (LSTMs), Transformers or a combination of these to achieve state-of-the-art results with real-time performance. However, these approaches have not been extensively evaluated in real-world situations where the input data may be different from the training data. This paper highlights the issue of data heterogeneity in machine learning applications and how it can hinder their deployment in pervasive settings. To address this problem, we propose and publicly release the code of two sensor-wise Transformer architectures called HART and MobileHART for Human Activity Recognition Transformer. Our experiments on several publicly available datasets show that these HART architectures outperform previous architectures with fewer floating point operations and parameters than conventional Transformers. The results also show they are more robust to changes in mobile position or device brand and hence better suited for the heterogeneous environments encountered in real-life settings. Finally, the source code has been made publicly available.

### [arxiv_2008_00235_2020] Two-step penalised logistic regression for multi-omic data with an application to cardiometabolic syndrome

- Authors: Alessandra Cabassi, Denis Seyres, Mattia Frontini, Paul D. W. Kirk
- Year: 2020
- URL: https://arxiv.org/abs/2008.00235v1
- arXiv: 2008.00235v1
- Categories: stat.AP, stat.ME, stat.ML
- Source: arxiv
- Used for: domain_background
- Relevance score: 7.0

Building classification models that predict a binary class label on the basis of high dimensional multi-omics datasets poses several challenges, due to the typically widely differing characteristics of the data layers in terms of number of predictors, type of data, and levels of noise. Previous research has shown that applying classical logistic regression with elastic-net penalty to these datasets can lead to poor results (Liu et al., 2018). We implement a two-step approach to multi-omic logistic regression in which variable selection is performed on each layer separately and a predictive model is then built using the variables selected in the first step. Here, our approach is compared to other methods that have been developed for the same purpose, and we adapt existing software for multi-omic linear regression (Zhao and Zucknick, 2020) to the logistic regression setting. Extensive simulation studies show that our approach should be preferred if the goal is to select as many relevant predictors as possible, as well as achieving prediction performances comparable to those of the best competitors. Our motivating example is a cardiometabolic syndrome dataset comprising eight 'omic data types for 2 extreme phenotype groups (10 obese and 10 lipodystrophy individuals) and 185 blood donors. Our proposed approach allows us to identify features that characterise cardiometabolic syndrome at the molecular level. R code is available at https://github.com/acabassi/logistic-regression-for-multi-omic-data.

### [arxiv_1511_06663_2015] L1 logistic regression as a feature selection step for training stable classification trees for the prediction of severity criteria in imported malaria

- Authors: Luca Talenti, Margaux Luck, Anastasia Yartseva, Nicolas Argy
- Year: 2015
- URL: https://arxiv.org/abs/1511.06663v1
- arXiv: 1511.06663v1
- Categories: cs.LG, q-bio.QM, stat.AP
- Source: arxiv
- Used for: domain_background
- Relevance score: 7.0

Multivariate classification methods using explanatory and predictive models are necessary for characterizing subgroups of patients according to their risk profiles. Popular methods include logistic regression and classification trees with performances that vary according to the nature and the characteristics of the dataset. In the context of imported malaria, we aimed at classifying severity criteria based on a heterogeneous patient population. We investigated these approaches by implementing two different strategies: L1 logistic regression (L1LR) that models a single global solution and classification trees that model multiple local solutions corresponding to discriminant subregions of the feature space. For each strategy, we built a standard model, and a sparser version of it. As an alternative to pruning, we explore a promising approach that first constrains the tree model with an L1LR-based feature selection, an approach we called L1LR-Tree. The objective is to decrease its vulnerability to small data variations by removing variables corresponding to unstable local phenomena. Our study is twofold: i) from a methodological perspective comparing the performances and the stability of the three previous methods, i.e L1LR, classification trees and L1LR-Tree, for the classification of severe forms of imported malaria, and ii) from an applied perspective improving the actual classification of severe forms of imported malaria by identifying more personalized profiles predictive of several clinical criteria based on variables dismissed for the clinical definition of the disease. The main methodological results show that the combined method L1LR-Tree builds sparse and stable models that significantly predicts the different severity criteria and outperforms all the other methods in terms of accuracy.

### [arxiv_2303_00064_2023] WEARDA: Recording Wearable Sensor Data for Human Activity Monitoring

- Authors: Richard M. K. van Dijk, Daniela Gawehns, Matthijs van Leeuwen
- Year: 2023
- URL: https://arxiv.org/abs/2303.00064v2
- arXiv: 2303.00064v2
- Categories: cs.HC, cs.CY
- Source: arxiv
- Used for: domain_background
- Relevance score: 5.0

We present WEARDA, the open source WEARable sensor Data Acquisition software package. WEARDA facilitates the acquisition of human activity data with smartwatches and is primarily aimed at researchers who require transparency, full control, and access to raw sensor data. It provides functionality to simultaneously record raw data from four sensors -- tri-axis accelerometer, tri-axis gyroscope, barometer, and GPS -- which should enable researchers to, for example, estimate energy expenditure and mine movement trajectories. A Samsung smartwatch running the Tizen OS was chosen because of 1) the required functionalities of the smartwatch software API, 2) the availability of software development tools and accessible documentation, 3) having the required sensors, and 4) the requirements on case design for acceptance by the target user group. WEARDA addresses five practical challenges concerning preparation, measurement, logistics, privacy preservation, and reproducibility to ensure efficient and errorless data collection. The software package was initially created for the project "Dementia back at the heart of the community", and has been successfully used in that context.

### [arxiv_2204_06518_2022] A pipeline and comparative study of 12 machine learning models for text classification

- Authors: Annalisa Occhipinti, Louis Rogers, Claudio Angione
- Year: 2022
- URL: https://arxiv.org/abs/2204.06518v1
- arXiv: 2204.06518v1
- Categories: cs.IR, cs.CL, cs.LG
- Source: arxiv
- Used for: domain_background
- Relevance score: 5.0

Text-based communication is highly favoured as a communication method, especially in business environments. As a result, it is often abused by sending malicious messages, e.g., spam emails, to deceive users into relaying personal information, including online accounts credentials or banking details. For this reason, many machine learning methods for text classification have been proposed and incorporated into the services of most email providers. However, optimising text classification algorithms and finding the right tradeoff on their aggressiveness is still a major research problem. We present an updated survey of 12 machine learning text classifiers applied to a public spam corpus. A new pipeline is proposed to optimise hyperparameter selection and improve the models' performance by applying specific methods (based on natural language processing) in the preprocessing stage. Our study aims to provide a new methodology to investigate and optimise the effect of different feature sizes and hyperparameters in machine learning classifiers that are widely used in text classification problems. The classifiers are tested and evaluated on different metrics including F-score (accuracy), precision, recall, and run time. By analysing all these aspects, we show how the proposed pipeline can be used to achieve a good accuracy towards spam filtering on the Enron dataset, a widely used public email corpus. Statistical tests and explainability techniques are applied to provide a robust analysis of the proposed pipeline and interpret the classification outcomes of the 12 machine learning models, also identifying words that drive the classification results. Our analysis shows that it is possible to identify an effective machine learning model to classify the Enron dataset with an F-score of 94%.

### [arxiv_1905_12285_2019] From User-independent to Personal Human Activity Recognition Models Exploiting the Sensors of a Smartphone

- Authors: Pekka Siirtola, Heli Koskimäki, Juha Röning
- Year: 2019
- URL: https://arxiv.org/abs/1905.12285v1
- arXiv: 1905.12285v1
- Categories: cs.LG, cs.HC
- Source: arxiv
- Used for: domain_background
- Relevance score: 5.0

In this study, a novel method to obtain user-dependent human activity recognition models unobtrusively by exploiting the sensors of a smartphone is presented. The recognition consists of two models: sensor fusion-based user-independent model for data labeling and single sensor-based user-dependent model for final recognition. The functioning of the presented method is tested with human activity data set, including data from accelerometer and magnetometer, and with two classifiers. Comparison of the detection accuracies of the proposed method to traditional user-independent model shows that the presented method has potential, in nine cases out of ten it is better than the traditional method, but more experiments using different sensor combinations should be made to show the full potential of the method.

### [arxiv_2404_15349_2024] A Survey on Multimodal Wearable Sensor-based Human Action Recognition

- Authors: Jianyuan Ni, Hao Tang, Syed Tousiful Haque, Yan Yan
- Year: 2024
- URL: https://arxiv.org/abs/2404.15349v1
- arXiv: 2404.15349v1
- Categories: eess.SP, cs.LG, cs.MM
- Source: arxiv
- Used for: domain_background
- Relevance score: 4.0

The combination of increased life expectancy and falling birth rates is resulting in an aging population. Wearable Sensor-based Human Activity Recognition (WSHAR) emerges as a promising assistive technology to support the daily lives of older individuals, unlocking vast potential for human-centric applications. However, recent surveys in WSHAR have been limited, focusing either solely on deep learning approaches or on a single sensor modality. In real life, our human interact with the world in a multi-sensory way, where diverse information sources are intricately processed and interpreted to accomplish a complex and unified sensing system. To give machines similar intelligence, multimodal machine learning, which merges data from various sources, has become a popular research area with recent advancements. In this study, we present a comprehensive survey from a novel perspective on how to leverage multimodal learning to WSHAR domain for newcomers and researchers. We begin by presenting the recent sensor modalities as well as deep learning approaches in HAR. Subsequently, we explore the techniques used in present multimodal systems for WSHAR. This includes inter-multimodal systems which utilize sensor modalities from both visual and non-visual systems and intra-multimodal systems that simply take modalities from non-visual systems. After that, we focus on current multimodal learning approaches that have applied to solve some of the challenges existing in WSHAR. Specifically, we make extra efforts by connecting the existing multimodal literature from other domains, such as computer vision and natural language processing, with current WSHAR area. Finally, we identify the corresponding challenges and potential research direction in current WSHAR area for further improvement.

### [arxiv_1912_00524_2019] Factor Analysis on Citation, Using a Combined Latent and Logistic Regression Model

- Authors: Namjoon Suh, Xiaoming Huo, Eric Heim, Lee Seversky
- Year: 2019
- URL: https://arxiv.org/abs/1912.00524v1
- arXiv: 1912.00524v1
- Categories: cs.LG, stat.ML
- Source: arxiv
- Used for: domain_background
- Relevance score: 4.0

We propose a combined model, which integrates the latent factor model and the logistic regression model, for the citation network. It is noticed that neither a latent factor model nor a logistic regression model alone is sufficient to capture the structure of the data. The proposed model has a latent (i.e., factor analysis) model to represents the main technological trends (a.k.a., factors), and adds a sparse component that captures the remaining ad-hoc dependence. Parameter estimation is carried out through the construction of a joint-likelihood function of edges and properly chosen penalty terms. The convexity of the objective function allows us to develop an efficient algorithm, while the penalty terms push towards a low-dimensional latent component and a sparse graphical structure. Simulation results show that the proposed method works well in practical situations. The proposed method has been applied to a real application, which contains a citation network of statisticians (Ji and Jin, 2016). Some interesting findings are reported.

### [arxiv_1905_13613_2019] Regression Networks for Meta-Learning Few-Shot Classification

- Authors: Arnout Devos, Matthias Grossglauser
- Year: 2019
- URL: https://arxiv.org/abs/1905.13613v2
- arXiv: 1905.13613v2
- Categories: cs.LG, cs.CV, stat.ML
- Source: arxiv
- Used for: domain_background
- Relevance score: 4.0

We propose regression networks for the problem of few-shot classification, where a classifier must generalize to new classes not seen in the training set, given only a small number of examples of each class. In high dimensional embedding spaces the direction of data generally contains richer information than magnitude. Next to this, state-of-the-art few-shot metric methods that compare distances with aggregated class representations, have shown superior performance. Combining these two insights, we propose to meta-learn classification of embedded points by regressing the closest approximation in every class subspace while using the regression error as a distance metric. Similarly to recent approaches for few-shot learning, regression networks reflect a simple inductive bias that is beneficial in this limited-data regime and they achieve excellent results, especially when more aggregate class representations can be formed with multiple shots.

## Excluded / Low-Relevance Sources

- None.

## Metrics Suggested By Literature

- `balanced_accuracy`
- `accuracy`
- `f1_macro`
- `precision_macro`
- `recall_macro`

## Baselines

- `literature_baseline`
- `simple_model_baseline`
