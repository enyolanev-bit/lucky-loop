# Related Work Context

Research question: Can repeated-seed evaluation show that nonlinear models do not robustly outperform logistic regression on public EEG sensor classification data?

## Search Protocol

- API: official arXiv Atom API (`https://export.arxiv.org/api/query`).
- Rate limit: one request every 3.1s.
- Curated fallback: known core papers are included and deduplicated against arXiv metadata.
- Citation stability: arXiv IDs and versions are preserved when available.

## Search Queries

- Can repeated-seed evaluation show that nonlinear models do not robustly outperform logistic regression on public EEG sensor classification data?
- repeated seed evaluation show nonlinear models robustly outperform logistic regression machine learning classification
- repeated seed evaluation show nonlinear models robustly outperform logistic regression logistic regression random forest svm gradient boosting
- repeated seed evaluation show nonlinear models robustly outperform logistic regression robustness repeated seeds cross validation
- "human activity recognition" sensor classification machine learning
- "EEG eye state" classification machine learning
- wearable sensor classification logistic regression random forest svm

## Source -> Gap -> Metric -> Experiment

| Gap | Sources | Metric | Experiment |
|---|---|---|---|
| The literature context must establish whether nonlinear models are actually expected to improve EEG/sensor classification over simple baselines. | [arxiv_2409_14508_2024], [arxiv_2209_11750_2022], [arxiv_2604_25452_2026], [arxiv_1511_06663_2015], [arxiv_2606_00815_2026] | `domain_source_coverage` | Compare baseline and nonlinear model families on a dataset selected for the user question. |
| Claims about EEG/sensor classification performance require robustness checks across splits or seeds, not a single score. | [arxiv_2409_14508_2024], [arxiv_2209_11750_2022], [arxiv_2604_25452_2026], [arxiv_1511_06663_2015], [arxiv_2606_00815_2026] | `effect_to_noise_ratio` | Run matched repeated-seed comparisons and verify effect size against seed noise. |

## Included Sources

### [arxiv_2409_14508_2024] Evaluating Machine Learning Models for Supernova Gravitational Wave Signal Classification

- Authors: Y. Sultan Abylkairov, Matthew C. Edwards, Daniil Orel, Ayan Mitra
- Year: 2024
- URL: https://arxiv.org/abs/2409.14508v2
- arXiv: 2409.14508v2
- Categories: astro-ph.HE, gr-qc
- Source: arxiv
- Used for: domain_background
- Relevance score: 11.0

We investigate the potential of using gravitational wave (GW) signals from rotating core-collapse supernovae to probe the equation of state (EOS) of nuclear matter. By generating GW signals from simulations with various EOSs, we train machine learning models to classify them and evaluate their performance. Our study builds on previous work by examining how different machine learning models, parameters, and data preprocessing techniques impact classification accuracy. We test convolutional and recurrent neural networks, as well as six classical algorithms: random forest, support vector machines, naïve Bayes, logistic regression, $k$-nearest neighbors, and eXtreme gradient boosting. All models, except naïve Bayes, achieve over 90 per cent accuracy on our dataset. Additionally, we assess the impact of approximating the GW signal using the general relativistic effective potential (GREP) on EOS classification. We find that models trained on GREP data exhibit low classification accuracy. However, normalizing time by the peak signal frequency, which partially compensates for the absence of the time dilation effect in GREP, leads to a notable improvement in accuracy. Despite this, the accuracy does not exceed 70 per cent, suggesting that GREP lacks the precision necessary for EOS classification. Finally, our study has several limitations, including the omission of detector noise and the focus on a single progenitor mass model, which will be addressed in future works.

### [arxiv_2209_11750_2022] Transformer-based Models to Deal with Heterogeneous Environments in Human Activity Recognition

- Authors: Sannara EK, François Portet, Philippe Lalanda
- Year: 2022
- URL: https://arxiv.org/abs/2209.11750v2
- arXiv: 2209.11750v2
- Categories: cs.CV, cs.AI, cs.LG
- Source: arxiv
- Used for: domain_background
- Relevance score: 9.0

Human Activity Recognition (HAR) on mobile devices has been demonstrated to be possible using neural models trained on data collected from the device's inertial measurement units. These models have used Convolutional Neural Networks (CNNs), Long Short-Term Memory (LSTMs), Transformers or a combination of these to achieve state-of-the-art results with real-time performance. However, these approaches have not been extensively evaluated in real-world situations where the input data may be different from the training data. This paper highlights the issue of data heterogeneity in machine learning applications and how it can hinder their deployment in pervasive settings. To address this problem, we propose and publicly release the code of two sensor-wise Transformer architectures called HART and MobileHART for Human Activity Recognition Transformer. Our experiments on several publicly available datasets show that these HART architectures outperform previous architectures with fewer floating point operations and parameters than conventional Transformers. The results also show they are more robust to changes in mobile position or device brand and hence better suited for the heterogeneous environments encountered in real-life settings. Finally, the source code has been made publicly available.

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

### [arxiv_1511_06663_2015] L1 logistic regression as a feature selection step for training stable classification trees for the prediction of severity criteria in imported malaria

- Authors: Luca Talenti, Margaux Luck, Anastasia Yartseva, Nicolas Argy
- Year: 2015
- URL: https://arxiv.org/abs/1511.06663v1
- arXiv: 1511.06663v1
- Categories: cs.LG, q-bio.QM, stat.AP
- Source: arxiv
- Used for: domain_background
- Relevance score: 8.0

Multivariate classification methods using explanatory and predictive models are necessary for characterizing subgroups of patients according to their risk profiles. Popular methods include logistic regression and classification trees with performances that vary according to the nature and the characteristics of the dataset. In the context of imported malaria, we aimed at classifying severity criteria based on a heterogeneous patient population. We investigated these approaches by implementing two different strategies: L1 logistic regression (L1LR) that models a single global solution and classification trees that model multiple local solutions corresponding to discriminant subregions of the feature space. For each strategy, we built a standard model, and a sparser version of it. As an alternative to pruning, we explore a promising approach that first constrains the tree model with an L1LR-based feature selection, an approach we called L1LR-Tree. The objective is to decrease its vulnerability to small data variations by removing variables corresponding to unstable local phenomena. Our study is twofold: i) from a methodological perspective comparing the performances and the stability of the three previous methods, i.e L1LR, classification trees and L1LR-Tree, for the classification of severe forms of imported malaria, and ii) from an applied perspective improving the actual classification of severe forms of imported malaria by identifying more personalized profiles predictive of several clinical criteria based on variables dismissed for the clinical definition of the disease. The main methodological results show that the combined method L1LR-Tree builds sparse and stable models that significantly predicts the different severity criteria and outperforms all the other methods in terms of accuracy.

### [arxiv_2606_00815_2026] OmniEEG-Bench: A Standardized Evaluation Benchmark for EEG Foundation Models

- Authors: Ziling Lu, Zongsheng Li, Xinke Shen, Kexin Lou
- Year: 2026
- URL: https://arxiv.org/abs/2606.00815v1
- arXiv: 2606.00815v1
- Categories: cs.LG
- Source: arxiv
- Used for: domain_background
- Relevance score: 6.0

Electroencephalography (EEG) supports a variety of brain-computer interface (BCI) tasks ranging from brain-state monitoring to human-LLM interactions. EEG foundation models are emerging, but evaluation remains fragmented due to heterogeneous datasets and nconsistent task protocols. Here, we introduce OmniEEG-Bench, a unified benchmark and downstream task roadmap for EEG foundation models (FMs). It organizes evaluation of EEG FMs into six task families spanning (i) signal reliability, (ii) biometrics and disease, (iii) consciousness and state, (iv) cognition and emotion, (v) naturalistic stimulus decoding, and (vi) motor and interaction, introducing a new generation of tasks not systematically benchmarked in prior EEG FM work. OmniEEG-Bench standardizes model deployment, task definitions, and metrics through a task-card specification, and unifies 54 EEG datasets with consistent evaluation protocols. We benchmark 10 representative EEG foundation models and report a leaderboard that covers diverse evaluation settings. Both pretraining dataset diversity and model size are significantly associated with better average ranks across datasets, revealing scaling-law behavior in EEG foundation models (Figure 1). These results suggest that scaling EEG foundation models requires not only larger architectures but also broader and more diverse pretraining data. The benchmark code is available at https://github.com/ncclab-sustech/omni-eegbench.git.

### [arxiv_1905_12285_2019] From User-independent to Personal Human Activity Recognition Models Exploiting the Sensors of a Smartphone

- Authors: Pekka Siirtola, Heli Koskimäki, Juha Röning
- Year: 2019
- URL: https://arxiv.org/abs/1905.12285v1
- arXiv: 1905.12285v1
- Categories: cs.LG, cs.HC
- Source: arxiv
- Used for: domain_background
- Relevance score: 6.0

In this study, a novel method to obtain user-dependent human activity recognition models unobtrusively by exploiting the sensors of a smartphone is presented. The recognition consists of two models: sensor fusion-based user-independent model for data labeling and single sensor-based user-dependent model for final recognition. The functioning of the presented method is tested with human activity data set, including data from accelerometer and magnetometer, and with two classifiers. Comparison of the detection accuracies of the proposed method to traditional user-independent model shows that the presented method has potential, in nine cases out of ten it is better than the traditional method, but more experiments using different sensor combinations should be made to show the full potential of the method.

### [arxiv_2407_20247_2024] How Homogenizing the Channel-wise Magnitude Can Enhance EEG Classification Model?

- Authors: Huyen Ngo, Khoi Do, Duong Nguyen, Viet Dung Nguyen
- Year: 2024
- URL: https://arxiv.org/abs/2407.20247v1
- arXiv: 2407.20247v1
- Categories: eess.SP, cs.AI, cs.LG
- Source: arxiv
- Used for: domain_background
- Relevance score: 5.0

A significant challenge in the electroencephalogram EEG lies in the fact that current data representations involve multiple electrode signals, resulting in data redundancy and dominant lead information. However extensive research conducted on EEG classification focuses on designing model architectures without tackling the underlying issues. Otherwise, there has been a notable gap in addressing data preprocessing for EEG, leading to considerable computational overhead in Deep Learning (DL) processes. In light of these issues, we propose a simple yet effective approach for EEG data pre-processing. Our method first transforms the EEG data into an encoded image by an Inverted Channel-wise Magnitude Homogenization (ICWMH) to mitigate inter-channel biases. Next, we apply the edge detection technique on the EEG-encoded image combined with skip connection to emphasize the most significant transitions in the data while preserving structural and invariant information. By doing so, we can improve the EEG learning process efficiently without using a huge DL network. Our experimental evaluations reveal that we can significantly improve (i.e., from 2% to 5%) over current baselines.

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

### [arxiv_1912_00524_2019] Factor Analysis on Citation, Using a Combined Latent and Logistic Regression Model

- Authors: Namjoon Suh, Xiaoming Huo, Eric Heim, Lee Seversky
- Year: 2019
- URL: https://arxiv.org/abs/1912.00524v1
- arXiv: 1912.00524v1
- Categories: stat.ML, cs.LG
- Source: arxiv
- Used for: domain_background
- Relevance score: 5.0

We propose a combined model, which integrates the latent factor model and the logistic regression model, for the citation network. It is noticed that neither a latent factor model nor a logistic regression model alone is sufficient to capture the structure of the data. The proposed model has a latent (i.e., factor analysis) model to represents the main technological trends (a.k.a., factors), and adds a sparse component that captures the remaining ad-hoc dependence. Parameter estimation is carried out through the construction of a joint-likelihood function of edges and properly chosen penalty terms. The convexity of the objective function allows us to develop an efficient algorithm, while the penalty terms push towards a low-dimensional latent component and a sparse graphical structure. Simulation results show that the proposed method works well in practical situations. The proposed method has been applied to a real application, which contains a citation network of statisticians (Ji and Jin, 2016). Some interesting findings are reported.

### [arxiv_2506_03524_2025] Seed-Coder: Let the Code Model Curate Data for Itself

- Authors: ByteDance Seed, Yuyu Zhang, Jing Su, Yifan Sun
- Year: 2025
- URL: https://arxiv.org/abs/2506.03524v2
- arXiv: 2506.03524v2
- Categories: cs.CL, cs.SE
- Source: arxiv
- Used for: domain_background
- Relevance score: 4.0

Code data in large language model (LLM) pretraining is recognized crucial not only for code-related tasks but also for enhancing general intelligence of LLMs. Current open-source LLMs often heavily rely on human effort to produce their code pretraining data, such as employing hand-crafted filtering rules tailored to individual programming languages, or using human-annotated data to train quality filters. However, these approaches are inherently limited in scalability, prone to subjective biases, and costly to extend and maintain across diverse programming languages. To address these challenges, we introduce Seed-Coder, a series of open-source LLMs comprising base, instruct and reasoning models of 8B size, minimizing human involvement in data construction. Our code pretraining data is produced by a model-centric data pipeline, which predominantly leverages LLMs for scoring and filtering code data. The instruct model is further trained via supervised fine-tuning and preference optimization, and the reasoning model leverages Long-Chain-of-Thought (LongCoT) reinforcement learning to improve multi-step code reasoning. Seed-Coder achieves state-of-the-art results among open-source models of similar size and even surpasses some much larger models, demonstrating superior performance in code generation, code completion, code editing, code reasoning, and software engineering tasks.

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
