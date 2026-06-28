# Related Work Context

Research question: UI smoke test autoresearch venv launch

## Search Protocol

- API: official arXiv Atom API (`https://export.arxiv.org/api/query`).
- Rate limit: one request every 3.1s.
- Curated fallback: known core papers are included and deduplicated against arXiv metadata.
- Citation stability: arXiv IDs and versions are preserved when available.

## Search Queries

- UI smoke test autoresearch venv launch
- smoke test autoresearch venv launch machine learning classification
- smoke test autoresearch venv launch logistic regression random forest svm gradient boosting
- smoke test autoresearch venv launch robustness repeated seeds cross validation

## Source -> Gap -> Metric -> Experiment

| Gap | Sources | Metric | Experiment |
|---|---|---|---|
| The literature context must establish whether nonlinear models are actually expected to improve the selected ML task over simple baselines. | [arxiv_2604_25452_2026], [arxiv_2208_06828_2022], [arxiv_2306_04338_2023], [arxiv_1912_00524_2019], [arxiv_1905_08737_2019] | `domain_source_coverage` | Compare baseline and nonlinear model families on a dataset selected for the user question. |
| Claims about the selected ML task performance require robustness checks across splits or seeds, not a single score. | [arxiv_2604_25452_2026], [arxiv_2208_06828_2022], [arxiv_2306_04338_2023], [arxiv_1912_00524_2019], [arxiv_1905_08737_2019] | `effect_to_noise_ratio` | Run matched repeated-seed comparisons and verify effect size against seed noise. |

## Included Sources

### [arxiv_2604_25452_2026] Benchmarking Logistic Regression, SVM, and LightGBM Against BiLSTM with Attention for Sentiment Analysis on Indonesian Product Reviews

- Authors: Razin Hafid Hamdi, Ivana Margareth Hutabarat, Hanna Gresia Sinaga, Luluk Muthoharoh
- Year: 2026
- URL: https://arxiv.org/abs/2604.25452v1
- arXiv: 2604.25452v1
- Categories: cs.CL
- Source: arxiv
- Used for: domain_background
- Relevance score: 5.0

Sentiment analysis of product reviews on e-commerce platforms plays a critical role in automatically understanding customer satisfaction and providing actionable insights for sellers seeking to improve product quality. This paper presents a comprehensive benchmarking study comparing a Machine Learning (ML) approach via the PyCaret AutoML framework against a Deep Learning (DL) approach based on a Bidirectional Long Short-Term Memory (BiLSTM) architecture with an Attention mechanism for binary sentiment classification on Indonesian product reviews. The dataset comprises 19,728 samples balanced equally between positive and negative reviews. For the ML approach, three prominent algorithms were evaluated via 10-fold stratified cross-validation: Logistic Regression (LR), Support Vector Machine (SVM) with a linear kernel, and Light Gradient Boosting Machine (LightGBM). Logistic Regression achieved the best ML performance with an accuracy of 97.26\% and an F1-score of 97.26\%. The BiLSTM with Attention model, evaluated on 3,946 held-out test samples, achieved an accuracy of 97.24\% and an F1-score of 97.24\%. These comparative results demonstrate that traditional ML algorithms with proper preprocessing and feature extraction can compete closely with, and even marginally outperform, more complex sequential DL architectures on high-dimensional datasets, while simultaneously offering greater computational efficiency.

### [arxiv_2208_06828_2022] Multinomial Logistic Regression Algorithms via Quadratic Gradient

- Authors: John Chiang
- Year: 2022
- URL: https://arxiv.org/abs/2208.06828v2
- arXiv: 2208.06828v2
- Categories: cs.LG, math.OC
- Source: arxiv
- Used for: domain_background
- Relevance score: 3.0

Multinomial logistic regression, also known by other names such as multiclass logistic regression and softmax regression, is a fundamental classification method that generalizes binary logistic regression to multiclass problems. A recently work proposed a faster gradient called $\texttt{quadratic gradient}$ that can accelerate the binary logistic regression training, and presented an enhanced Nesterov's accelerated gradient (NAG) method for binary logistic regression. In this paper, we extend this work to multiclass logistic regression and propose an enhanced Adaptive Gradient Algorithm (Adagrad) that can accelerate the original Adagrad method. We test the enhanced NAG method and the enhanced Adagrad method on some multiclass-problem datasets. Experimental results show that both enhanced methods converge faster than their original ones respectively.

### [arxiv_2306_04338_2023] Changing Data Sources in the Age of Machine Learning for Official Statistics

- Authors: Cedric De Boom, Michael Reusens
- Year: 2023
- URL: https://arxiv.org/abs/2306.04338v1
- arXiv: 2306.04338v1
- Categories: stat.ML, cs.LG
- Source: arxiv
- Used for: domain_background
- Relevance score: 1.0

Data science has become increasingly essential for the production of official statistics, as it enables the automated collection, processing, and analysis of large amounts of data. With such data science practices in place, it enables more timely, more insightful and more flexible reporting. However, the quality and integrity of data-science-driven statistics rely on the accuracy and reliability of the data sources and the machine learning techniques that support them. In particular, changes in data sources are inevitable to occur and pose significant risks that are crucial to address in the context of machine learning for official statistics. This paper gives an overview of the main risks, liabilities, and uncertainties associated with changing data sources in the context of machine learning for official statistics. We provide a checklist of the most prevalent origins and causes of changing data sources; not only on a technical level but also regarding ownership, ethics, regulation, and public perception. Next, we highlight the repercussions of changing data sources on statistical reporting. These include technical effects such as concept drift, bias, availability, validity, accuracy and completeness, but also the neutrality and potential discontinuation of the statistical offering. We offer a few important precautionary measures, such as enhancing robustness in both data sourcing and statistical techniques, and thorough monitoring. In doing so, machine learning-based official statistics can maintain integrity, reliability, consistency, and relevance in policy-making, decision-making, and public discourse.

### [arxiv_1912_00524_2019] Factor Analysis on Citation, Using a Combined Latent and Logistic Regression Model

- Authors: Namjoon Suh, Xiaoming Huo, Eric Heim, Lee Seversky
- Year: 2019
- URL: https://arxiv.org/abs/1912.00524v1
- arXiv: 1912.00524v1
- Categories: stat.ML, cs.LG
- Source: arxiv
- Used for: domain_background
- Relevance score: 1.0

We propose a combined model, which integrates the latent factor model and the logistic regression model, for the citation network. It is noticed that neither a latent factor model nor a logistic regression model alone is sufficient to capture the structure of the data. The proposed model has a latent (i.e., factor analysis) model to represents the main technological trends (a.k.a., factors), and adds a sparse component that captures the remaining ad-hoc dependence. Parameter estimation is carried out through the construction of a joint-likelihood function of edges and properly chosen penalty terms. The convexity of the objective function allows us to develop an efficient algorithm, while the penalty terms push towards a low-dimensional latent component and a sparse graphical structure. Simulation results show that the proposed method works well in practical situations. The proposed method has been applied to a real application, which contains a citation network of statisticians (Ji and Jin, 2016). Some interesting findings are reported.

### [arxiv_1905_08737_2019] On the marginal likelihood and cross-validation

- Authors: Edwin Fong, Chris Holmes
- Year: 2019
- URL: https://arxiv.org/abs/1905.08737v2
- arXiv: 1905.08737v2
- Categories: stat.ME, stat.ML
- Source: arxiv
- Used for: domain_background
- Relevance score: 1.0

In Bayesian statistics, the marginal likelihood, also known as the evidence, is used to evaluate model fit as it quantifies the joint probability of the data under the prior. In contrast, non-Bayesian models are typically compared using cross-validation on held-out data, either through $k$-fold partitioning or leave-$p$-out subsampling. We show that the marginal likelihood is formally equivalent to exhaustive leave-$p$-out cross-validation averaged over all values of $p$ and all held-out test sets when using the log posterior predictive probability as the scoring rule. Moreover, the log posterior predictive is the only coherent scoring rule under data exchangeability. This offers new insight into the marginal likelihood and cross-validation and highlights the potential sensitivity of the marginal likelihood to the choice of the prior. We suggest an alternative approach using cumulative cross-validation following a preparatory training phase. Our work has connections to prequential analysis and intrinsic Bayes factors but is motivated through a different course.

### [arxiv_1612_03614_2016] Increasing Launch Efficiency with the PEGASUS Launcher

- Authors: S. Hundertmark, G. Vincent, D. Simicic, M. Schneider
- Year: 2016
- URL: https://arxiv.org/abs/1612.03614v1
- arXiv: 1612.03614v1
- Categories: physics.ins-det, physics.plasm-ph
- Source: arxiv
- Used for: domain_background
- Relevance score: 1.0

In the real world application of railguns, the launch efficiency is one of the most important parameters. This efficiency directly relates to the capacity of the electrical energy storage that is needed for the launch. In this study, the rail/armature contact behavior for two different armature technologies was compared. To this end, experiments using aluminum c-shaped armature and copper brush armature type projectiles were performed under same initial conditions. The c-shaped armature type showed a superior behavior with respect to electrical contact to the rails and in acceleration. A 300 g projectile with a c-shaped armature reached a velocity of 3100 m/s and an overall launch efficiency including the power supply of 41%. This is to be compared to 2500 m/s and 23% for the launching of a projectile using a brush armature.

### [arxiv_1612_04143_2016] Developing a Launch Package for the PEGASUS Launcher

- Authors: S. Hundertmark, G. Vincent, D. Simicic
- Year: 2016
- URL: https://arxiv.org/abs/1612.04143v1
- arXiv: 1612.04143v1
- Categories: physics.plasm-ph
- Source: arxiv
- Used for: domain_background
- Relevance score: 1.0

Railguns are capable to far exceed the muzzle energies of current naval deck guns. Therefore one of the most promising scenario for the future application of railguns in naval warfare is the long range artillery. Hypervelocity projectiles being propelled to velocities above 2 km/s reach targets at distances of 200 km or more. At the French-German Research Institute the PEGASUS launcher is used for investigations with respect to this scenario. The 6 m long barrel has a square caliber of 40 mm. The power supply unit is able to deliver 10 MJ to the gun. Within this investigation, a complete launch package is being developed and experiments are performed that aim at showing that this package can be accelerated to velocities ranging from 2000 m/s to 2500 m/s. A launch package consists out of an armature, a sabot and the projectile. The armature ensures the electrical contact during launch and pushes the sabot with its payload through the barrel. The sabot guides and protects the payload during the acceleration. At the same time the accelerating forces generated at the armature needs to be transferred to the projectile. After the launch package has left the barrel, the sabot should open and release its payload, the projectile into free-flight. Here the current status of the launch package development and results from experiments with the PEGASUS railgun are presented.

## Excluded / Low-Relevance Sources

- Design-Based Cross-Validation for Comparing Small Area Estimators (https://arxiv.org/abs/2604.23464v3): low relevance to the domain research question; score=0.0
- UI-R1: Enhancing Efficient Action Prediction of GUI Agents by Reinforcement Learning (https://arxiv.org/abs/2503.21620v5): low relevance to the domain research question; score=0.0
- Learning Curves for Decision Making in Supervised Machine Learning: A Survey (https://arxiv.org/abs/2201.12150v2): low relevance to the domain research question; score=0.0
- DOME: Recommendations for supervised machine learning validation in biology (https://arxiv.org/abs/2006.16189v4): low relevance to the domain research question; score=0.0

## Metrics Suggested By Literature

- `balanced_accuracy`
- `accuracy`
- `f1_macro`
- `precision_macro`
- `recall_macro`

## Baselines

- `literature_baseline`
- `simple_model_baseline`
