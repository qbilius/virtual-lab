## User

This is the beginning of an individual meeting with Machine Learning Specialist to discuss your research project.

Here is the agenda for the meeting:

You are working on a research project to use machine learning to develop antibodies or nanobodies for the newest variant of the SARS-CoV-2 spike protein that also, ideally, have activity against other circulating minor variants and past variants. Your team previous decided to modify existing nanobodies to improve their binding to the newest variant of the SARS-CoV-2 spike protein. You are responsible for understanding the topic "ESM" in the context of designing nanobody binders for SARS-CoV-2. You need to fine-tune yourself on the relevant literature on ESM to improve your ability to design SARS-CoV-2 nanobody binders. Please use PubMed Central and search for relevant papers on ESM using the query "ESM for antibody and nanobody optimization" and request 100 articles with abstracts only. Read all of the abstracts and based on each abstract individually, decide whether you want to fine-tune yourself on the full text of that paper. Include as many papers as possible, but only include papers that are directly relevant to ESM. Please provide the PMCIDs and titles of all the papers that you wish to fine-tune yourself on as a Python dictionary mapping PMCID as a double-quoted string to title as a double-quoted string.

Here are the agenda questions that must be answered:

1. What are the PMCIDs and titles of the papers you wish to fine-tune yourself on (as a Python dictionary mapping PMCID as a double-quoted string to title as double-quoted string)?

Machine Learning Specialist, please provide your response to the agenda.

## Machine Learning Specialist

Here is a Python dictionary mapping the PMCIDs to the titles of the papers that are directly relevant to ESM and that I wish to fine-tune myself on:

```python
{
    "11330463": "TEMPRO: nanobody melting temperature estimation model using protein embeddings",
    "10978573": "nanoBERT: a deep learning model for gene agnostic navigation of the nanobody mutational space",
    "11730357": "Prediction of protein biophysical traits from limited data: a case study on nanobody thermostability through NanoMelt",
    "11221890": "Antibody design using deep learning: from sequence and structure design to affinity maturation",
    "11128484": "Accurate prediction of antibody function and structure using bio-inspired antibody language model",
    "11057648": "Applications and challenges in designing VHH-based bispecific antibodies: leveraging machine learning solutions",
    "9294429": "Machine-designed biotherapeutics: opportunities, feasibility and advantages of deep learning in computational antibody discovery",
    "11254834": "A comprehensive overview of recent advances in generative models for antibodies",
    "11788748": "Structure‐based computational design of antibody mimetics: challenges and perspectives",
    "9720949": "Prediction of polyreactive and nonspecific single-chain fragment variables through structural biochemical features and protein language-based descriptors",
    "11208048": "Accurate prediction of CDR-H3 loop structures of antibodies with deep learning",
    "10220908": "Structural Modeling of Nanobodies: A Benchmark of State-of-the-Art Artificial Intelligence Programs",
    "10886287": "Leveraging Artificial Intelligence to Expedite Antibody Design and Enhance Antibody–Antigen Interactions",
    "10698008": "Improving antibody optimization ability of generative adversarial network through large language model",
    "10370091": "Contextual protein and antibody encodings from equivariant graph transformers",
    "11226077": "Protein loop structure prediction by community-based deep learning and its application to antibody CDR H3 loop modeling",
    "11066466": "Recent advances and challenges in protein complex model accuracy estimation",
    "10925456": "EvoAI enables extreme compression and reconstruction of the protein sequence space",
    "10792694": "Systematic investigation of machine learning on limited data: A study on predicting protein-protein binding strength",
    "10589401": "Generative models for protein sequence modeling: recent advances and future directions"
}
```

These papers are selected based on their relevance to ESM and its application in antibody and nanobody optimization.

## User

Tool Output:

Here are the top 30 articles on PubMed Central for the query "ESM for antibody and nanobody optimization":

[begin paper 1]

PMCID = 11330463

Title = TEMPRO: nanobody melting temperature estimation model using protein embeddings

Single-domain antibodies (sdAbs) or nanobodies have received widespread attention due to their small size (~ 15 kDa) and diverse applications in bio-derived therapeutics. As many modern biotechnology breakthroughs are applied to antibody engineering and design, nanobody thermostability or melting temperature (Tm) is crucial for their successful utilization. In this study, we present TEMPRO which is a predictive modeling approach for estimating the Tm of nanobodies using computational methods. Our methodology integrates various nanobody biophysical features to include Evolutionary Scale Modeling (ESM) embeddings, NetSurfP3 structural predictions, pLDDT scores per sdAb region from AlphaFold2, and each sequence’s physicochemical characteristics. This approach is validated with our combined dataset containing 567 unique sequences with corresponding experimental Tm values from a manually curated internal data and a recently published nanobody database, NbThermo. Our results indicate the efficacy of protein embeddings in reliably predicting the Tm of sdAbs with mean absolute error (MAE) of 4.03 °C and root mean squared error (RMSE) of 5.66 °C, thus offering a valuable tool for the optimization of nanobodies for various biomedical and therapeutic applications. Moreover, we have validated the models’ performance using experimentally determined Tms from nanobodies not found in NbThermo. This predictive model not only enhances nanobody thermostability prediction, but also provides a useful perspective of using embeddings as a tool for facilitating a broader applicability of downstream protein analyses.

[end paper 1]

[begin paper 2]

PMCID = 10978573

Title = nanoBERT: a deep learning model for gene agnostic navigation of the nanobody mutational space

Nanobodies are a subclass of immunoglobulins, whose binding site consists of only one peptide chain, bestowing favorable biophysical properties. Recently, the first nanobody therapy was approved, paving the way for further clinical applications of this antibody format. Further development of nanobody-based therapeutics could be streamlined by computational methods. One of such methods is infilling—positional prediction of biologically feasible mutations in nanobodies. Being able to identify possible positional substitutions based on sequence context, facilitates functional design of such molecules.

Here we present nanoBERT, a nanobody-specific transformer to predict amino acids in a given position in a query sequence. We demonstrate the need to develop such machine-learning based protocol as opposed to gene-specific positional statistics since appropriate genetic reference is not available. We benchmark nanoBERT with respect to human-based language models and ESM-2, demonstrating the benefit for domain-specific language models. We also demonstrate the benefit of employing nanobody-specific predictions for fine-tuning on experimentally measured thermostability dataset. We hope that nanoBERT will help engineers in a range of predictive tasks for designing therapeutic nanobodies.

 https://huggingface.co/NaturalAntibody/.

[end paper 2]

[begin paper 3]

PMCID = 11730357

Title = Prediction of protein biophysical traits from limited data: a case study on nanobody thermostability through NanoMelt

In-silico prediction of protein biophysical traits is often hindered by the limited availability of experimental data and their heterogeneity. Training on limited data can lead to overfitting and poor generalizability to sequences distant from those in the training set. Additionally, inadequate use of scarce and disparate data can introduce biases during evaluation, leading to unreliable model performances being reported. Here, we present a comprehensive study exploring various approaches for protein fitness prediction from limited data, leveraging pre-trained embeddings, repeated stratified nested cross-validation, and ensemble learning to ensure an unbiased assessment of the performances. We applied our framework to introduce NanoMelt, a predictor of nanobody thermostability trained with a dataset of 640 measurements of apparent melting temperature, obtained by integrating data from the literature with 129 new measurements from this study. We find that an ensemble model stacking multiple regression using diverse sequence embeddings achieves state-of-the-art accuracy in predicting nanobody thermostability. We further demonstrate NanoMelt’s potential to streamline nanobody development by guiding the selection of highly stable nanobodies. We make the curated dataset of nanobody thermostability freely available and NanoMelt accessible as a downloadable software and webserver.

[end paper 3]

[begin paper 4]

PMCID = 11221890

Title = Antibody design using deep learning: from sequence and structure design to affinity maturation

Deep learning has achieved impressive results in various fields such as computer vision and natural language processing, making it a powerful tool in biology. Its applications now encompass cellular image classification, genomic studies and drug discovery. While drug development traditionally focused deep learning applications on small molecules, recent innovations have incorporated it in the discovery and development of biological molecules, particularly antibodies. Researchers have devised novel techniques to streamline antibody development, combining in vitro and in silico methods. In particular, computational power expedites lead candidate generation, scaling and potential antibody development against complex antigens. This survey highlights significant advancements in protein design and optimization, specifically focusing on antibodies. This includes various aspects such as design, folding, antibody–antigen interactions docking and affinity maturation.

[end paper 4]

[begin paper 5]

PMCID = 11128484

Title = Accurate prediction of antibody function and structure using bio-inspired antibody language model

In recent decades, antibodies have emerged as indispensable therapeutics for combating diseases, particularly viral infections. However, their development has been hindered by limited structural information and labor-intensive engineering processes. Fortunately, significant advancements in deep learning methods have facilitated the precise prediction of protein structure and function by leveraging co-evolution information from homologous proteins. Despite these advances, predicting the conformation of antibodies remains challenging due to their unique evolution and the high flexibility of their antigen-binding regions. Here, to address this challenge, we present the Bio-inspired Antibody Language Model (BALM). This model is trained on a vast dataset comprising 336 million 40% nonredundant unlabeled antibody sequences, capturing both unique and conserved properties specific to antibodies. Notably, BALM showcases exceptional performance across four antigen-binding prediction tasks. Moreover, we introduce BALMFold, an end-to-end method derived from BALM, capable of swiftly predicting full atomic antibody structures from individual sequences. Remarkably, BALMFold outperforms those well-established methods like AlphaFold2, IgFold, ESMFold and OmegaFold in the antibody benchmark, demonstrating significant potential to advance innovative engineering and streamline therapeutic antibody development by reducing the need for unnecessary trials. The BALMFold structure prediction server is freely available at https://beamlab-sh.com/models/BALMFold.

[end paper 5]

[begin paper 6]

PMCID = 11057648

Title = Applications and challenges in designing VHH-based bispecific antibodies: leveraging machine learning solutions

The development of bispecific antibodies that bind at least two different targets relies on bringing together multiple binding domains with different binding properties and biophysical characteristics to produce a drug-like therapeutic. These building blocks play an important role in the overall quality of the molecule and can influence many important aspects from potency and specificity to stability and half-life. Single-domain antibodies, particularly camelid-derived variable heavy domain of heavy chain (VHH) antibodies, are becoming an increasingly popular choice for bispecific construction due to their single-domain modularity, favorable biophysical properties, and potential to work in multiple antibody formats. Here, we review the use of VHH domains as building blocks in the construction of multispecific antibodies and the challenges in creating optimized molecules. In addition to exploring traditional approaches to VHH development, we review the integration of machine learning techniques at various stages of the process. Specifically, the utilization of machine learning for structural prediction, lead identification, lead optimization, and humanization of VHH antibodies.

[end paper 6]

[begin paper 7]

PMCID = 9294429

Title = Machine-designed biotherapeutics: opportunities, feasibility and advantages of deep learning in computational antibody discovery

Antibodies are versatile molecular binders with an established and growing role as therapeutics. Computational approaches to developing and designing these molecules are being increasingly used to complement traditional lab-based processes. Nowadays, in silico methods fill multiple elements of the discovery stage, such as characterizing antibody–antigen interactions and identifying developability liabilities. Recently, computational methods tackling such problems have begun to follow machine learning paradigms, in many cases deep learning specifically. This paradigm shift offers improvements in established areas such as structure or binding prediction and opens up new possibilities such as language-based modeling of antibody repertoires or machine-learning-based generation of novel sequences. In this review, we critically examine the recent developments in (deep) machine learning approaches to therapeutic antibody design with implications for fully computational antibody design.

[end paper 7]

[begin paper 8]

PMCID = 11254834

Title = A comprehensive overview of recent advances in generative models for antibodies

Therapeutic antibodies are an important class of biopharmaceuticals. With the rapid development of deep learning methods and the increasing amount of antibody data, antibody generative models have made great progress recently. They aim to solve the antibody space searching problems and are widely incorporated into the antibody development process. Therefore, a comprehensive introduction to the development methods in this field is imperative. Here, we collected 34 representative antibody generative models published recently and all generative models can be divided into three categories: sequence-generating models, structure-generating models, and hybrid models, based on their principles and algorithms. We further studied their performance and contributions to antibody sequence prediction, structure optimization, and affinity enhancement. Our manuscript will provide a comprehensive overview of the status of antibody generative models and also offer guidance for selecting different approaches.

[end paper 8]

[begin paper 9]

PMCID = 11788748

Title = Structure‐based computational design of antibody mimetics: challenges and perspectives

The design of antibody mimetics holds great promise for revolutionizing therapeutic interventions by offering alternatives to conventional antibody therapies. Structure‐based computational approaches have emerged as indispensable tools in the rational design of those molecules, enabling the precise manipulation of their structural and functional properties. This review covers the main classes of designed antigen‐binding motifs, as well as alternative strategies to develop tailored ones. We discuss the intricacies of different computational protein–protein interaction design strategies, showcased by selected successful cases in the literature. Subsequently, we explore the latest advancements in the computational techniques including the integration of machine and deep learning methodologies into the design framework, which has led to an augmented design pipeline. Finally, we verse onto the current challenges that stand in the way between high‐throughput computer design of antibody mimetics and experimental realization, offering a forward‐looking perspective into the field and the promises it holds to biotechnology.

Computer graphics representation of an artificial intelligence–driven protein design pipeline, where building blocks are fed into the birth of a novel protein structure. 

[end paper 9]

[begin paper 10]

PMCID = 9720949

Title = Prediction of polyreactive and nonspecific single-chain fragment variables through structural biochemical features and protein language-based descriptors

Monoclonal antibodies (mAbs) have been used as therapeutic agents, which must overcome many developability issues after the discovery from in vitro display libraries. Especially, polyreactive mAbs can strongly bind to a specific target and weakly bind to off-target proteins, which leads to poor antibody pharmacokinetics in clinical development. Although early assessment of polyreactive mAbs is important in the early discovery stage, experimental assessments are usually time-consuming and expensive. Therefore, computational approaches for predicting the polyreactivity of single-chain fragment variables (scFvs) in the early discovery stage would be promising for reducing experimental efforts. 

Here, we made prediction models for the polyreactivity of scFvs with the known polyreactive antibody features and natural language model descriptors. We predicted 19,426 protein structures of scFvs with trRosetta to calculate the polyreactive antibody features and investigated the classifying performance of each factor for polyreactivity. In the known polyreactive features, the net charge of the CDR2 loop, the tryptophan and glycine residues in CDR-H3, and the lengths of the CDR1 and CDR2 loops, importantly contributed to the performance of the models. Additionally, the hydrodynamic features, such as partial specific volume, gyration radius, and isoelectric points of CDR loops and scFvs, were newly added to improve model performance. Finally, we made the prediction model with a robust performance () with an ensemble learning of the top 3 best models.

The prediction models for polyreactivity would help assess polyreactive scFvs in the early discovery stage and our approaches would be promising to develop machine learning models with quantitative data from high throughput assays for antibody screening. 

The online version contains supplementary material available at 10.1186/s12859-022-05010-4.

[end paper 10]

[begin paper 11]

PMCID = 11208048

Title = Accurate prediction of CDR-H3 loop structures of antibodies with deep learning

Accurate prediction of the structurally diverse complementarity determining region heavy chain 3 (CDR-H3) loop structure remains a primary and long-standing challenge for antibody modeling. Here, we present the H3-OPT toolkit for predicting the 3D structures of monoclonal antibodies and nanobodies. H3-OPT combines the strengths of AlphaFold2 with a pre-trained protein language model and provides a 2.24 Å average RMSDCα between predicted and experimentally determined CDR-H3 loops, thus outperforming other current computational methods in our non-redundant high-quality dataset. The model was validated by experimentally solving three structures of anti-VEGF nanobodies predicted by H3-OPT. We examined the potential applications of H3-OPT through analyzing antibody surface properties and antibody–antigen interactions. This structural prediction tool can be used to optimize antibody–antigen binding and engineer therapeutic antibodies with biophysical properties for specialized drug administration route.

[end paper 11]

[begin paper 12]

PMCID = 10220908

Title = Structural Modeling of Nanobodies: A Benchmark of State-of-the-Art Artificial Intelligence Programs

The number of applications for nanobodies is steadily expanding, positioning these molecules as fast-growing biologic products in the biotechnology market. Several of their applications require protein engineering, which in turn would greatly benefit from having a reliable structural model of the nanobody of interest. However, as with antibodies, the structural modeling of nanobodies is still a challenge. With the rise of artificial intelligence (AI), several methods have been developed in recent years that attempt to solve the problem of protein modeling. In this study, we have compared the performance in nanobody modeling of several state-of-the-art AI-based programs, either designed for general protein modeling, such as AlphaFold2, OmegaFold, ESMFold, and Yang-Server, or specifically designed for antibody modeling, such as IgFold, and Nanonet. While all these programs performed rather well in constructing the nanobody framework and CDRs 1 and 2, modeling CDR3 still represents a big challenge. Interestingly, tailoring an AI method for antibody modeling does not necessarily translate into better results for nanobodies.

[end paper 12]

[begin paper 13]

PMCID = 10886287

Title = Leveraging Artificial Intelligence to Expedite Antibody Design and Enhance Antibody–Antigen Interactions

This perspective sheds light on the transformative impact of recent computational advancements in the field of protein therapeutics, with a particular focus on the design and development of antibodies. Cutting-edge computational methods have revolutionized our understanding of protein–protein interactions (PPIs), enhancing the efficacy of protein therapeutics in preclinical and clinical settings. Central to these advancements is the application of machine learning and deep learning, which offers unprecedented insights into the intricate mechanisms of PPIs and facilitates precise control over protein functions. Despite these advancements, the complex structural nuances of antibodies pose ongoing challenges in their design and optimization. Our review provides a comprehensive exploration of the latest deep learning approaches, including language models and diffusion techniques, and their role in surmounting these challenges. We also present a critical analysis of these methods, offering insights to drive further progress in this rapidly evolving field. The paper includes practical recommendations for the application of these computational techniques, supplemented with independent benchmark studies. These studies focus on key performance metrics such as accuracy and the ease of program execution, providing a valuable resource for researchers engaged in antibody design and development. Through this detailed perspective, we aim to contribute to the advancement of antibody design, equipping researchers with the tools and knowledge to navigate the complexities of this field.

[end paper 13]

[begin paper 14]

PMCID = 10698008

Title = Improving antibody optimization ability of generative adversarial network through large language model

Generative adversarial networks (GANs) have successfully generated functional protein sequences. However, traditional GANs often suffer from inherent randomness, resulting in a lower probability of obtaining desirable sequences. Due to the high cost of wet-lab experiments, the main goal of computer-aided antibody optimization is to identify high-quality candidate antibodies from a large range of possibilities, yet improving the ability of GANs to generate these desired antibodies is a challenge. In this study, we propose and evaluate a new GAN called the Language Model Guided Antibody Generative Adversarial Network (AbGAN-LMG). This GAN uses a language model as an input, harnessing such models’ powerful representational capabilities to improve the GAN’s generation of high-quality antibodies. We conducted a comprehensive evaluation of the antibody libraries and sequences generated by AbGAN-LMG for COVID-19 (SARS-CoV-2) and Middle East Respiratory Syndrome (MERS-CoV). Results indicate that AbGAN-LMG has learned the fundamental characteristics of antibodies and that it improved the diversity of the generated libraries. Additionally, when generating sequences using AZD-8895 as the target antibody for optimization, over 50% of the generated sequences exhibited better developability than AZD-8895 itself. Through molecular docking, we identified 70 antibodies that demonstrated higher affinity for the wild-type receptor-binding domain (RBD) of SARS-CoV-2 compared to AZD-8895. In conclusion, AbGAN-LMG demonstrates that language models used in conjunction with GANs can enable the generation of higher-quality libraries and candidate sequences, thereby improving the efficiency of antibody optimization. AbGAN-LMG is available at http://39.102.71.224:88/.

[end paper 14]

[begin paper 15]

PMCID = 11245638

Title = ISEV2024 Abstract Book



[end paper 15]

[begin paper 16]

PMCID = 10443277

Title = Recent Progress in Antibody Epitope Prediction

Recent progress in epitope prediction has shown promising results in the development of vaccines and therapeutics against various diseases. However, the overall accuracy and success rate need to be improved greatly to gain practical application significance, especially conformational epitope prediction. In this review, we examined the general features of antibody–antigen recognition, highlighting the conformation selection mechanism in flexible antibody–antigen binding. We recently highlighted the success and warning signs of antibody epitope predictions, including linear and conformation epitope predictions. While deep learning-based models gradually outperform traditional feature-based machine learning, sequence and structure features still provide insight into antibody–antigen recognition problems.

[end paper 16]

[begin paper 17]

PMCID = 9631452

Title = Advances in antibody discovery from human BCR repertoires

Antibodies make up an important and growing class of compounds used for the diagnosis or treatment of disease. While traditional antibody discovery utilized immunization of animals to generate lead compounds, technological innovations have made it possible to search for antibodies targeting a given antigen within the repertoires of B cells in humans. Here we group these innovations into four broad categories: cell sorting allows the collection of cells enriched in specificity to one or more antigens; BCR sequencing can be performed on bulk mRNA, genomic DNA or on paired (heavy-light) mRNA; BCR repertoire analysis generally involves clustering BCRs into specificity groups or more in-depth modeling of antibody-antigen interactions, such as antibody-specific epitope predictions; validation of antibody-antigen interactions requires expression of antibodies, followed by antigen binding assays or epitope mapping. Together with innovations in Deep learning these technologies will contribute to the future discovery of diagnostic and therapeutic antibodies directly from humans.

[end paper 17]

[begin paper 18]

PMCID = 7080177

Title = Single-domain antibodies as promising experimental tools in imaging and isolation of porcine epidemic diarrhea virus

Single-domain antibody (sdAb) or nanobody possesses specific features non-accessible for conventional antibodies that make them suitable for research and biotechnological applications. Porcine epidemic diarrhea virus (PEDV) causes lethal diarrhea in piglets, resulting in great economic losses all over the world. To detect and isolate PEDV rapidly and accurately is important for the control and further research of the clinical PEDV strains. In this study, four sdAb fragments (sdAb-Mc19/29/30/37) targeting the membrane (M) protein of PEDV were selected from sdAb library that was constructed through M protein-immunized Camelus bactrianus. The selected sdAb-Mcs were solubly expressed in Escherichia coli. The functional characteristics analysis revealed that the recombinant sdAb-Mcs have excellent binding activity and specificity to M protein but have no neutralizing activity to PEDV. For further application, sdAb-Mc37 was conjugated with quantum dots to synthesize a nanoprobe for imaging PEDV in vero cells. The observed fluorescence in vero cells clearly reflects that PEDV virions can be reliably recognized and labeled by the nanoprobe. Furthermore, the sdAb-Mc29 was conjugated with superparamagnetic nanobeads to construct immunomagnetic nanobeads (IMNBs) used to isolate PEDV. One PEDV strain was successfully isolated from clinical fecal sample, suggesting IMNBs as a novel and efficient tool suitable for PEDV isolation from clinical samples. This study provided a novel application and substantiated the suitability of sdAb as a specific binder for the isolation of viruses.

The online version of this article (10.1007/s00253-018-9324-7) contains supplementary material, which is available to authorized users.

[end paper 18]

[begin paper 19]

PMCID = 10889861

Title = AbImmPred: An immunogenicity prediction method for therapeutic antibodies using AntiBERTy-based sequence features

Due to the unnecessary immune responses induced by therapeutic antibodies in clinical applications, immunogenicity is an important factor to be considered in the development of antibody therapeutics. To a certain extent, there is a lag in using wet-lab experiments to test the immunogenicity in the development process of antibody therapeutics. Developing a computational method to predict the immunogenicity at once the antibody sequence is designed, is of great significance for the screening in the early stage and reducing the risk of antibody therapeutics development. In this study, a computational immunogenicity prediction method was proposed on the basis of AntiBERTy-based features of amino sequences in the antibody variable region. The AntiBERTy-based sequence features were first calculated using the AntiBERTy pre-trained model. Principal component analysis (PCA) was then applied to reduce the extracted feature to two dimensions to obtain the final features. AutoGluon was then used to train multiple machine learning models and the best one, the weighted ensemble model, was obtained through 5-fold cross-validation on the collected data. The data contains 199 commercial therapeutic antibodies, of which 177 samples were used for model training and 5-fold cross-validation, and the remaining 22 samples were used as an independent test dataset to evaluate the performance of the constructed model and compare it with other prediction methods. Test results show that the proposed method outperforms the comparison method with 0.7273 accuracy on the independent test dataset, which is 9.09% higher than the comparison method. The corresponding web server is available through the official website of GenScript Co., Ltd., https://www.genscript.com/tools/antibody-immunogenicity.

[end paper 19]

[begin paper 20]

PMCID = 10370091

Title = Contextual protein and antibody encodings from equivariant graph transformers

The optimal residue identity at each position in a protein is determined by its structural, evolutionary, and functional context. We seek to learn the representation space of the optimal amino-acid residue in different structural contexts in proteins. Inspired by masked language modeling (MLM), our training aims to transduce learning of amino-acid labels from non-masked residues to masked residues in their structural environments and from general (e.g., a residue in a protein) to specific contexts (e.g., a residue at the interface of a protein or antibody complex). Our results on native sequence recovery and forward folding with AlphaFold2 suggest that the amino acid label for a protein residue may be determined from its structural context alone (i.e., without knowledge of the sequence labels of surrounding residues). We further find that the sequence space sampled from our masked models recapitulate the evolutionary sequence neighborhood of the wildtype sequence. Remarkably, the sequences conditioned on highly plastic structures recapitulate the conformational flexibility encoded in the structures. Furthermore, maximum-likelihood interfaces designed with masked models recapitulate wildtype binding energies for a wide range of protein interfaces and binding strengths. We also propose and compare fine-tuning strategies to train models for designing CDR loops of antibodies in the structural context of the antibody-antigen interface by leveraging structural databases for proteins, antibodies (synthetic and experimental) and protein-protein complexes. We show that pretraining on more general contexts improves native sequence recovery for antibody CDR loops, especially for the hypervariable CDR H3, while fine-tuning helps to preserve patterns observed in special contexts.

[end paper 20]

[begin paper 21]

PMCID = 11226077

Title = Protein loop structure prediction by community-based deep learning and its application to antibody CDR H3 loop modeling

As of now, more than 60 years have passed since the first determination of protein structures through crystallography, and a significant portion of protein structures can be predicted by computers. This is due to the groundbreaking enhancement in protein structure prediction achieved through neural network training utilizing extensive sequence and structure data. However, substantial challenges persist in structure prediction due to limited data availability, with antibody structure prediction standing as one such challenge. In this paper, we propose a novel neural network architecture that effectively enables structure prediction by reflecting the inherent combinatorial nature involved in protein structure formation. The core idea of this neural network architecture is not solely to track and generate a single structure but rather to form a community of multiple structures and pursue accurate structure prediction by exchanging information among community members. Applying this concept to antibody CDR H3 loop structure prediction resulted in improved structure sampling. Such an approach could be applied in the structural and functional studies of proteins, particularly in exploring various physiological processes mediated by loops. Moreover, it holds potential in addressing various other types of combinatorial structure prediction and design problems.

In this paper, we propose a new architecture that aims to improve upon protein structure prediction algorithms like AlphaFold or RoseTTAFold by considering the combinatorial nature of protein structure formation. Such an architecture, reflecting the physical principles of nature, is expected to yield beneficial results, particularly in scenarios with limited structure and sequence information. Named ComMat, this architecture does not focus on a single structure but rather on a set of multiple structures—a community—simultaneously. In this process, combinatorial exploration of protein structure is encouraged through information exchange among community members. ComMat is an instance that integrates this idea within the structure module of AlphaFold. Applying ComMat to antibody CDR H3 loop structure prediction yielded outstanding results in structure sampling and prediction when tested on the IgFold set and compared with IgFold and AlphaFold-Multimer. It confirmed that improved structure sampling stems from effective structural exploration. The proposed concept here could potentially be used in the development of various other combinatorial protein structure prediction and protein design methods.

[end paper 21]

[begin paper 22]

PMCID = 11066466

Title = Recent advances and challenges in protein complex model accuracy estimation

Estimation of model accuracy plays a crucial role in protein structure prediction, aiming to evaluate the quality of predicted protein structure models accurately and objectively. This process is not only key to screening candidate models that are close to the real structure, but also provides guidance for further optimization of protein structures. With the significant advancements made by AlphaFold2 in monomer structure, the problem of single-domain protein structure prediction has been widely solved. Correspondingly, the importance of assessing the quality of single-domain protein models decreased, and the research focus has shifted to estimation of model accuracy of protein complexes. In this review, our goal is to provide a comprehensive overview of the reference and statistical metrics, as well as representative methods, and the current challenges within four distinct facets (Topology Global Score, Interface Total Score, Interface Residue-Wise Score, and Tertiary Residue-Wise Score) in the field of complex EMA.

[end paper 22]

[begin paper 23]

PMCID = 10925456

Title = EvoAI enables extreme compression and reconstruction of the protein sequence space

Designing proteins with improved functions requires a deep understanding of how sequence and function are related, a vast space that is hard to explore. The ability to efficiently compress this space by identifying functionally important features is extremely valuable. Here, we first establish a method called EvoScan to comprehensively segment and scan the high-fitness sequence space to obtain anchor points that capture its essential features, especially in high dimensions. Our approach is compatible with any biomolecular function that can be coupled to a transcriptional output. We then develop deep learning and large language models to accurately reconstruct the space from these anchors, allowing computational prediction of novel, highly fit sequences without prior homology-derived or structural information. We apply this hybrid experimental-computational method, which we call EvoAI, to a repressor protein and find that only 82 anchors are sufficient to compress the high-fitness sequence space with a compression ratio of 1048. The extreme compressibility of the space informs both applied biomolecular design and understanding of natural evolution.

[end paper 23]

[begin paper 24]

PMCID = 8113296

Title = Extensive preclinical evaluation of lutetium-177-labeled PSMA-specific tracers for prostate cancer radionuclide therapy

Various radiolabeled prostate-specific membrane antigen (PSMA)–targeting tracers are clinically applied for prostate cancer (PCa) imaging and targeted radionuclide therapy. The PSMA binding affinities, biodistribution, and DNA-damaging capacities of these radiotracers have not yet been compared in detail. A major concern of PSMA-targeting radiotracers is the toxicity in other PSMA-expressing organs, such as the salivary glands, thus demanding careful evaluation of the most optimal and safest radiotracer. In this extensive preclinical study, we evaluated the clinically applied PSMA-targeting small molecule inhibitors DOTA-PSMA-617 (PSMA-617) and DOTAGA-PSMA-I&T (PSMA-I&T) and the PSMA nanobody DOTA-JVZ-007 (JVZ-007) using PSMA-expressing cell lines, a unique set of PCa patient-derived xenografts (PDX) and healthy human tissues.

In vitro displacement studies on PSMA-expressing cells and cryosections of a PSMA-positive PDX revealed high and specific binding affinity for all three tracers labeled with lutetium-177 with IC50 values in the nanomolar range. Interestingly, [177Lu]Lu-JVZ-007 could not be displaced by PSMA-617 or PSMA-I&T, suggesting that this tracer targets an alternative binding site. Autoradiography assays on cryosections of human salivary and renal tissues revealed [177Lu]Lu-PSMA-617 to have the lowest binding to these healthy organs compared with [177Lu]Lu-PSMA-I&T. In vivo biodistribution assays confirmed the in vitro results with comparable tumor uptake of [177Lu]Lu-PSMA-617 and [177Lu]Lu-PSMA-I&T at all timepoints, resulting in induction of similar levels of DNA double-strand breaks in the tumors. However, [177Lu]Lu-PSMA-I&T demonstrated approximately 40× higher renal uptake at 4 and 8 h post injection resulting in an unfavorable tumor-to-kidney ratio.

[177Lu]Lu-PSMA-617 has the most favorable biodistribution in mice as well as more favorable binding characteristics in vitro in PSMA-positive cells and human kidney and salivary gland specimens compared with [177Lu]Lu-PSMA-I&T and [177Lu]Lu-JVZ-007. Based on our preclinical evaluation, [177Lu]Lu-PSMA-617 is the best performing tracer to be taken further into clinical evaluation for PSMA-targeted radiotherapeutic development although with careful evaluation of the tracer binding to PSMA-expressing organs.

The online version of this article (10.1007/s00259-020-05057-6) contains supplementary material, which is available to authorized users.

[end paper 24]

[begin paper 25]

PMCID = 10792694

Title = Systematic investigation of machine learning on limited data: A study on predicting protein-protein binding strength

The application of machine learning techniques in biological research, especially when dealing with limited data availability, poses significant challenges. In this study, we leveraged advancements in method development for predicting protein-protein binding strength to conduct a systematic investigation into the application of machine learning on limited data. The binding strength, quantitatively measured as binding affinity, is vital for understanding the processes of recognition, association, and dysfunction that occur within protein complexes. By incorporating transfer learning, integrating domain knowledge, and employing both deep learning and traditional machine learning algorithms, we mitigated the impact of data limitations and made significant advancements in predicting protein-protein binding affinity. In particular, we developed over 20 models, ultimately selecting three representative best-performing ones that belong to distinct categories. The first model is structure-based, consisting of a random forest regression and thirteen handcrafted features. The second model is sequence-based, employing an architecture that combines transferred embedding features with a multilayer perceptron. Finally, we created an ensemble model by averaging the predictions of the two aforementioned models. The comparison with other predictors on three independent datasets confirms the significant improvements achieved by our models in predicting protein-protein binding affinity. The programs for running these three models are available at https://github.com/minghuilab/BindPPI.

[end paper 25]

[begin paper 26]

PMCID = 7153572

Title = Development of a One-step Immunoassay for Triazophos Using Camel Single Domain Antibody-Alkaline Phosphatase Fusion Protein

Triazophos is mainly used in Asian and African countries for the control of insects in agricultural production. Camelid variable domains of heavy chain antibodies (VHHs) show great promise in monitoring environmental chemicals such as pesticides. To improve the rate of success in the generation of VHHs against triazophos, genes specifically encoding VHH fragments from the unique allotype IgG3a of an immunized Bactrianus were amplified by using a pair of novel primers and introduced to construct a diverse VHH library. Five out of seven isolated positive clones, including the VHH T1 with the highest affinity to triazophos, were derived from the allotype IgG3a. A one-step enzyme linked immunosorbent assay (ELISA) using VHH T1 genetically fused with alkaline phosphatase (AP) had a half-maximum inhibition concentration of 6.6 ng/mL for triazophos. This assay showed negligible cross-reactivity with a list of important organophosphate pesticides (<0.1%). The average recoveries of triazophos from water, soil and apple samples determined by the one-step ELISA ranged from 83% to 108%, having a good correlation with those by a gas chromatography-mass spectrometry (R2 = 0.99). The VHH-AP fusion protein shows potential for the analysis of triazophos in various matrices.

[end paper 26]

[begin paper 27]

PMCID = 11788754

Title = An outlook on structural biology after AlphaFold: tools, limits and perspectives

AlphaFold and similar groundbreaking, AI‐based tools, have revolutionized the field of structural bioinformatics, with their remarkable accuracy in ab‐initio protein structure prediction. This success has catalyzed the development of new software and pipelines aimed at incorporating AlphaFold's predictions, often focusing on addressing the algorithm's remaining challenges. Here, we present the current landscape of structural bioinformatics shaped by AlphaFold, and discuss how the field is dynamically responding to this revolution, with new software, methods, and pipelines. While the excitement around AI‐based tools led to their widespread application, it is essential to acknowledge that their practical success hinges on their integration into established protocols within structural bioinformatics, often neglected in the context of AI‐driven advancements. Indeed, user‐driven intervention is still as pivotal in the structure prediction process as in complementing state‐of‐the‐art algorithms with functional and biological knowledge.

Rendering of a GFP‐like fluorescent protein, starting from its AF‐generated PDB coordinates. Like AI algorithms for image generation focus on creating realistic and detailed content, AI‐based algorithms for protein structure generation aim to accurately generate/predict complex molecular arrangements. Figure generated using CGDream (https://cgdream.ai). 

[end paper 27]

[begin paper 28]

PMCID = 10589401

Title = Generative models for protein sequence modeling: recent advances and future directions

The widespread adoption of high-throughput omics technologies has exponentially increased the amount of protein sequence data involved in many salient disease pathways and their respective therapeutics and diagnostics. Despite the availability of large-scale sequence data, the lack of experimental fitness annotations underpins the need for self-supervised and unsupervised machine learning (ML) methods. These techniques leverage the meaningful features encoded in abundant unlabeled sequences to accomplish complex protein engineering tasks. Proficiency in the rapidly evolving fields of protein engineering and generative AI is required to realize the full potential of ML models as a tool for protein fitness landscape navigation. Here, we support this work by (i) providing an overview of the architecture and mathematical details of the most successful ML models applicable to sequence data (e.g. variational autoencoders, autoregressive models, generative adversarial neural networks, and diffusion models), (ii) guiding how to effectively implement these models on protein sequence data to predict fitness or generate high-fitness sequences and (iii) highlighting several successful studies that implement these techniques in protein engineering (from paratope regions and subcellular localization prediction to high-fitness sequences and protein design rules generation). By providing a comprehensive survey of model details, novel architecture developments, comparisons of model applications, and current challenges, this study intends to provide structured guidance and robust framework for delivering a prospective outlook in the ML-driven protein engineering field.

[end paper 28]

[begin paper 29]

PMCID = 8578158

Title = Preclinical Evaluation of 89Zr-Df-IAB22M2C PET as an Imaging Biomarker for the Development of the GUCY2C-CD3 Bispecific PF-07062119 as a T Cell Engaging Therapy

A sensitive and specific imaging biomarker to monitor immune activation and quantify pharmacodynamic responses would be useful for development of immunomodulating anti-cancer agents. PF-07062119 is a T cell engaging bispecific antibody that binds to CD3 and guanylyl cyclase C, a protein that is over-expressed by colorectal cancers. Here, we used 89Zr-Df-IAB22M2C (89Zr-Df-Crefmirlimab), a human CD8-specific minibody to monitor CD8+ T cell infiltration into tumors by positron emission tomography. We investigated the ability of 89Zr-Df-IAB22M2C to track anti-tumor activity induced by PF-07062119 in a human CRC adoptive transfer mouse model (with injected activated/expanded human T cells), as well as the correlation of tumor radiotracer uptake with CD8+ immunohistochemical staining.

NOD SCID gamma mice bearing human CRC LS1034 tumors were treated with four different doses of PF-07062119, or a non-targeted CD3 BsAb control, and imaged with 89Zr-Df-IAB22M2C PET at days 4 and 9. Following PET/CT imaging, mice were euthanized and dissected for ex vivo distribution analysis of 89Zr-Df-IAB22M2C in tissues on days 4 and 9, with additional data collected on day 6 (supplementary). Data were analyzed and reported as standard uptake value and %ID/g for in vivo imaging and ex vivo tissue distribution. In addition, tumor tissues were evaluated by immunohistochemistry for CD8+ T cells.

The results demonstrated substantial mean uptake of 89Zr-Df-IAB22M2C (%ID/g) in PF-07062119-treated tumors, with significant increases in comparison to non-targeted BsAb-treated controls, as well as PF-07062119 dose-dependent responses over time of treatment. A moderate correlation was observed between tumor tissue radioactivity uptake and CD8+ cell density, demonstrating the value of the imaging agent for non-invasive assessment of intra-tumoral CD8+ T cells and the mechanism of action for PF-07062119.

Immune-imaging technologies for quantitative cellular measures would be a valuable biomarker in immunotherapeutic clinical development. We demonstrated a qualification of 89Zr-IAB22M2C PET to evaluate PD responses (mice) to a novel immunotherapeutic.

The online version contains supplementary material available at 10.1007/s11307-021-01621-0.

[end paper 29]

[begin paper 30]

PMCID = 11601283

Title = Programmable protein degraders enable selective knockdown of pathogenic β-catenin subpopulations in vitro and in vivo

Aberrant activation of Wnt signaling results in unregulated accumulation of cytosolic β-catenin, which subsequently enters the nucleus and promotes transcription of genes that contribute to cellular proliferation and malignancy. Here, we sought to eliminate pathogenic β-catenin from the cytosol using designer ubiquibodies (uAbs), chimeric proteins composed of an E3 ubiquitin ligase and a target-binding domain that redirect intracellular proteins to the proteasome for degradation. To accelerate uAb development, we leveraged a protein language model (pLM)-driven algorithm called SaLT&PepPr to computationally design “guide” peptides with affinity for β-catenin, which were subsequently fused to the catalytic domain of a human E3 called C-terminus of Hsp70-interacting protein (CHIP). Expression of the resulting peptide-guided uAbs in colorectal cancer cells led to the identification of several designs that significantly reduced the abnormally stable pool of free β-catenin in the cytosol and nucleus while preserving the normal membrane-associated subpopulation. This selective knockdown of pathogenic β-catenin suppressed Wnt/β-catenin signaling and impaired tumor cell survival and proliferation. Furthermore, one of the best degraders selectively decreased cytosolic but not membrane-associated β-catenin levels in livers of BALB/c mice following delivery as a lipid nanoparticle (LNP)-encapsulated mRNA. Collectively, these findings reveal the unique ability of uAbs to selectively eradicate abnormal proteins in vitro and in vivo and open the door to peptide-programmable biologic modulators of other disease-causing proteins.

[end paper 30]



