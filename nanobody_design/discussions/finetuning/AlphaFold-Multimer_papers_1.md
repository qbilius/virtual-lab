## User

This is the beginning of an individual meeting with Computational Biologist to discuss your research project.

Here is the agenda for the meeting:

You are working on a research project to use machine learning to develop antibodies or nanobodies for the newest variant of the SARS-CoV-2 spike protein that also, ideally, have activity against other circulating minor variants and past variants. Your team previous decided to modify existing nanobodies to improve their binding to the newest variant of the SARS-CoV-2 spike protein. You are responsible for understanding the topic "AlphaFold-Multimer" in the context of designing nanobody binders for SARS-CoV-2. You need to fine-tune yourself on the relevant literature on AlphaFold-Multimer to improve your ability to design SARS-CoV-2 nanobody binders. Please use PubMed Central and search for relevant papers on AlphaFold-Multimer using the query "AlphaFold-Multimer protein structure prediction" and request 100 articles with abstracts only. Read all of the abstracts and based on each abstract individually, decide whether you want to fine-tune yourself on the full text of that paper. Include as many papers as possible, but only include papers that are directly relevant to AlphaFold-Multimer. Please provide the PMCIDs and titles of all the papers that you wish to fine-tune yourself on as a Python dictionary mapping PMCID as a double-quoted string to title as a double-quoted string.

Here are the agenda questions that must be answered:

1. What are the PMCIDs and titles of the papers you wish to fine-tune yourself on (as a Python dictionary mapping PMCID as a double-quoted string to title as double-quoted string)?

Computational Biologist, please provide your response to the agenda.

## Computational Biologist

Based on the abstracts provided, I have identified the following papers as directly relevant to AlphaFold-Multimer and suitable for further fine-tuning. Here is the Python dictionary mapping PMCID to title:

```python
{
    "10638423": "Enhancing alphafold-multimer-based protein complex structure prediction with MULTICOM in CASP15",
    "10987090": "Protein language model-embedded geometric graphs power inter-protein contact prediction",
    "10090944": "Protein complexes in cells by AI‐assisted structural proteomics",
    "9580857": "Improving peptide-protein docking with AlphaFold-Multimer using forced sampling",
    "9278006": "Benchmarking AlphaFold for protein complex modeling reveals accuracy determinants",
    "10883643": "ABAG-docking benchmark: a non-redundant structure benchmark dataset for antibody–antigen computational docking",
    "11383924": "Modelling protein complexes with crosslinking mass spectrometry and deep learning",
    "10348836": "Evaluation of AlphaFold-Multimer prediction on multi-chain protein complexes",
    "11392808": "Docking a flexible basket onto the core of the nuclear pore complex",
    "11052572": "Deep learning insights into the architecture of the mammalian egg-sperm fusion synapse",
    "11302914": "Improved protein complex prediction with AlphaFold-multimer by denoising the MSA profile",
    "10613300": "Topological links in predicted protein complex structures reveal limitations of AlphaFold",
    "10987651": "AI-guided pipeline for protein–protein interaction drug discovery identifies a SARS-CoV-2 inhibitor",
    "10966450": "SpatialPPI: Three-dimensional space protein-protein interaction prediction with AlphaFold Multimer",
    "10844748": "Methylation of elongation factor 1A by yeast Efm4 or human eEF1A-KMT2 involves a beta-hairpin recognition motif and crosstalks with phosphorylation",
    "10312674": "AI-guided pipeline for protein-protein interaction drug discovery identifies a SARS-CoV-2 inhibitor",
    "11230293": "A Comparison of Antibody-Antigen Complex Sequence-to-Structure Prediction Methods and their Systematic Biases",
    "11148199": "Identification of type VI secretion system effector-immunity pairs using structural bioinformatics",
    "10933411": "Enhancing coevolutionary signals in protein–protein interaction prediction through clade-wise alignment integration",
    "10129313": "Fast, accurate antibody structure prediction from deep learning on massive set of natural antibodies",
    "11405125": "AI-based IsAb2.0 for antibody design",
    "10497418": "DNA-PK and the TRF2 iDDR inhibit MRN-initiated resection at leading-end telomeres",
    "9859041": "Structure-based prediction of T cell receptor:peptide-MHC interactions",
    "10883280": "Systematic discovery of protein interaction interfaces using AlphaFold and experimental validation",
    "11168924": "Accurate structure prediction of biomolecular interactions with AlphaFold 3",
    "10227038": "ImmuneBuilder: Deep-Learning models for predicting the structures of immune proteins",
    "11162299": "Integration of protein sequence and protein–protein interaction data by hypergraph learning to identify novel protein complexes",
    "11702601": "A bacterial NLR-related protein recognizes multiple unrelated phage triggers to sense infection",
    "11065178": "NFATC2IP is a mediator of SUMO-dependent genome integrity",
    "9957715": "Structural modelling of human complement FHR1 and two of its synthetic derivatives provides insight into their in-vivo functions",
    "10011802": "AlphaFold2 and its applications in the fields of biology and medicine",
    "11009965": "Computational Approaches to Predict Protein–Protein Interactions in Crowded Cellular Environments",
    "10151615": "Cross-linking mass spectrometry discovers, evaluates, and corroborates structures and protein–protein interactions in the human cell",
    "10893003": "Recent Progress of Protein Tertiary Structure Prediction",
    "10484931": "Improving AlphaFold2-based protein tertiary structure prediction with MULTICOM in CASP15",
    "11799971": "Physical-aware model accuracy estimation for protein complex using deep learning method",
    "10113261": "PeSTo: parameter-free geometric deep learning for accurate prediction of protein binding interfaces",
    "11623436": "Challenges in bridging the gap between protein structure prediction and functional interpretation",
    "10692239": "Multi-domain and complex protein structure prediction using inter-domain interactions from deep learning",
    "8874714": "Prediction of Monomeric and Dimeric Structures of CYP102A1 Using AlphaFold2 and AlphaFold Multimer and Assessment of Point Mutation Effect on the Efficiency of Intra- and Interprotein Electron Transfer",
    "9758053": "De novo design of obligate ABC-type heterotrimeric proteins",
    "10533508": "AlphaFold-Multimer predicts cross-kingdom interactions at the plant-pathogen interface",
    "9570074": "AI-Based Protein Interaction Screening and Identification (AISID)",
    "10320053": "PAE viewer: a webserver for the interactive visualization of the predicted aligned error for multimer structure predictions and crosslinks",
    "10312770": "Structural modeling of cytokine-receptor-JAK2 signaling complexes using AlphaFold Multimer",
    "10534052": "AFsample: improving multimer prediction with AlphaFold using massive sampling",
    "11405395": "A common mechanism for recruiting the Rrm3 and RTEL1 accessory helicases to the eukaryotic replisome",
    "11319189": "AlphaFold2 in biomedical research: facilitating the development of diagnostic strategies for disease",
    "11575592": "ProBID-Net: a deep learning model for protein–protein binding interface design†",
    "11461844": "Unmasking AlphaFold to integrate experiments and predictions in multimeric complexes",
    "11763140": "Structure-Based Approaches for Protein–Protein Interaction Prediction Using Machine Learning and Deep Learning",
    "9772387": "Protein complex prediction using Rosetta, AlphaFold, and mass spectrometry covalent labeling",
    "9234698": "AlphaFold Models of Small Proteins Rival the Accuracy of Solution NMR Structures",
    "9973325": "Identification and Structural Modeling of the RNA Polymerase Omega Subunits in Chlamydiae and Other Obligate Intracellular Bacteria",
    "11070728": "HighFold: accurately predicting structures of cyclic peptides and complexes with head-to-tail and disulfide bridge constraints",
    "11564622": "Diversification of molecular pattern recognition in bacterial NLR-like proteins",
    "10165273": "Itch receptor MRGPRX4 interacts with the receptor activity–modifying proteins",
    "11218997": "Human cytomegalovirus deploys molecular mimicry to recruit VPS4A to sites of virus assembly",
    "9459051": "RL-MLZerD: Multimeric protein docking using reinforcement learning",
    "11513016": "Accelerating protein–protein interaction screens with reduced AlphaFold-Multimer sampling",
    "9094657": "A bioactive phlebovirus-like envelope protein in a hookworm endogenous virus",
    "10808870": "In silico protein function prediction: the rise of machine learning-based approaches",
    "11702703": "A deep learning method for predicting interactions for intrinsically disordered regions of proteins",
    "10281963": "PIQLE: protein–protein interface quality estimation by deep graph learning of multimeric interaction geometries",
    "11016909": "Evolutionary analysis reveals the role of a non-catalytic domain of peptidyl arginine deiminase 2 in transcriptional regulation",
    "11775587": "The ortholog of human DNAJC9 promotes histone H3–H4 degradation and is counteracted by Asf1 in fission yeast",
    "9907820": "AlphaFold-multimer predicts ATG8 protein binding motifs crucial for autophagy research",
    "10760632": "Characterization of the SF3B1–SUGP1 interface reveals how numerous cancer mutations cause mRNA missplicing",
    "10540290": "Disrupting Dimeric β-Amyloid by Electric Fields",
    "11030396": "Predictomes: A classifier-curated database of AlphaFold-modeled protein-protein interactions",
    "11281715": "African Swine Fever Virus Protein–Protein Interaction Prediction",
    "10707872": "Intelligent Protein Design and Molecular Characterization Techniques: A Comprehensive Review",
    "11754096": "In situ analysis reveals the TRiC duty cycle and PDCD5 as an open-state cofactor",
    "10554751": "Allosteric inhibition of the T cell receptor by a designed membrane ligand",
    "11228504": "Pairing interacting protein sequences using masked language modeling",
    "10497406": "PCNA recruits cohesin loader Scc2 to ensure sister chromatid cohesion",
    "11188072": "The Cryptic Bacterial Microproteome",
    "9674347": "Integrative modeling reveals the molecular architecture of the intraflagellar transport A (IFT-A) complex",
    "11697402": "State of the interactomes: an evaluation of molecular networks for generating biological insights",
    "9992458": "Domain-Based Protein Docking with Extremely Large Conformational Changes",
    "11294035": "Artificial intelligence in fusion protein three‐dimensional structure prediction: Review and perspective",
    "9850923": "Design of Diverse Asymmetric Pockets in De Novo Homo-oligomeric Proteins",
    "10136805": "Using AlphaFold Predictions in Viral Research",
    "11117562": "A Survey of Deep Learning Methods for Estimating the Accuracy of Protein Quaternary Structure Models",
    "9846249": "Mapping the FtsQBL divisome components in bacterial NTD pathogens as potential drug targets",
    "10274727": "Structural Organization of the Retriever-CCC Endosomal Recycling Complex",
    "10349958": "Evaluation of AlphaFold Antibody-Antigen Modeling with Implications for Improving Predictive Accuracy",
    "11507785": "Exploring voltage-gated sodium channel conformations and protein-protein interactions using AlphaFold2",
    "10246253": "Establishing comprehensive quaternary structural proteomes from genome sequence",
    "11112044": "Peptriever: a Bi-Encoder approach for large-scale protein–peptide binding search",
    "10356834": "GPCRome-wide analysis of G-protein-coupling diversity using a computational biology approach",
    "11321591": "Arl2 GTPase associates with the centrosomal protein Cdk5rap2 to regulate cortical development via microtubule organization",
    "10285254": "The FAM86 domain of FAM86A confers substrate specificity to promote EEF2-Lys525 methylation",
    "11116810": "ERLIN1/2 scaffolds bridge TMUB1 and RNF170 and restrict cholesterol esterification to regulate the secretory pathway"
}
```

These papers are directly relevant to AlphaFold-Multimer and will provide valuable insights for your research project.

## User

Tool Output:

Here are the top 100 articles on PubMed Central for the query "AlphaFold-Multimer protein structure prediction":

[begin paper 1]

PMCID = 10638423

Title = Enhancing alphafold-multimer-based protein complex structure prediction with MULTICOM in CASP15

To enhance the AlphaFold-Multimer-based protein complex structure prediction, we developed a quaternary structure prediction system (MULTICOM) to improve the input fed to AlphaFold-Multimer and evaluate and refine its outputs. MULTICOM samples diverse multiple sequence alignments (MSAs) and templates for AlphaFold-Multimer to generate structural predictions by using both traditional sequence alignments and Foldseek-based structure alignments, ranks structural predictions through multiple complementary metrics, and refines the structural predictions via a Foldseek structure alignment-based refinement method. The MULTICOM system with different implementations was blindly tested in the assembly structure prediction in the 15th Critical Assessment of Techniques for Protein Structure Prediction (CASP15) in 2022 as both server and human predictors. MULTICOM_qa ranked 3rd among 26 CASP15 server predictors and MULTICOM_human ranked 7th among 87 CASP15 server and human predictors. The average TM-score of the first predictions submitted by MULTICOM_qa for CASP15 assembly targets is ~0.76, 5.3% higher than ~0.72 of the standard AlphaFold-Multimer. The average TM-score of the best of top 5 predictions submitted by MULTICOM_qa is ~0.80, about 8% higher than ~0.74 of the standard AlphaFold-Multimer. Moreover, the Foldseek Structure Alignment-based Multimer structure Generation (FSAMG) method outperforms the widely used sequence alignment-based multimer structure generation.

A protein complex structure prediction system (MULTICOM3) improved AlphaFold-Multimer-based prediction by 5–10% in the 15th Critical Assessment of Techniques for Protein Structure Prediction (CASP15).

[end paper 1]

[begin paper 2]

PMCID = 10987090

Title = Protein language model-embedded geometric graphs power inter-protein contact prediction

Accurate prediction of contacting residue pairs between interacting proteins is very useful for structural characterization of protein–protein interactions. Although significant improvement has been made in inter-protein contact prediction recently, there is still a large room for improving the prediction accuracy. Here we present a new deep learning method referred to as PLMGraph-Inter for inter-protein contact prediction. Specifically, we employ rotationally and translationally invariant geometric graphs obtained from structures of interacting proteins to integrate multiple protein language models, which are successively transformed by graph encoders formed by geometric vector perceptrons and residual networks formed by dimensional hybrid residual blocks to predict inter-protein contacts. Extensive evaluation on multiple test sets illustrates that PLMGraph-Inter outperforms five top inter-protein contact prediction methods, including DeepHomo, GLINTER, CDPred, DeepHomo2, and DRN-1D2D_Inter, by large margins. In addition, we also show that the prediction of PLMGraph-Inter can complement the result of AlphaFold-Multimer. Finally, we show leveraging the contacts predicted by PLMGraph-Inter as constraints for protein–protein docking can dramatically improve its performance for protein complex structure prediction.

[end paper 2]

[begin paper 3]

PMCID = 10245707

Title = Enhancing AlphaFold-Multimer-based Protein Complex Structure Prediction with MULTICOM in CASP15

AlphaFold-Multimer has emerged as the state-of-the-art tool for predicting the quaternary structure of protein complexes (assemblies or multimers) since its release in 2021. To further enhance the AlphaFold-Multimer-based complex structure prediction, we developed a new quaternary structure prediction system (MULTICOM) to improve the input fed to AlphaFold-Multimer and evaluate and refine the outputs generated by AlphaFold2-Multimer. Specifically, MULTICOM samples diverse multiple sequence alignments (MSAs) and templates for AlphaFold-Multimer to generate structural models by using both traditional sequence alignments and new Foldseek-based structure alignments, ranks structural models through multiple complementary metrics, and refines the structural models via a Foldseek structure alignment-based refinement method. The MULTICOM system with different implementations was blindly tested in the assembly structure prediction in the 15th Critical Assessment of Techniques for Protein Structure Prediction (CASP15) in 2022 as both server and human predictors. Our server (MULTICOM_qa) ranked 3rd among 26 CASP15 server predictors and our human predictor (MULTICOM_human) ranked 7th among 87 CASP15 server and human predictors. The average TM-score of the first models predicted by MULTICOM_qa for CASP15 assembly targets is ~0.76, 5.3% higher than ~0.72 of the standard AlphaFold-Multimer. The average TM-score of the best of top 5 models predicted by MULTICOM_qa is ~0.80, about 8% higher than ~0.74 of the standard AlphaFold-Multimer. Moreover, the novel Foldseek Structure Alignment-based Model Generation (FSAMG) method based on AlphaFold-Multimer outperforms the widely used sequence alignment-based model generation. The source code of MULTICOM is available at: https://github.com/BioinfoMachineLearning/MULTICOM3.

[end paper 3]

[begin paper 4]

PMCID = 10090944

Title = Protein complexes in cells by AI‐assisted structural proteomics

Accurately modeling the structures of proteins and their complexes using artificial intelligence is revolutionizing molecular biology. Experimental data enable a candidate‐based approach to systematically model novel protein assemblies. Here, we use a combination of in‐cell crosslinking mass spectrometry and co‐fractionation mass spectrometry (CoFrac‐MS) to identify protein–protein interactions in the model Gram‐positive bacterium Bacillus subtilis. We show that crosslinking interactions prior to cell lysis reveals protein interactions that are often lost upon cell lysis. We predict the structures of these protein interactions and others in the SubtiWiki database with AlphaFold‐Multimer and, after controlling for the false‐positive rate of the predictions, we propose novel structural models of 153 dimeric and 14 trimeric protein assemblies. Crosslinking MS data independently validates the AlphaFold predictions and scoring. We report and validate novel interactors of central cellular machineries that include the ribosome, RNA polymerase, and pyruvate dehydrogenase, assigning function to several uncharacterized proteins. Our approach uncovers protein–protein interactions inside intact cells, provides structural insight into their interaction interfaces, and is applicable to genetically intractable organisms, including pathogenic bacteria.

An integrative approach using crosslinking mass spectrometry (MS), co‐fractionation MS, and AlphaFold‐Multimer discovers novel protein complexes and their topologies in the model Gram‐positive bacterium Bacillus subtillis. 

[end paper 4]

[begin paper 5]

PMCID = 9580857

Title = Improving peptide-protein docking with AlphaFold-Multimer using forced sampling

Protein interactions are key in vital biological processes. In many cases, particularly in regulation, this interaction is between a protein and a shorter peptide fragment. Such peptides are often part of larger disordered regions in other proteins. The flexible nature of peptides enables the rapid yet specific regulation of important functions in cells, such as their life cycle. Consequently, knowledge of the molecular details of peptide-protein interactions is crucial for understanding and altering their function, and many specialized computational methods have been developed to study them. The recent release of AlphaFold and AlphaFold-Multimer has led to a leap in accuracy for the computational modeling of proteins. In this study, the ability of AlphaFold to predict which peptides and proteins interact, as well as its accuracy in modeling the resulting interaction complexes, are benchmarked against established methods. We find that AlphaFold-Multimer predicts the structure of peptide-protein complexes with acceptable or better quality (DockQ ≥0.23) for 66 of the 112 complexes investigated—25 of which were high quality (DockQ ≥0.8). This is a massive improvement on previous methods with 23 or 47 acceptable models and only four or eight high quality models, when using energy-based docking or interaction templates, respectively. In addition, AlphaFold-Multimer can be used to predict whether a peptide and a protein will interact. At 1% false positives, AlphaFold-Multimer found 26% of the possible interactions with a precision of 85%, the best among the methods benchmarked. However, the most interesting result is the possibility of improving AlphaFold by randomly perturbing the neural network weights to force the network to sample more of the conformational space. This increases the number of acceptable models from 66 to 75 and improves the median DockQ from 0.47 to 0.55 (17%) for first ranked models. The best possible DockQ improves from 0.58 to 0.72 (24%), indicating that selecting the best possible model is still a challenge. This scheme of generating more structures with AlphaFold should be generally useful for many applications involving multiple states, flexible regions, and disorder.

[end paper 5]

[begin paper 6]

PMCID = 9278006

Title = Benchmarking AlphaFold for protein complex modeling reveals accuracy determinants

High‐resolution experimental structural determination of protein–protein interactions has led to valuable mechanistic insights, yet due to the massive number of interactions and experimental limitations there is a need for computational methods that can accurately model their structures. Here we explore the use of the recently developed deep learning method, AlphaFold, to predict structures of protein complexes from sequence. With a benchmark of 152 diverse heterodimeric protein complexes, multiple implementations and parameters of AlphaFold were tested for accuracy. Remarkably, many cases (43%) had near‐native models (medium or high critical assessment of predicted interactions accuracy) generated as top‐ranked predictions by AlphaFold, greatly surpassing the performance of unbound protein–protein docking (9% success rate for near‐native top‐ranked models), however AlphaFold modeling of antibody–antigen complexes within our set was unsuccessful. We identified sequence and structural features associated with lack of AlphaFold success, and we also investigated the impact of multiple sequence alignment input. Benchmarking of a multimer‐optimized version of AlphaFold (AlphaFold‐Multimer) with a set of recently released antibody–antigen structures confirmed a low rate of success for antibody–antigen complexes (11% success), and we found that T cell receptor–antigen complexes are likewise not accurately modeled by that algorithm, showing that adaptive immune recognition poses a challenge for the current AlphaFold algorithm and model. Overall, our study demonstrates that end‐to‐end deep learning can accurately model many transient protein complexes, and highlights areas of improvement for future developments to reliably model any protein–protein interaction of interest.

[end paper 6]

[begin paper 7]

PMCID = 10883643

Title = ABAG-docking benchmark: a non-redundant structure benchmark dataset for antibody–antigen computational docking

Accurate prediction of antibody–antigen complex structures is pivotal in drug discovery, vaccine design and disease treatment and can facilitate the development of more effective therapies and diagnostics. In this work, we first review the antibody–antigen docking (ABAG-docking) datasets. Then, we present the creation and characterization of a comprehensive benchmark dataset of antibody–antigen complexes. We categorize the dataset based on docking difficulty, interface properties and structural characteristics, to provide a diverse set of cases for rigorous evaluation. Compared with Docking Benchmark 5.5, we have added 112 cases, including 14 single-domain antibody (sdAb) cases and 98 monoclonal antibody (mAb) cases, and also increased the proportion of Difficult cases. Our dataset contains diverse cases, including human/humanized antibodies, sdAbs, rodent antibodies and other types, opening the door to better algorithm development. Furthermore, we provide details on the process of building the benchmark dataset and introduce a pipeline for periodic updates to keep it up to date. We also utilize multiple complex prediction methods including ZDOCK, ClusPro, HDOCK and AlphaFold-Multimer for testing and analyzing this dataset. This benchmark serves as a valuable resource for evaluating and advancing docking computational methods in the analysis of antibody–antigen interaction, enabling researchers to develop more accurate and effective tools for predicting and designing antibody–antigen complexes. The non-redundant ABAG-docking structure benchmark dataset is available at https://github.com/Zhaonan99/Antibody-antigen-complex-structure-benchmark-dataset.

[end paper 7]

[begin paper 8]

PMCID = 11383924

Title = Modelling protein complexes with crosslinking mass spectrometry and deep learning

Scarcity of structural and evolutionary information on protein complexes poses a challenge to deep learning-based structure modelling. We integrate experimental distance restraints obtained by crosslinking mass spectrometry (MS) into AlphaFold-Multimer, by extending AlphaLink to protein complexes. Integrating crosslinking MS data substantially improves modelling performance on challenging targets, by helping to identify interfaces, focusing sampling, and improving model selection. This extends to single crosslinks from whole-cell crosslinking MS, opening the possibility of whole-cell structural investigations driven by experimental data. We demonstrate this by revealing the molecular basis of iron homoeostasis in Bacillus subtilis.

Elucidating the structure of protein complexes is key to understanding life at the molecular level. Here, the authors improve modelling performance on challenging targets by integrating experimental distance restraints from crosslinking mass spectrometry into AlphaFold-Multimer.

[end paper 8]

[begin paper 9]

PMCID = 10348836

Title = Evaluation of AlphaFold-Multimer prediction on multi-chain protein complexes

Despite near-experimental accuracy on single-chain predictions, there is still scope for improvement among multimeric predictions. Methods like AlphaFold-Multimer and FoldDock can accurately model dimers. However, how well these methods fare on larger complexes is still unclear. Further, evaluation methods of the quality of multimeric complexes are not well established.

We analysed the performance of AlphaFold-Multimer on a homology-reduced dataset of homo- and heteromeric protein complexes. We highlight the differences between the pairwise and multi-interface evaluation of chains within a multimer. We describe why certain complexes perform well on one metric (e.g. TM-score) but poorly on another (e.g. DockQ). We propose a new score, Predicted DockQ version 2 (pDockQ2), to estimate the quality of each interface in a multimer. Finally, we modelled protein complexes (from CORUM) and identified two highly confident structures that do not have sequence homology to any existing structures.

All scripts, models, and data used to perform the analysis in this study are freely available at https://gitlab.com/ElofssonLab/afm-benchmark.

[end paper 9]

[begin paper 10]

PMCID = 11392808

Title = Docking a flexible basket onto the core of the nuclear pore complex

The nuclear basket attaches to the nucleoplasmic side of the nuclear pore complex (NPC), coupling transcription to mRNA quality control and export. The basket expands the functional repertoire of a subset of NPCs in Saccharomyces cerevisiae by drawing a unique RNA/protein interactome. Yet, how the basket docks onto the NPC core remains unknown. By integrating AlphaFold-based interaction screens, electron microscopy and membrane-templated reconstitution, we uncovered a membrane-anchored tripartite junction between basket and NPC core. The basket subunit Nup60 harbours three adjacent short linear motifs, which connect Mlp1, a parallel homodimer consisting of coiled-coil segments interrupted by flexible hinges, and the Nup85 subunit of the Y-complex. We reconstituted the Y-complex•Nup60•Mlp1 assembly on a synthetic membrane and validated the protein interfaces in vivo. Here we explain how a short linear motif-based protein junction can substantially reshape NPC structure and function, advancing our understanding of compositional and conformational NPC heterogeneity.

Stankunas and Köhler define how the nucleoplasmic portion of the nuclear pore complex (NPC), the basket, docks onto the NPC core by integrating AlphaFold-based interaction screens, electron microscopy, and membrane-templated reconstitutions.

[end paper 10]

[begin paper 11]

PMCID = 11052572

Title = Deep learning insights into the architecture of the mammalian egg-sperm fusion synapse

A crucial event in sexual reproduction is when haploid sperm and egg fuse to form a new diploid organism at fertilization. In mammals, direct interaction between egg JUNO and sperm IZUMO1 mediates gamete membrane adhesion, yet their role in fusion remains enigmatic. We used AlphaFold to predict the structure of other extracellular proteins essential for fertilization to determine if they could form a complex that may mediate fusion. We first identified TMEM81, whose gene is expressed by mouse and human spermatids, as a protein having structural homologies with both IZUMO1 and another sperm molecule essential for gamete fusion, SPACA6. Using a set of proteins known to be important for fertilization and TMEM81, we then systematically searched for predicted binary interactions using an unguided approach and identified a pentameric complex involving sperm IZUMO1, SPACA6, TMEM81 and egg JUNO, CD9. This complex is structurally consistent with both the expected topology on opposing gamete membranes and the location of predicted N-glycans not modeled by AlphaFold-Multimer, suggesting that its components could organize into a synapse-like assembly at the point of fusion. Finally, the structural modeling approach described here could be more generally useful to gain insights into transient protein complexes difficult to detect experimentally.

[end paper 11]

[begin paper 12]

PMCID = 11302914

Title = Improved protein complex prediction with AlphaFold-multimer by denoising the MSA profile

Structure prediction of protein complexes has improved significantly with AlphaFold2 and AlphaFold-multimer (AFM), but only 60% of dimers are accurately predicted. Here, we learn a bias to the MSA representation that improves the predictions by performing gradient descent through the AFM network. We demonstrate the performance on seven difficult targets from CASP15 and increase the average MMscore to 0.76 compared to 0.63 with AFM. We evaluate the procedure on 487 protein complexes where AFM fails and obtain an increased success rate (MMscore>0.75) of 33% on these difficult targets. Our protocol, AFProfile, provides a way to direct predictions towards a defined target function guided by the MSA. We expect gradient descent over the MSA to be useful for different tasks.

AI networks can now predict the structure of protein complexes with high accuracy in the majority of cases. The accuracy of the predicted protein complexes is directly related to the quality of the input information. However, this information can be very noisy making the output of varying quality. An interesting finding is that AI networks used for structure prediction tend to know when wrong predictions are made based on confidence in the predictions themselves. Together, this suggests that one can look for more useful input information with the predicted confidence from the AI network. To improve the structure prediction of protein complexes, we here learn how to filter the input information so that AlphaFold-multimer can use it better based on the predicted confidence. We show that it is possible to do this efficiently and improve the structures in 33% of cases where AlphaFold-multimer struggles. The same filtering procedure can be used for other tasks as well, e.g. to search for alternative conformations although this remains to be studied.

[end paper 12]

[begin paper 13]

PMCID = 10613300

Title = Topological links in predicted protein complex structures reveal limitations of AlphaFold

AlphaFold is making great progress in protein structure prediction, not only for single-chain proteins but also for multi-chain protein complexes. When using AlphaFold-Multimer to predict protein‒protein complexes, we observed some unusual structures in which chains are looped around each other to form topologically intertwining links at the interface. Based on physical principles, such topological links should generally not exist in native protein complex structures unless covalent modifications of residues are involved. Although it is well known and has been well studied that protein structures may have topologically complex shapes such as knots and links, existing methods are hampered by the chain closure problem and show poor performance in identifying topologically linked structures in protein‒protein complexes. Therefore, we address the chain closure problem by using sliding windows from a local perspective and propose an algorithm to measure the topological–geometric features that can be used to identify topologically linked structures. An application of the method to AlphaFold-Multimer-predicted protein complex structures finds that approximately 1.72% of the predicted structures contain topological links. The method presented in this work will facilitate the computational study of protein‒protein interactions and help further improve the structural prediction of multi-chain protein complexes.

An efficient computational method detects the topological links in the protein complex structures and finds that the topological links nearly do not exist in PDB experimentally determined structures but exist in the AlphaFold2-Multimer predicted models.

[end paper 13]

[begin paper 14]

PMCID = 10987651

Title = AI-guided pipeline for protein–protein interaction drug discovery identifies a SARS-CoV-2 inhibitor

Protein–protein interactions (PPIs) offer great opportunities to expand the druggable proteome and therapeutically tackle various diseases, but remain challenging targets for drug discovery. Here, we provide a comprehensive pipeline that combines experimental and computational tools to identify and validate PPI targets and perform early-stage drug discovery. We have developed a machine learning approach that prioritizes interactions by analyzing quantitative data from binary PPI assays or AlphaFold-Multimer predictions. Using the quantitative assay LuTHy together with our machine learning algorithm, we identified high-confidence interactions among SARS-CoV-2 proteins for which we predicted three-dimensional structures using AlphaFold-Multimer. We employed VirtualFlow to target the contact interface of the NSP10-NSP16 SARS-CoV-2 methyltransferase complex by ultra-large virtual drug screening. Thereby, we identified a compound that binds to NSP10 and inhibits its interaction with NSP16, while also disrupting the methyltransferase activity of the complex, and SARS-CoV-2 replication. Overall, this pipeline will help to prioritize PPI targets to accelerate the discovery of early-stage drug candidates targeting protein complexes and pathways.

A new pipeline for prioritizing protein-protein interactions (PPIs) for drug discovery, combines machine learning-based scoring of quantitative PPI data, protein complex structure prediction and virtual drug screening.

A multi-adaptive support vector machine (maSVM) classifier is used for scoring PPIs from quantitative interaction and structure prediction data.

The machine learning-based classifier is applicable to PPI datasets from various assays and AlphaFold-Multimer predictions improving comparability between different methods.

Interaction mapping with LuTHy and maSVM-based scoring identified high-confidence SARS-CoV-2 PPIs. Subsequent AlphaFold-Multimer predictions revealed key interaction residues within the NSP10-NSP16 methyltransferase complex.

Targeting the complex with virtual compound screening identified an early-stage small molecule inhibitor that disrupts the NSP10-NSP16 interaction and SARS-CoV-2 replication.

A new pipeline for prioritizing protein-protein interactions (PPIs) for drug discovery, combines machine learning-based scoring of quantitative PPI data, protein complex structure prediction and virtual drug screening.

[end paper 14]

[begin paper 15]

PMCID = 10966450

Title = SpatialPPI: Three-dimensional space protein-protein interaction prediction with AlphaFold Multimer

Rapid advancements in protein sequencing technology have resulted in gaps between proteins with identified sequences and those with mapped structures. Although sequence-based predictions offer insights, they can be incomplete due to the absence of structural details. Conversely, structure-based methods face challenges with respect to newly sequenced proteins. The AlphaFold Multimer has remarkable accuracy in predicting the structure of protein complexes. However, it cannot distinguish whether the input protein sequences can interact. Nonetheless, by analyzing the information in the models predicted by the AlphaFold Multimer, we propose a highly accurate method for predicting protein interactions. This study focuses on the use of deep neural networks, specifically to analyze protein complex structures predicted by the AlphaFold Multimer. By transforming atomic coordinates and utilizing sophisticated image-processing techniques, vital 3D structural details were extracted from protein complexes. Recognizing the significance of evaluating residue distances in protein interactions, this study leveraged image recognition approaches by integrating Densely Connected Convolutional Networks (DenseNet) and Deep Residual Network (ResNet) within 3D convolutional networks for protein 3D structure analysis. When benchmarked against leading protein-protein interaction prediction methods, such as SpeedPPI, D-script, DeepTrio, and PEPPI, our proposed method, named SpatialPPI, exhibited notable efficacy, emphasizing the promising role of 3D spatial processing in advancing the realm of structural biology. The SpatialPPI code is available at: https://github.com/ohuelab/SpatialPPI.

SpatialPPI predicts protein-protein interactions based on the structure of protein complexes from AlphaFold Multimer results.

SpatialPPI renders the predicted structures into 3D tensors through spatial-based rendering and analysis using deep neural networks.

Achieves superior prediction performance of protein-protein interactions over other approaches.

SpatialPPI boosts pathway analysis and drug target identification through accurate protein-protein interaction predictions.

[end paper 15]

[begin paper 16]

PMCID = 11126948

Title = YdbH and YnbE form an intermembrane bridge to maintain lipid homeostasis in the outer membrane of Escherichia coli

Bridge-like protein structures are emerging as a conserved mechanism to transport lipids between organelles. These structures connect two membranes to efficiently move multiple lipids per transporter through a continuous hydrophobic groove proposed to shield the acyl chains of lipids as they travel through an aqueous cellular compartment. The bacterial AsmA-like proteins are the ancestors of the repeating β-groove superfamily that includes proteins that bridge organelles in eukaryotic cells to transport lipids. Here, we show that the AsmA-like YdbH protein interacts with the lipoprotein YnbE to form a bridge connecting the inner and outer membranes (OMs) of Escherichia coli. We propose that this complex is one of several intermembrane bridges that didermic bacteria possess to translocate lipids to build their OMs.

The outer membrane (OM) of didermic gram-negative bacteria is essential for growth, maintenance of cellular integrity, and innate resistance to many antimicrobials. Its asymmetric lipid distribution, with phospholipids in the inner leaflet and lipopolysaccharides (LPS) in the outer leaflet, is required for these functions. Lpt proteins form a transenvelope bridge that transports newly synthesized LPS from the inner membrane (IM) to OM, but how the bulk of phospholipids are transported between these membranes is poorly understood. Recently, three members of the AsmA-like protein family, TamB, YhdP, and YdbH, were shown to be functionally redundant and were proposed to transport phospholipids between IM and OM in Escherichia coli. These proteins belong to the repeating β-groove superfamily, which includes eukaryotic lipid-transfer proteins that mediate phospholipid transport between organelles at contact sites. Here, we show that the IM-anchored YdbH protein interacts with the OM lipoprotein YnbE to form a functional protein bridge between the IM and OM in E. coli. Based on AlphaFold-Multimer predictions, genetic data, and in vivo site-directed cross-linking, we propose that YnbE interacts with YdbH through β-strand augmentation to extend the continuous hydrophobic β-groove of YdbH that is thought to shield acyl chains of phospholipids as they travel through the aqueous intermembrane periplasmic compartment. Our data also suggest that the periplasmic protein YdbL prevents extensive amyloid-like multimerization of YnbE in cells. We, therefore, propose that YdbL has a chaperone-like function that prevents uncontrolled runaway multimerization of YnbE to ensure the proper formation of the YdbH-YnbE intermembrane bridge.

[end paper 16]

[begin paper 17]

PMCID = 11380579

Title = A replisome-associated histone H3-H4 chaperone required for epigenetic inheritance

Faithful transfer of parental histones to newly replicated daughter DNA strands is critical for inheritance of epigenetic states. Although replication proteins that facilitate parental histone transfer have been identified, how intact histone H3-H4 tetramers travel from the front to the back of the replication fork remains unknown. Here, we use AlphaFold-Multimer structural predictions combined with biochemical and genetic approaches to identify the Mrc1/CLASPIN subunit of the replisome as a histone chaperone. Mrc1 contains a conserved histone binding domain that forms a brace around the H3-H4 tetramer mimicking nucleosomal DNA and H2A-H2B histones, is required for heterochromatin inheritance, and promotes parental histone recycling during replication. We further identify binding sites for the FACT histone chaperone in Swi1/TIMELESS and DNA polymerase α that are required for heterochromatin inheritance. We propose that Mrc1, in concert with FACT acting as a mobile co-chaperone, coordinates the distribution of parental histones to newly replicated DNA.

A histone H3-H4 binding domain in the Mrc1/CLASPN component of the replisome is required for parental histone transfer and epigenetic inheritance of heterochromatin. The predicted location of Mrc1 and other histone binding proteins in the replisome suggests a path for parental histone transfer to newly synthesized DNA.

[end paper 17]

[begin paper 18]

PMCID = 11226077

Title = Protein loop structure prediction by community-based deep learning and its application to antibody CDR H3 loop modeling

As of now, more than 60 years have passed since the first determination of protein structures through crystallography, and a significant portion of protein structures can be predicted by computers. This is due to the groundbreaking enhancement in protein structure prediction achieved through neural network training utilizing extensive sequence and structure data. However, substantial challenges persist in structure prediction due to limited data availability, with antibody structure prediction standing as one such challenge. In this paper, we propose a novel neural network architecture that effectively enables structure prediction by reflecting the inherent combinatorial nature involved in protein structure formation. The core idea of this neural network architecture is not solely to track and generate a single structure but rather to form a community of multiple structures and pursue accurate structure prediction by exchanging information among community members. Applying this concept to antibody CDR H3 loop structure prediction resulted in improved structure sampling. Such an approach could be applied in the structural and functional studies of proteins, particularly in exploring various physiological processes mediated by loops. Moreover, it holds potential in addressing various other types of combinatorial structure prediction and design problems.

In this paper, we propose a new architecture that aims to improve upon protein structure prediction algorithms like AlphaFold or RoseTTAFold by considering the combinatorial nature of protein structure formation. Such an architecture, reflecting the physical principles of nature, is expected to yield beneficial results, particularly in scenarios with limited structure and sequence information. Named ComMat, this architecture does not focus on a single structure but rather on a set of multiple structures—a community—simultaneously. In this process, combinatorial exploration of protein structure is encouraged through information exchange among community members. ComMat is an instance that integrates this idea within the structure module of AlphaFold. Applying ComMat to antibody CDR H3 loop structure prediction yielded outstanding results in structure sampling and prediction when tested on the IgFold set and compared with IgFold and AlphaFold-Multimer. It confirmed that improved structure sampling stems from effective structural exploration. The proposed concept here could potentially be used in the development of various other combinatorial protein structure prediction and protein design methods.

[end paper 18]

[begin paper 19]

PMCID = 9235477

Title = Topsy-Turvy: integrating a global view into sequence-based PPI prediction

Computational methods to predict protein–protein interaction (PPI) typically segregate into sequence-based ‘bottom-up’ methods that infer properties from the characteristics of the individual protein sequences, or global ‘top-down’ methods that infer properties from the pattern of already known PPIs in the species of interest. However, a way to incorporate top-down insights into sequence-based bottom-up PPI prediction methods has been elusive. We thus introduce Topsy-Turvy, a method that newly synthesizes both views in a sequence-based, multi-scale, deep-learning model for PPI prediction. While Topsy-Turvy makes predictions using only sequence data, during the training phase it takes a transfer-learning approach by incorporating patterns from both global and molecular-level views of protein interaction. In a cross-species context, we show it achieves state-of-the-art performance, offering the ability to perform genome-scale, interpretable PPI prediction for non-model organisms with no existing experimental PPI data. In species with available experimental PPI data, we further present a Topsy-Turvy hybrid (TT-Hybrid) model which integrates Topsy-Turvy with a purely network-based model for link prediction that provides information about species-specific network rewiring. TT-Hybrid makes accurate predictions for both well- and sparsely-characterized proteins, outperforming both its constituent components as well as other state-of-the-art PPI prediction methods. Furthermore, running Topsy-Turvy and TT-Hybrid screens is feasible for whole genomes, and thus these methods scale to settings where other methods (e.g. AlphaFold-Multimer) might be infeasible. The generalizability, accuracy and genome-level scalability of Topsy-Turvy and TT-Hybrid unlocks a more comprehensive map of protein interaction and organization in both model and non-model organisms.

 https://topsyturvy.csail.mit.edu.

 Supplementary data are available at Bioinformatics online.

[end paper 19]

[begin paper 20]

PMCID = 11405013

Title = PPI-hotspotID for detecting protein–protein interaction hot spots from the free protein structure

Experimental detection of residues critical for protein–protein interactions (PPI) is a time-consuming, costly, and labor-intensive process. Hence, high-throughput PPI-hot spot prediction methods have been developed, but they have been validated using relatively small datasets, which may compromise their predictive reliability. Here, we introduce PPI-hotspotID, a novel method for identifying PPI-hot spots using the free protein structure, and validated it on the largest collection of experimentally confirmed PPI-hot spots to date. We explored the possibility of detecting PPI-hot spots using (i) FTMap in the PPI mode, which identifies hot spots on protein–protein interfaces from the free protein structure, and (ii) the interface residues predicted by AlphaFold-Multimer. PPI-hotspotID yielded better performance than FTMap and SPOTONE, a webserver for predicting PPI-hot spots given the protein sequence. When combined with the AlphaFold-Multimer-predicted interface residues, PPI-hotspotID yielded better performance than either method alone. Furthermore, we experimentally verified several PPI-hotspotID-predicted PPI-hot spots of eukaryotic elongation factor 2. Notably, PPI-hotspotID can reveal PPI-hot spots not obvious from complex structures, including those in indirect contact with binding partners. PPI-hotspotID serves as a valuable tool for understanding PPI mechanisms and aiding drug design. It is available as a web server (https://ppihotspotid.limlab.dnsalias.org/) and open-source code (https://github.com/wrigjz/ppihotspotid/).

[end paper 20]

[begin paper 21]

PMCID = 10844748

Title = Methylation of elongation factor 1A by yeast Efm4 or human eEF1A-KMT2 involves a beta-hairpin recognition motif and crosstalks with phosphorylation

Translation elongation factor 1A (eEF1A) is an essential and highly conserved protein required for protein synthesis in eukaryotes. In both Saccharomyces cerevisiae and human, five different methyltransferases methylate specific residues on eEF1A, making eEF1A the eukaryotic protein targeted by the highest number of dedicated methyltransferases after histone H3. eEF1A methyltransferases are highly selective enzymes, only targeting eEF1A and each targeting just one or two specific residues in eEF1A. However, the mechanism of this selectivity remains poorly understood. To reveal how S. cerevisiae elongation factor methyltransferase 4 (Efm4) specifically methylates eEF1A at K316, we have used AlphaFold-Multimer modeling in combination with crosslinking mass spectrometry (XL-MS) and enzyme mutagenesis. We find that a unique beta-hairpin motif, which extends out from the core methyltransferase fold, is important for the methylation of eEF1A K316 in vitro. An alanine mutation of a single residue on this beta-hairpin, F212, significantly reduces Efm4 activity in vitro and in yeast cells. We show that the equivalent residue in human eEF1A-KMT2 (METTL10), F220, is also important for its activity towards eEF1A in vitro. We further show that the eEF1A guanine nucleotide exchange factor, eEF1Bα, inhibits Efm4 methylation of eEF1A in vitro, likely due to competitive binding. Lastly, we find that phosphorylation of eEF1A at S314 negatively crosstalks with Efm4-mediated methylation of K316. Our findings demonstrate how protein methyltransferases can be highly selective towards a single residue on a single protein in the cell.

[end paper 21]

[begin paper 22]

PMCID = 10312674

Title = AI-guided pipeline for protein-protein interaction drug discovery identifies a SARS-CoV-2 inhibitor

Protein-protein interactions (PPIs) offer great opportunities to expand the druggable proteome and therapeutically tackle various diseases, but remain challenging targets for drug discovery. Here, we provide a comprehensive pipeline that combines experimental and computational tools to identify and validate PPI targets and perform early-stage drug discovery. We have developed a machine learning approach that prioritizes interactions by analyzing quantitative data from binary PPI assays and AlphaFold-Multimer predictions. Using the quantitative assay LuTHy together with our machine learning algorithm, we identified high-confidence interactions among SARS-CoV-2 proteins for which we predicted three-dimensional structures using AlphaFold Multimer. We employed VirtualFlow to target the contact interface of the NSP10-NSP16 SARS-CoV-2 methyltransferase complex by ultra-large virtual drug screening. Thereby, we identified a compound that binds to NSP10 and inhibits its interaction with NSP16, while also disrupting the methyltransferase activity of the complex, and SARS-CoV-2 replication. Overall, this pipeline will help to prioritize PPI targets to accelerate the discovery of early-stage drug candidates targeting protein complexes and pathways.

[end paper 22]

[begin paper 23]

PMCID = 11230293

Title = A Comparison of Antibody-Antigen Complex Sequence-to-Structure Prediction Methods and their Systematic Biases



[end paper 23]

[begin paper 24]

PMCID = 11148199

Title = Identification of type VI secretion system effector-immunity pairs using structural bioinformatics

The type VI secretion system (T6SS) is an important mediator of microbe–microbe and microbe–host interactions. Gram-negative bacteria use the T6SS to inject T6SS effectors (T6Es), which are usually proteins with toxic activity, into neighboring cells. Antibacterial effectors have cognate immunity proteins that neutralize self-intoxication. Here, we applied novel structural bioinformatic tools to perform systematic discovery and functional annotation of T6Es and their cognate immunity proteins from a dataset of 17,920 T6SS-encoding bacterial genomes. Using structural clustering, we identified 517 putative T6E families, outperforming sequence-based clustering. We developed a logistic regression model to reliably quantify protein–protein interaction of new T6E-immunity pairs, yielding candidate immunity proteins for 231 out of the 517 T6E families. We used sensitive structure-based annotation which yielded functional annotations for 51% of the T6E families, again outperforming sequence-based annotation. Next, we validated four novel T6E-immunity pairs using basic experiments in E. coli. In particular, we showed that the Pfam domain DUF3289 is a homolog of Colicin M and that DUF943 acts as its cognate immunity protein. Furthermore, we discovered a novel T6E that is a structural homolog of SleB, a lytic transglycosylase, and identified a specific glutamate that acts as its putative catalytic residue. Overall, this study applies novel structural bioinformatic tools to T6E-immunity pair discovery, and provides an extensive database of annotated T6E-immunity pairs.

Structural bioinformatic tools were utilized for the discovery of novel specialized Type VI Secretion System (T6SS) effectors and their cognate immunity proteins, highlighting their utility over standard sequence-based tools. The effector predictions were supported by experimental results.

Structural clustering provided better compression of effectors than sequence-based methods, with 517 structural clusters representing the structure space of specialized effectors in Proteobacteria.

The ipTM score from Alphafold-multimer was used as a reliable and quantitative measure for predicting candidate immunity proteins in 231 out of 517 effector clusters.

Annotations were provided for 265 out of the 517 specialized effector domain families using fast and sensitive searches with Foldseek, expanding capabilities beyond Pfam-based annotation alone.

Four putative effectors were demonstrated to be toxic to Escherichia coli, with co-expression of cognate immunity proteins neutralizing their toxicity.

Structural bioinformatic tools were utilized for the discovery of novel specialized Type VI Secretion System (T6SS) effectors and their cognate immunity proteins, highlighting their utility over standard sequence-based tools. The effector predictions were supported by experimental results.

[end paper 24]

[begin paper 25]

PMCID = 10933411

Title = Enhancing coevolutionary signals in protein–protein interaction prediction through clade-wise alignment integration

Protein–protein interactions (PPIs) play essential roles in most biological processes. The binding interfaces between interacting proteins impose evolutionary constraints that have successfully been employed to predict PPIs from multiple sequence alignments (MSAs). To construct MSAs, critical choices have to be made: how to ensure the reliable identification of orthologs, and how to optimally balance the need for large alignments versus sufficient alignment quality. Here, we propose a divide-and-conquer strategy for MSA generation: instead of building a single, large alignment for each protein, multiple distinct alignments are constructed under distinct clades in the tree of life. Coevolutionary signals are searched separately within these clades, and are only subsequently integrated using machine learning techniques. We find that this strategy markedly improves overall prediction performance, concomitant with better alignment quality. Using the popular DCA algorithm to systematically search pairs of such alignments, a genome-wide all-against-all interaction scan in a bacterial genome is demonstrated. Given the recent successes of AlphaFold in predicting direct PPIs at atomic detail, a discover-and-refine approach is proposed: our method could provide a fast and accurate strategy for pre-screening the entire genome, submitting to AlphaFold only promising interaction candidates—thus reducing false positives as well as computation time.

[end paper 25]

[begin paper 26]

PMCID = 10129313

Title = Fast, accurate antibody structure prediction from deep learning on massive set of natural antibodies

Antibodies have the capacity to bind a diverse set of antigens, and they have become critical therapeutics and diagnostic molecules. The binding of antibodies is facilitated by a set of six hypervariable loops that are diversified through genetic recombination and mutation. Even with recent advances, accurate structural prediction of these loops remains a challenge. Here, we present IgFold, a fast deep learning method for antibody structure prediction. IgFold consists of a pre-trained language model trained on 558 million natural antibody sequences followed by graph networks that directly predict backbone atom coordinates. IgFold predicts structures of similar or better quality than alternative methods (including AlphaFold) in significantly less time (under 25 s). Accurate structure prediction on this timescale makes possible avenues of investigation that were previously infeasible. As a demonstration of IgFold’s capabilities, we predicted structures for 1.4 million paired antibody sequences, providing structural insights to 500-fold more antibodies than have experimentally determined structures.

Prediction of antibody structures is critical for understanding and designing novel therapeutic and diagnostic molecules. Here, the authors present IgFold: a fast, accurate method for antibody structure prediction using an end-to-end deep learning model.

[end paper 26]

[begin paper 27]

PMCID = 11405125

Title = AI-based IsAb2.0 for antibody design

Therapeutic antibody design has garnered widespread attention, highlighting its interdisciplinary importance. Advancements in technology emphasize the critical role of designing nanobodies and humanized antibodies in antibody engineering. However, current experimental methods are costly and time-consuming. Computational approaches, while progressing, faced limitations due to insufficient structural data and the absence of a standardized protocol. To tackle these challenges, our lab previously developed IsAb1.0, an in silico antibody design protocol. Yet, IsAb1.0 lacked accuracy, had a complex procedure, and required extensive antibody bioinformation. Moreover, it overlooked nanobody and humanized antibody design, hindering therapeutic antibody development. Building upon IsAb1.0, we enhanced our design protocol with artificial intelligence methods to create IsAb2.0. IsAb2.0 utilized AlphaFold-Multimer (2.3/3.0) for accurate modeling and complex construction without templates and employed the precise FlexddG method for in silico antibody optimization. Validated through optimization of a humanized nanobody J3 (HuJ3) targeting HIV-1 gp120, IsAb2.0 predicted five mutations that can improve HuJ3-gp120 binding affinity. These predictions were confirmed by commercial software and validated through binding and neutralization assays. IsAb2.0 streamlined antibody design, offering insights into future techniques to accelerate immunotherapy development.

[end paper 27]

[begin paper 28]

PMCID = 10497418

Title = DNA-PK and the TRF2 iDDR inhibit MRN-initiated resection at leading-end telomeres

Telomeres replicated by leading-strand synthesis lack the 3′ overhang required for telomere protection. Surprisingly, resection of these blunt telomeres is initiated by the telomere-specific 5′ exonuclease Apollo rather than the Mre11–Rad50–Nbs1 (MRN) complex, the nuclease that acts at DNA breaks. Without Apollo, leading-end telomeres undergo fusion, which, as demonstrated here, is mediated by alternative end joining. Here, we show that DNA-PK and TRF2 coordinate the repression of MRN at blunt mouse telomeres. DNA-PK represses an MRN-dependent long-range resection, while the endonuclease activity of MRN–CtIP, which could cleave DNA-PK off of blunt telomere ends, is inhibited in vitro and in vivo by the iDDR of TRF2. AlphaFold-Multimer predicts a conserved association of the iDDR with Rad50, potentially interfering with CtIP binding and MRN endonuclease activation. We propose that repression of MRN-mediated resection is a conserved aspect of telomere maintenance and represents an ancient feature of DNA-PK and the iDDR.

Here, the authors show that DNA-PK and TRF2, via its iDDR, suppress MRN nucleolytic processing of leading-end blunt telomeres, which are instead processed by Apollo or, in its absence, are aberrantly fused by alternative end-joining.

[end paper 28]

[begin paper 29]

PMCID = 9859041

Title = Structure-based prediction of T cell receptor:peptide-MHC interactions

The regulatory and effector functions of T cells are initiated by the binding of their cell-surface T cell receptor (TCR) to peptides presented by major histocompatibility complex (MHC) proteins on other cells. The specificity of TCR:peptide-MHC interactions, thus, underlies nearly all adaptive immune responses. Despite intense interest, generalizable predictive models of TCR:peptide-MHC specificity remain out of reach; two key barriers are the diversity of TCR recognition modes and the paucity of training data. Inspired by recent breakthroughs in protein structure prediction achieved by deep neural networks, we evaluated structural modeling as a potential avenue for prediction of TCR epitope specificity. We show that a specialized version of the neural network predictor AlphaFold can generate models of TCR:peptide-MHC interactions that can be used to discriminate correct from incorrect peptide epitopes with substantial accuracy. Although much work remains to be done for these predictions to have widespread practical utility, we are optimistic that deep learning-based structural modeling represents a path to generalizable prediction of TCR:peptide-MHC interaction specificity.

[end paper 29]

[begin paper 30]

PMCID = 10883280

Title = Systematic discovery of protein interaction interfaces using AlphaFold and experimental validation

Structural resolution of protein interactions enables mechanistic and functional studies as well as interpretation of disease variants. However, structural data is still missing for most protein interactions because we lack computational and experimental tools at scale. This is particularly true for interactions mediated by short linear motifs occurring in disordered regions of proteins. We find that AlphaFold-Multimer predicts with high sensitivity but limited specificity structures of domain-motif interactions when using small protein fragments as input. Sensitivity decreased substantially when using long protein fragments or full length proteins. We delineated a protein fragmentation strategy particularly suited for the prediction of domain-motif interfaces and applied it to interactions between human proteins associated with neurodevelopmental disorders. This enabled the prediction of highly confident and likely disease-related novel interfaces, which we further experimentally corroborated for FBXO23-STX1B, STX1B-VAMP2, ESRRG-PSMC5, PEX3-PEX19, PEX3-PEX16, and SNRPB-GIGYF1 providing novel molecular insights for diverse biological processes. Our work highlights exciting perspectives, but also reveals clear limitations and the need for future developments to maximize the power of Alphafold-Multimer for interface predictions.

Based on thorough benchmarking of AlphaFold-Multimer a strategy for structure prediction was developed and applied to 62 protein interactions linked to neurological disease. Six novel protein interfaces were further experimentally corroborated.

AlphaFold-Multimer (AF) largely fails to predict structures of interacting proteins involving short linear motifs when using full length sequences.

A prediction strategy was developed based on protein fragmentation, which boosts AF sensitivity at costs of specificity.

Application of this strategy to 62 protein interactions linked to neurological disease resulted in 18 correct or likely correct structural models.

Six novel protein interfaces were further supported by experiments.

Based on thorough benchmarking of AlphaFold-Multimer a strategy for structure prediction was developed and applied to 62 protein interactions linked to neurological disease. Six novel protein interfaces were further experimentally corroborated.

[end paper 30]

[begin paper 31]

PMCID = 11168924

Title = Accurate structure prediction of biomolecular interactions with AlphaFold 3

The introduction of AlphaFold 2 has spurred a revolution in modelling the structure of proteins and their interactions, enabling a huge range of applications in protein modelling and design. Here we describe our AlphaFold 3 model with a substantially updated diffusion-based architecture that is capable of predicting the joint structure of complexes including proteins, nucleic acids, small molecules, ions and modified residues. The new AlphaFold model demonstrates substantially improved accuracy over many previous specialized tools: far greater accuracy for protein–ligand interactions compared with state-of-the-art docking tools, much higher accuracy for protein–nucleic acid interactions compared with nucleic-acid-specific predictors and substantially higher antibody–antigen prediction accuracy compared with AlphaFold-Multimer v.2.3. Together, these results show that high-accuracy modelling across biomolecular space is possible within a single unified deep-learning framework.

AlphaFold 3 has a substantially updated architecture that is capable of predicting the joint structure of complexes including proteins, nucleic acids, small molecules, ions and modified residues with greatly improved accuracy over many previous specialized tools.

[end paper 31]

[begin paper 32]

PMCID = 10227038

Title = ImmuneBuilder: Deep-Learning models for predicting the structures of immune proteins

Immune receptor proteins play a key role in the immune system and have shown great promise as biotherapeutics. The structure of these proteins is critical for understanding their antigen binding properties. Here, we present ImmuneBuilder, a set of deep learning models trained to accurately predict the structure of antibodies (ABodyBuilder2), nanobodies (NanoBodyBuilder2) and T-Cell receptors (TCRBuilder2). We show that ImmuneBuilder generates structures with state of the art accuracy while being far faster than AlphaFold2. For example, on a benchmark of 34 recently solved antibodies, ABodyBuilder2 predicts CDR-H3 loops with an RMSD of 2.81Å, a 0.09Å improvement over AlphaFold-Multimer, while being over a hundred times faster. Similar results are also achieved for nanobodies, (NanoBodyBuilder2 predicts CDR-H3 loops with an average RMSD of 2.89Å, a 0.55Å improvement over AlphaFold2) and TCRs. By predicting an ensemble of structures, ImmuneBuilder also gives an error estimate for every residue in its final prediction. ImmuneBuilder is made freely available, both to download (https://github.com/oxpig/ImmuneBuilder) and to use via our webserver (http://opig.stats.ox.ac.uk/webapps/newsabdab/sabpred). We also make available structural models for ~150 thousand non-redundant paired antibody sequences (10.5281/zenodo.7258553).

ImmuneBuilder is a set of deep learning models trained to predict the structure of antibodies, nanobodies, and T-Cell receptors with state-of-the-art accuracy while being much faster than AlphaFold2 and AlphaFoldMultimer.

[end paper 32]

[begin paper 33]

PMCID = 11162299

Title = Integration of protein sequence and protein–protein interaction data by hypergraph learning to identify novel protein complexes

Protein–protein interactions (PPIs) are the basis of many important biological processes, with protein complexes being the key forms implementing these interactions. Understanding protein complexes and their functions is critical for elucidating mechanisms of life processes, disease diagnosis and treatment and drug development. However, experimental methods for identifying protein complexes have many limitations. Therefore, it is necessary to use computational methods to predict protein complexes. Protein sequences can indicate the structure and biological functions of proteins, while also determining their binding abilities with other proteins, influencing the formation of protein complexes. Integrating these characteristics to predict protein complexes is very promising, but currently there is no effective framework that can utilize both protein sequence and PPI network topology for complex prediction. To address this challenge, we have developed HyperGraphComplex, a method based on hypergraph variational autoencoder that can capture expressive features from protein sequences without feature engineering, while also considering topological properties in PPI networks, to predict protein complexes. Experiment results demonstrated that HyperGraphComplex achieves satisfactory predictive performance when compared with state-of-art methods. Further bioinformatics analysis shows that the predicted protein complexes have similar attributes to known ones. Moreover, case studies corroborated the remarkable predictive capability of our model in identifying protein complexes, including 3 that were not only experimentally validated by recent studies but also exhibited high-confidence structural predictions from AlphaFold-Multimer. We believe that the HyperGraphComplex algorithm and our provided proteome-wide high-confidence protein complex prediction dataset will help elucidate how proteins regulate cellular processes in the form of complexes, and facilitate disease diagnosis and treatment and drug development. Source codes are available at https://github.com/LiDlab/HyperGraphComplex.

[end paper 33]

[begin paper 34]

PMCID = 11702601

Title = A bacterial NLR-related protein recognizes multiple unrelated phage triggers to sense infection

Immune systems must rapidly sense viral infections to initiate antiviral signaling and protect the host. Bacteria encode >100 distinct viral (phage) defense systems and each has evolved to sense crucial components or activities associated with the viral lifecycle. Here we used a high-throughput AlphaFold-multimer screen to discover that a bacterial NLR-related protein directly senses multiple phage proteins, thereby limiting immune evasion. Phages encoded as many as 5 unrelated activators that were predicted to bind the same interface of a C-terminal sensor domain. Genetic and biochemical assays confirmed activators bound to the bacterial NLR-related protein at high affinity, induced oligomerization, and initiated signaling. This work highlights how in silico strategies can identify complex protein interaction networks that regulate immune signaling across the tree of life.

[end paper 34]

[begin paper 35]

PMCID = 11065178

Title = NFATC2IP is a mediator of SUMO-dependent genome integrity

In this study, Cho et al. show that NFATC2IP plays a key role in maintaining genome stability and cell survival when protein SUMOylation is suppressed. NFATC2IP interacts with SUMO machinery (namely, SMC5/6 and UBC9) via its SUMO-like domains to control formation of chromosomal aberrations such as micronuclei and chromatin bridges and to promote SUMOylation of chromatin-associated proteins.

The post-translational modification of proteins by SUMO is crucial for cellular viability and mammalian development in part due to the contribution of SUMOylation to genome duplication and repair. To investigate the mechanisms underpinning the essential function of SUMO, we undertook a genome-scale CRISPR/Cas9 screen probing the response to SUMOylation inhibition. This effort identified 130 genes whose disruption reduces or enhances the toxicity of TAK-981, a clinical-stage inhibitor of the SUMO E1-activating enzyme. Among the strongest hits, we validated and characterized NFATC2IP, an evolutionarily conserved protein related to the fungal Esc2 and Rad60 proteins that harbors tandem SUMO-like domains. Cells lacking NFATC2IP are viable but are hypersensitive to SUMO E1 inhibition, likely due to the accumulation of mitotic chromosome bridges and micronuclei. NFATC2IP primarily acts in interphase and associates with nascent DNA, suggesting a role in the postreplicative resolution of replication or recombination intermediates. Mechanistically, NFATC2IP interacts with the SMC5/6 complex and UBC9, the SUMO E2, via its first and second SUMO-like domains, respectively. AlphaFold-Multimer modeling suggests that NFATC2IP positions and activates the UBC9–NSMCE2 complex, the SUMO E3 ligase associated with SMC5/SMC6. We conclude that NFATC2IP is a key mediator of SUMO-dependent genomic integrity that collaborates with the SMC5/6 complex.

[end paper 35]

[begin paper 36]

PMCID = 9957715

Title = Structural modelling of human complement FHR1 and two of its synthetic derivatives provides insight into their in-vivo functions

Human complement is the first line of defence against invading pathogens and is involved in tissue homeostasis. Complement-targeted therapies to treat several diseases caused by a dysregulated complement are highly desirable. Despite huge efforts invested in their development, only very few are currently available, and a deeper understanding of the numerous interactions and complement regulation mechanisms is indispensable. Two important complement regulators are human Factor H (FH) and Factor H-related protein 1 (FHR1). MFHR1 and MFHR13, two promising therapeutic candidates based on these regulators, combine the dimerization and C5-regulatory domains of FHR1 with the central C3-regulatory and cell surface-recognition domains of FH. Here, we used AlphaFold2 to model the structure of these two synthetic regulators. Moreover, we used AlphaFold-Multimer (AFM) to study possible interactions of C3 fragments and membrane attack complex (MAC) components C5, C7 and C9 in complex with FHR1, MFHR1, MFHR13 as well as the best-known MAC regulators vitronectin (Vn), clusterin and CD59, whose experimental structures remain undetermined. AFM successfully predicted the binding interfaces of FHR1 and the synthetic regulators with C3 fragments and suggested binding to C3. The models revealed structural differences in binding to these ligands through different interfaces. Additionally, AFM predictions of Vn, clusterin or CD59 with C7 or C9 agreed with previously published experimental results. Because the role of FHR1 as MAC regulator has been controversial, we analysed possible interactions with C5, C7 and C9. AFM predicted interactions of FHR1 with proteins of the terminal complement complex (TCC) as indicated by experimental observations, and located the interfaces in FHR11–2 and FHR14–5. According to AFM prediction, FHR1 might partially block the C3b binding site in C5, inhibiting C5 activation, and block C5b-7 complex formation and C9 polymerization, with similar mechanisms of action as clusterin and vitronectin. Here, we generate hypotheses and give the basis for the design of rational approaches to understand the molecular mechanism of MAC inhibition, which will facilitate the development of further complement therapeutics.

[end paper 36]

[begin paper 37]

PMCID = 10011802

Title = AlphaFold2 and its applications in the fields of biology and medicine

AlphaFold2 (AF2) is an artificial intelligence (AI) system developed by DeepMind that can predict three-dimensional (3D) structures of proteins from amino acid sequences with atomic-level accuracy. Protein structure prediction is one of the most challenging problems in computational biology and chemistry, and has puzzled scientists for 50 years. The advent of AF2 presents an unprecedented progress in protein structure prediction and has attracted much attention. Subsequent release of structures of more than 200 million proteins predicted by AF2 further aroused great enthusiasm in the science community, especially in the fields of biology and medicine. AF2 is thought to have a significant impact on structural biology and research areas that need protein structure information, such as drug discovery, protein design, prediction of protein function, et al. Though the time is not long since AF2 was developed, there are already quite a few application studies of AF2 in the fields of biology and medicine, with many of them having preliminarily proved the potential of AF2. To better understand AF2 and promote its applications, we will in this article summarize the principle and system architecture of AF2 as well as the recipe of its success, and particularly focus on reviewing its applications in the fields of biology and medicine. Limitations of current AF2 prediction will also be discussed.

[end paper 37]

[begin paper 38]

PMCID = 11009965

Title = Computational Approaches to Predict Protein–Protein Interactions in Crowded Cellular Environments

Investigating protein–protein interactions is crucial for understanding cellular biological processes because proteins often function within molecular complexes rather than in isolation. While experimental and computational methods have provided valuable insights into these interactions, they often overlook a critical factor: the crowded cellular environment. This environment significantly impacts protein behavior, including structural stability, diffusion, and ultimately the nature of binding. In this review, we discuss theoretical and computational approaches that allow the modeling of biological systems to guide and complement experiments and can thus significantly advance the investigation, and possibly the predictions, of protein–protein interactions in the crowded environment of cell cytoplasm. We explore topics such as statistical mechanics for lattice simulations, hydrodynamic interactions, diffusion processes in high-viscosity environments, and several methods based on molecular dynamics simulations. By synergistically leveraging methods from biophysics and computational biology, we review the state of the art of computational methods to study the impact of molecular crowding on protein–protein interactions and discuss its potential revolutionizing effects on the characterization of the human interactome.

[end paper 38]

[begin paper 39]

PMCID = 10151615

Title = Cross-linking mass spectrometry discovers, evaluates, and corroborates structures and protein–protein interactions in the human cell

Proteins function through their ability to form specific three-dimensional structures or interactions, but only a small proportion of these features within a proteome have been experimentally determined. Protein structural modelers have now generated predictions for millions of proteins, including for complexes, but the accuracy of these predictions lacks systematic experimental assessment. We generated a significant resource of protein cross-links within the human cell, experimentally mapping spatially constrained pairs of amino acids within proteins or between interaction interfaces. Critically, these capture proteins with native sequences, posttranslational modifications, subcellular niches, and cofactors. We demonstrate how our resource—and large-scale cross-linking mass spectrometry in general—can be used for the mapping, assessment, and contextualization of the recently expanded structural proteome.

Significant recent advances in structural biology, particularly in the field of cryoelectron microscopy, have dramatically expanded our ability to create structural models of proteins and protein complexes. However, many proteins remain refractory to these approaches because of their low abundance, low stability, or—in the case of complexes—simply not having yet been analyzed. Here, we demonstrate the power of using cross-linking mass spectrometry (XL-MS) for the high-throughput experimental assessment of the structures of proteins and protein complexes. This included those produced by high-resolution but in vitro experimental data, as well as in silico predictions based on amino acid sequence alone. We present the largest XL-MS dataset to date, describing 28,910 unique residue pairs captured across 4,084 unique human proteins and 2,110 unique protein–protein interactions. We show that models of proteins and their complexes predicted by AlphaFold2, and inspired and corroborated by the XL-MS data, offer opportunities to deeply mine the structural proteome and interactome and reveal mechanisms underlying protein structure and function.

[end paper 39]

[begin paper 40]

PMCID = 10893003

Title = Recent Progress of Protein Tertiary Structure Prediction

The prediction of three-dimensional (3D) protein structure from amino acid sequences has stood as a significant challenge in computational and structural bioinformatics for decades. Recently, the widespread integration of artificial intelligence (AI) algorithms has substantially expedited advancements in protein structure prediction, yielding numerous significant milestones. In particular, the end-to-end deep learning method AlphaFold2 has facilitated the rise of structure prediction performance to new heights, regularly competitive with experimental structures in the 14th Critical Assessment of Protein Structure Prediction (CASP14). To provide a comprehensive understanding and guide future research in the field of protein structure prediction for researchers, this review describes various methodologies, assessments, and databases in protein structure prediction, including traditionally used protein structure prediction methods, such as template-based modeling (TBM) and template-free modeling (FM) approaches; recently developed deep learning-based methods, such as contact/distance-guided methods, end-to-end folding methods, and protein language model (PLM)-based methods; multi-domain protein structure prediction methods; the CASP experiments and related assessments; and the recently released AlphaFold Protein Structure Database (AlphaFold DB). We discuss their advantages, disadvantages, and application scopes, aiming to provide researchers with insights through which to understand the limitations, contexts, and effective selections of protein structure prediction methods in protein-related fields.

[end paper 40]

[begin paper 41]

PMCID = 10484931

Title = Improving AlphaFold2-based protein tertiary structure prediction with MULTICOM in CASP15

Since the 14th Critical Assessment of Techniques for Protein Structure Prediction (CASP14), AlphaFold2 has become the standard method for protein tertiary structure prediction. One remaining challenge is to further improve its prediction. We developed a new version of the MULTICOM system to sample diverse multiple sequence alignments (MSAs) and structural templates to improve the input for AlphaFold2 to generate structural models. The models are then ranked by both the pairwise model similarity and AlphaFold2 self-reported model quality score. The top ranked models are refined by a novel structure alignment-based refinement method powered by Foldseek. Moreover, for a monomer target that is a subunit of a protein assembly (complex), MULTICOM integrates tertiary and quaternary structure predictions to account for tertiary structural changes induced by protein-protein interaction. The system participated in the tertiary structure prediction in 2022 CASP15 experiment. Our server predictor MULTICOM_refine ranked 3rd among 47 CASP15 server predictors and our human predictor MULTICOM ranked 7th among all 132 human and server predictors. The average GDT-TS score and TM-score of the first structural models that MULTICOM_refine predicted for 94 CASP15 domains are ~0.80 and ~0.92, 9.6% and 8.2% higher than ~0.73 and 0.85 of the standard AlphaFold2 predictor respectively.

AlphaFold2 is a popular protein structure prediction tool, however, achieving high accuracy remains challenging for certain proteins that share fewer homologs with the database. Here, the authors develop a new version of the MULTICOM system to improve the multi-sequence alignment, structural template, model ranking, model refinement, and hence the accuracy of AlphaFold2 prediction.

[end paper 41]

[begin paper 42]

PMCID = 11799971

Title = Physical-aware model accuracy estimation for protein complex using deep learning method

With the breakthrough of AlphaFold2 on monomers, the research focus of structure prediction has shifted to protein complexes, driving the continued development of new methods for multimer structure prediction. Therefore, it is crucial to accurately estimate quality scores for the multimer model independent of the used prediction methods. In this work, we propose a physical-aware deep learning method, DeepUMQA-PA, to evaluate the residue-wise quality of protein complex models. Given the input protein complex model, the residue-based contact area and orientation features were first constructed using Voronoi tessellation, representing the potential physical interactions and hydrophobic properties. Then, the relationship between local residues and the overall complex topology as well as the inter-residue evolutionary information are characterized by geometry-based features, protein language model embedding representation, and knowledge-based statistical potential features. Finally, these features are fed into a fused network architecture employing equivalent graph neural network and ResNet network to estimate residue-wise model accuracy. Experimental results on the CASP15 test set demonstrate that our method outperforms the state-of-the-art method DeepUMQA3 by 3.69 % and 3.49 % on Pearson and Spearman, respectively. Notably, our method achieved 16.8 % and 15.5 % improvement in Pearson and Spearman, respectively, for the evaluation of nanobody-antigens. In addition, DeepUMQA-PA achieved better MAE scores than AlphaFold-Multimer and AlphaFold3 self-assessment methods on 43 % and 50 % of the targets, respectively. All these results suggest that physical-aware information based on the area and orientation of atom-atom and atom-solvent contacts has the potential to capture sequence-structure-quality relationships of proteins, especially in the case of flexible proteins. The DeepUMQA-PA server is freely available at http://zhanglab-bioinf.com/DeepUMQA-PA/.

[end paper 42]

[begin paper 43]

PMCID = 10113261

Title = PeSTo: parameter-free geometric deep learning for accurate prediction of protein binding interfaces

Proteins are essential molecular building blocks of life, responsible for most biological functions as a result of their specific molecular interactions. However, predicting their  binding  interfaces remains a challenge. In this study, we present a geometric transformer that acts directly on atomic coordinates labeled only with element names. The resulting model—the Protein Structure Transformer, PeSTo—surpasses the current state of the art in predicting protein-protein interfaces and can also predict and differentiate between interfaces involving nucleic acids, lipids, ions, and small molecules with high confidence. Its low computational cost enables processing high volumes of structural data, such as molecular dynamics ensembles allowing for the discovery of interfaces that remain otherwise inconspicuous in static experimentally solved structures. Moreover, the growing foldome provided by de novo structural predictions can be easily analyzed, providing new opportunities to uncover unexplored biology.

Predicting protein interactions is crucial for understanding biological functions. Here, authors introduce a geometric transformer that accurately identifies protein binding interfaces, enabling new insights into unexplored biology.

[end paper 43]

[begin paper 44]

PMCID = 11623436

Title = Challenges in bridging the gap between protein structure prediction and functional interpretation

The rapid evolution of protein structure prediction tools has significantly broadened access to protein structural data. Although predicted structure models have the potential to accelerate and impact fundamental and translational research significantly, it is essential to note that they are not validated and cannot be considered the ground truth. Thus, challenges persist, particularly in capturing protein dynamics, predicting multi‐chain structures, interpreting protein function, and assessing model quality. Interdisciplinary collaborations are crucial to overcoming these obstacles. Databases like the AlphaFold Protein Structure Database, the ESM Metagenomic Atlas, and initiatives like the 3D‐Beacons Network provide FAIR access to these data, enabling their interpretation and application across a broader scientific community. Whilst substantial advancements have been made in protein structure prediction, further progress is required to address the remaining challenges. Developing training materials, nurturing collaborations, and ensuring open data sharing will be paramount in this pursuit. The continued evolution of these tools and methodologies will deepen our understanding of protein function and accelerate disease pathogenesis and drug development discoveries.

[end paper 44]

[begin paper 45]

PMCID = 10692239

Title = Multi-domain and complex protein structure prediction using inter-domain interactions from deep learning

Accurately capturing domain-domain interactions is key to understanding protein function and designing structure-based drugs. Although AlphaFold2 has made a breakthrough on single domain, it should be noted that the structure modeling for multi-domain protein and complex remains a challenge. In this study, we developed a multi-domain and complex structure assembly protocol, named DeepAssembly, based on domain segmentation and single domain modeling algorithms. Firstly, DeepAssembly uses a population-based evolutionary algorithm to assemble multi-domain proteins by inter-domain interactions inferred from a developed deep learning network. Secondly, protein complexes are assembled by means of domains rather than chains using DeepAssembly. Experimental results show that on 219 multi-domain proteins, the average inter-domain distance precision by DeepAssembly is 22.7% higher than that of AlphaFold2. Moreover, DeepAssembly improves accuracy by 13.1% for 164 multi-domain structures with low confidence deposited in AlphaFold database. We apply DeepAssembly for the prediction of 247 heterodimers. We find that DeepAssembly successfully predicts the interface (DockQ ≥ 0.23) for 32.4% of the dimers, suggesting a lighter way to assemble complex structures by treating domains as assembly units and using inter-domain interactions learned from monomer structures.

Multi-domain and complex structure assembly through domain segmentation and single domain modeling algorithms method outperforms existing state-of-the-art approaches.

[end paper 45]

[begin paper 46]

PMCID = 8874714

Title = Prediction of Monomeric and Dimeric Structures of CYP102A1 Using AlphaFold2 and AlphaFold Multimer and Assessment of Point Mutation Effect on the Efficiency of Intra- and Interprotein Electron Transfer

The three-dimensional structure of monomers and homodimers of CYP102A1/WT (wild-type) proteins and their A83F and A83I mutant forms was predicted using the AlphaFold2 (AF2) and AlphaFold Multimer (AFMultimer) programs, which were compared with the rate constants of hydroxylation reactions of these enzyme forms to determine the efficiency of intra- and interprotein electron transport in the CYP102A1 hydroxylase system. The electron transfer rate constants (ket), which determine the rate of indole hydroxylation by the CYP102A1 system, were calculated based on the distances (R) between donor-acceptor prosthetic groups (PG) FAD→FMN→HEME of these proteins using factor β, which describes an exponential decay from R the speed of electron transport (ET) according to the tunnelling mechanism. It was shown that the structure of monomers in the homodimer, calculated using the AlpfaFold Multimer program, is in good agreement with the experimental structures of globular domains (HEME-, FMN-, and FAD-domains) in CYP102A1/WT obtained by X-ray structural analysis, and the structure of isolated monomers predicted in AF2 does not coincide with the structure of monomers in the homodimer, although a high level of similarity in individual domains remains. The structures of monomers and homodimers of A83F and A83I mutants were also calculated, and their structures were compared with the wild-type protein. Significant differences in the structure of all isolated monomers with respect to the structures of monomers in homodimers were also found for them, and at the same time, insignificant differences were revealed for all homodimers. Comparative analysis for CYP102A1/WT between the calculated intra- and interprotein distances FAD→FMN→HEME and the rate constants of hydroxylation in these proteins showed that the distance between prosthetic groups both in the monomer and in the dimer allows the implementation of electron transfer between PGs, which is consistent with experimental literature data about kcat. For the mutant form of monomer A83I, an increase in the distance between PGs was obtained, which can restrict electron transportation compared to WT; however, for the dimer of this protein, a decrease in the distance between PGs was observed compared to the WT form, which can lead to an increase in the electron transfer rate constant and, accordingly, kcat. For the monomer and homodimer of the A83F mutant, the calculations showed an increase in the distance between the PGs compared to the WT form, which should have led to a decrease in the electron transfer rate, but at the same time, for the homodimer, the approach of the aromatic group F262 with heme can speed up transportation for this form and, accordingly, the rate of hydroxylation.

[end paper 46]

[begin paper 47]

PMCID = 9758053

Title = De novo design of obligate ABC-type heterotrimeric proteins

The de novo design of three protein chains that associate to form a heterotrimer (but not any of the possible two-chain heterodimers) and that can drive the assembly of higher-order branching structures is an important challenge for protein design. We designed helical heterotrimers with specificity conferred by buried hydrogen bond networks and large aromatic residues to enhance shape complementary packing. We obtained ten designs for which all three chains cooperatively assembled into heterotrimers with few or no other species present. Crystal structures of a helical bundle heterotrimer and extended versions, with helical repeat proteins fused to individual subunits, showed all three chains assembling in the designed orientation. We used these heterotrimers as building blocks to construct larger cyclic oligomers, which were structurally validated by electron microscopy. Our three-way junction designs provide new routes to complex protein nanostructures and enable the scaffolding of three distinct ligands for modulation of cell signaling.

ABC-type heterotrimeric proteins can be designed de novo using coiled coils and helical bundles as starting scaffolds, extended using helical repeat proteins and then used as building blocks for higher-order oligomeric assemblies.

[end paper 47]

[begin paper 48]

PMCID = 10533508

Title = AlphaFold-Multimer predicts cross-kingdom interactions at the plant-pathogen interface

Adapted plant pathogens from various microbial kingdoms produce hundreds of unrelated small secreted proteins (SSPs) with elusive roles. Here, we used AlphaFold-Multimer (AFM) to screen 1879 SSPs of seven tomato pathogens for interacting with six defence-related hydrolases of tomato. This screen of 11,274 protein pairs identified 15 non-annotated SSPs that are predicted to obstruct the active site of chitinases and proteases with an intrinsic fold. Four SSPs were experimentally verified to be inhibitors of pathogenesis-related subtilase P69B, including extracellular protein-36 (Ecp36) and secreted-into-xylem-15 (Six15) of the fungal pathogens Cladosporium fulvum and Fusarium oxysporum, respectively. Together with a P69B inhibitor from the bacterial pathogen Xanthomonas perforans and Kazal-like inhibitors of the oomycete pathogen Phytophthora infestans, P69B emerges as an effector hub targeted by different microbial kingdoms, consistent with a diversification of P69B orthologs and paralogs. This study demonstrates the power of artificial intelligence to predict cross-kingdom interactions at the plant-pathogen interface.

AlphaFold-Multimer was used to screen of 1,879 small secreted proteins from plant pathogens to be inhibitors of six tomato defense enzymes. Four of these inhibit subtilase P69B, showing the use of AI to predict cross-kingdom protein interactions.

[end paper 48]

[begin paper 49]

PMCID = 9570074

Title = AI-Based Protein Interaction Screening and Identification (AISID)

In this study, we presented an AISID method extending AlphaFold-Multimer’s success in structure prediction towards identifying specific protein interactions with an optimized AISIDscore. The method was tested to identify the binding proteins in 18 human TNFSF (Tumor Necrosis Factor superfamily) members for each of 27 human TNFRSF (TNF receptor superfamily) members. For each TNFRSF member, we ranked the AISIDscore among the 18 TNFSF members. The correct pairing resulted in the highest AISIDscore for 13 out of 24 TNFRSF members which have known interactions with TNFSF members. Out of the 33 correct pairing between TNFSF and TNFRSF members, 28 pairs could be found in the top five (including 25 pairs in the top three) seats in the AISIDscore ranking. Surprisingly, the specific interactions between TNFSF10 (TNF-related apoptosis-inducing ligand, TRAIL) and its decoy receptors DcR1 and DcR2 gave the highest AISIDscore in the list, while the structures of DcR1 and DcR2 are unknown. The data strongly suggests that AlphaFold-Multimer might be a useful computational screening tool to find novel specific protein bindings. This AISID method may have broad applications in protein biochemistry, extending the application of AlphaFold far beyond structure predictions.

[end paper 49]

[begin paper 50]

PMCID = 10320053

Title = PAE viewer: a webserver for the interactive visualization of the predicted aligned error for multimer structure predictions and crosslinks

The development of AlphaFold for protein structure prediction has opened a new era in structural biology. This is even more the case for AlphaFold-Multimer for the prediction of protein complexes. The interpretation of these predictions has become more important than ever, but it is difficult for the non-specialist. While an evaluation of the prediction quality is provided for monomeric protein predictions by the AlphaFold Protein Structure Database, such a tool is missing for predicted complex structures. Here, we present the PAE Viewer webserver (http://www.subtiwiki.uni-goettingen.de/v4/paeViewerDemo), an online tool for the integrated visualization of predicted protein complexes using a 3D structure display combined with an interactive representation of the Predicted Aligned Error (PAE). This metric allows an estimation of the quality of the prediction. Importantly, our webserver also allows the integration of experimental cross-linking data which helps to interpret the reliability of the structure predictions. With the PAE Viewer, the user obtains a unique online tool which for the first time allows the intuitive evaluation of the PAE for protein complex structure predictions with integrated crosslinks.

The Predicted Aligned Error (PAE) is a metric for the reliability of protein structure predictions. The PAE Viewer webserver allows the intuitive evaluation of complex structure predictions and crosslinking data.

[end paper 50]

[begin paper 51]

PMCID = 10312770

Title = Structural modeling of cytokine-receptor-JAK2 signaling complexes using AlphaFold Multimer

Homodimeric class 1 cytokine receptors include the erythropoietin (EPOR), thrombopoietin (TPOR), granulocyte colony-stimulating factor 3 (CSF3R), growth hormone (GHR), and prolactin receptors (PRLR). They are cell-surface single-pass transmembrane (TM) glycoproteins that regulate cell growth, proliferation, and differentiation and induce oncogenesis. An active TM signaling complex consists of a receptor homodimer, one or two ligands bound to the receptor extracellular domains and two molecules of Janus Kinase 2 (JAK2) constitutively associated with the receptor intracellular domains. Although crystal structures of soluble extracellular domains with ligands have been obtained for all the receptors except TPOR, little is known about the structure and dynamics of the complete TM complexes that activate the downstream JAK-STAT signaling pathway. Three-dimensional models of five human receptor complexes with cytokines and JAK2 were generated using AlphaFold Multimer. Given the large size of the complexes (from 3220 to 4074 residues), the modeling required a stepwise assembly from smaller parts with selection and validation of the models through comparisons with published experimental data. The modeling of active and inactive complexes supports a general activation mechanism that involves ligand binding to a monomeric receptor followed by receptor dimerization and rotational movement of the receptor TM α-helices causing proximity, dimerization, and activation of associated JAK2 subunits. The binding mode of two eltrombopag molecules to TM α-helices of the active TPOR dimer was proposed. The models also help elucidating the molecular basis of oncogenic mutations that may involve non-canonical activation route. Models equilibrated in explicit lipids of the plasma membrane are publicly available.

[end paper 51]

[begin paper 52]

PMCID = 10534052

Title = AFsample: improving multimer prediction with AlphaFold using massive sampling

The AlphaFold2 neural network model has revolutionized structural biology with unprecedented performance. We demonstrate that by stochastically perturbing the neural network by enabling dropout at inference combined with massive sampling, it is possible to improve the quality of the generated models. We generated ∼6000 models per target compared with 25 default for AlphaFold-Multimer, with v1 and v2 multimer network models, with and without templates, and increased the number of recycles within the network. The method was benchmarked in CASP15, and compared with AlphaFold-Multimer v2 it improved the average DockQ from 0.41 to 0.55 using identical input and was ranked at the very top in the protein assembly category when compared with all other groups participating in CASP15. The simplicity of the method should facilitate the adaptation by the field, and the method should be useful for anyone interested in modeling multimeric structures, alternate conformations, or flexible structures.

AFsample is available online at http://wallnerlab.org/AFsample.

[end paper 52]

[begin paper 53]

PMCID = 11405395

Title = A common mechanism for recruiting the Rrm3 and RTEL1 accessory helicases to the eukaryotic replisome

The eukaryotic replisome is assembled around the CMG (CDC45-MCM-GINS) replicative helicase, which encircles the leading-strand DNA template at replication forks. When CMG stalls during DNA replication termination, or at barriers such as DNA-protein crosslinks on the leading strand template, a second helicase is deployed on the lagging strand template to support replisome progression. How these ‘accessory’ helicases are targeted to the replisome to mediate barrier bypass and replication termination remains unknown. Here, by combining AlphaFold structural modelling with experimental validation, we show that the budding yeast Rrm3 accessory helicase contains two Short Linear Interaction Motifs (SLIMs) in its disordered N-terminus, which interact with CMG and the leading-strand DNA polymerase Polε on one side of the replisome. This flexible tether positions Rrm3 adjacent to the lagging strand template on which it translocates, and is critical for replication termination in vitro and Rrm3 function in vivo. The primary accessory helicase in metazoa, RTEL1, is evolutionarily unrelated to Rrm3, but binds to CMG and Polε in an analogous manner, revealing a conserved docking mechanism for accessory helicases in the eukaryotic replisome.

Accessory DNA helicases are required to overcome replisome stalling during DNA replication. This study describes a mechanism for targeting the Rrm3 accessory helicase to the budding yeast replisome via interactions with the CMG replicative helicase and DNA polymerase ε. This binding mode positions Rrm3 close to the lagging strand DNA template and is highly similar to RTEL1 in the human replisome.

Budding yeast Rrm3 contains an N-terminal Intrinsically Disordered Region (IDR) that mediates its recruitment to the replisome.

The Rrm3 IDR contains adjacent motifs that interact with the CMG helicase and DNA polymerase ε, positioning Rrm3 close to the lagging strand DNA template.

Rrm3 binding to DNA polymerase ε is critical for Rrm3 function during DNA replication in vitro and in vivo.

The metazoan accessory helicase RTEL1 interacts with CMG and DNA polymerase ε in a highly similar manner to Rrm3.

Short linear interaction motifs present in accessory DNA helicases from yeast and humans mediate their interaction with the CMG replicative helicase and DNA polymerase ε.

[end paper 53]

[begin paper 54]

PMCID = 11319189

Title = AlphaFold2 in biomedical research: facilitating the development of diagnostic strategies for disease

Proteins, as the primary executors of physiological activity, serve as a key factor in disease diagnosis and treatment. Research into their structures, functions, and interactions is essential to better understand disease mechanisms and potential therapies. DeepMind’s AlphaFold2, a deep-learning protein structure prediction model, has proven to be remarkably accurate, and it is widely employed in various aspects of diagnostic research, such as the study of disease biomarkers, microorganism pathogenicity, antigen-antibody structures, and missense mutations. Thus, AlphaFold2 serves as an exceptional tool to bridge fundamental protein research with breakthroughs in disease diagnosis, developments in diagnostic strategies, and the design of novel therapeutic approaches and enhancements in precision medicine. This review outlines the architecture, highlights, and limitations of AlphaFold2, placing particular emphasis on its applications within diagnostic research grounded in disciplines such as immunology, biochemistry, molecular biology, and microbiology.

[end paper 54]

[begin paper 55]

PMCID = 11575592

Title = ProBID-Net: a deep learning model for protein–protein binding interface design†

Protein–protein interactions are pivotal in numerous biological processes. The computational design of these interactions facilitates the creation of novel binding proteins, crucial for advancing biopharmaceutical products. With the evolution of artificial intelligence (AI), protein design tools have swiftly transitioned from scoring-function-based to AI-based models. However, many AI models for protein design are constrained by assuming complete unfamiliarity with the amino acid sequence of the input protein, a feature most suited for de novo design but posing challenges in designing protein–protein interactions when the receptor sequence is known. To bridge this gap in computational protein design, we introduce ProBID-Net. Trained using natural protein–protein complex structures and protein domain–domain interface structures, ProBID-Net can discern features from known target protein structures to design specific binding proteins based on their binding sites. In independent tests, ProBID-Net achieved interface sequence recovery rates of 52.7%, 43.9%, and 37.6%, surpassing or being on par with ProteinMPNN in binding protein design. Validated using AlphaFold-Multimer, the sequences designed by ProBID-Net demonstrated a close correspondence between the design target and the predicted structure. Moreover, the model's output can predict changes in binding affinity upon mutations in protein complexes, even in scenarios where no data on such mutations were provided during training (zero-shot prediction). In summary, the ProBID-Net model is poised to significantly advance the design of protein–protein interactions.

ProBID-Net is a deep-learning model for designing amino acids on protein–protein binding interfaces.

[end paper 55]

[begin paper 56]

PMCID = 11461844

Title = Unmasking AlphaFold to integrate experiments and predictions in multimeric complexes

Since the release of AlphaFold, researchers have actively refined its predictions and attempted to integrate it into existing pipelines for determining protein structures. These efforts have introduced a number of functionalities and optimisations at the latest Critical Assessment of protein Structure Prediction edition (CASP15), resulting in a marked improvement in the prediction of multimeric protein structures. However, AlphaFold’s capability of predicting large protein complexes is still limited and integrating experimental data in the prediction pipeline is not straightforward. In this study, we introduce AF_unmasked to overcome these limitations. Our results demonstrate that AF_unmasked can integrate experimental information to build larger or hard to predict protein assemblies with high confidence. The resulting predictions can help interpret and augment experimental data. This approach generates high quality (DockQ score > 0.8) structures even when little to no evolutionary information is available and imperfect experimental structures are used as a starting point. AF_unmasked is developed and optimised to fill incomplete experimental structures (structural inpainting), which may provide insights into protein dynamics. In summary, AF_unmasked provides an easy-to-use method that efficiently integrates experiments to predict large protein complexes more confidently.

Integrating AlphaFold (AF) predictions with experimental data is not straightforward. Here, authors introduce AF_unmasked, a tool to integrate AF with experimental information to predict large or challenging protein assemblies with high confidence.

[end paper 56]

[begin paper 57]

PMCID = 11763140

Title = Structure-Based Approaches for Protein–Protein Interaction Prediction Using Machine Learning and Deep Learning

Protein–Protein Interaction (PPI) prediction plays a pivotal role in understanding cellular processes and uncovering molecular mechanisms underlying health and disease. Structure-based PPI prediction has emerged as a robust alternative to sequence-based methods, offering greater biological accuracy by integrating three-dimensional spatial and biochemical features. This work summarizes the recent advances in computational approaches leveraging protein structure information for PPI prediction, focusing on machine learning (ML) and deep learning (DL) techniques. These methods not only improve predictive accuracy but also provide insights into functional sites, such as binding and catalytic residues. However, challenges such as limited high-resolution structural data and the need for effective negative sampling persist. Through the integration of experimental and computational tools, structure-based prediction paves the way for comprehensive proteomic network analysis, holding promise for advancements in drug discovery, biomarker identification, and personalized medicine. Future directions include enhancing scalability and dataset reliability to expand these approaches across diverse proteomes.

[end paper 57]

[begin paper 58]

PMCID = 9772387

Title = Protein complex prediction using Rosetta, AlphaFold, and mass spectrometry covalent labeling

Covalent labeling (CL) in combination with mass spectrometry can be used as an analytical tool to study and determine structural properties of protein-protein complexes. However, data from these experiments is sparse and does not unambiguously elucidate protein structure. Thus, computational algorithms are needed to deduce structure from the CL data. In this work, we present a hybrid method that combines models of protein complex subunits generated with AlphaFold with differential CL data via a CL-guided protein-protein docking in Rosetta. In a benchmark set, the RMSD (root-mean-square deviation) of the best-scoring models was below 3.6 Å for 5/5 complexes with inclusion of CL data, whereas the same quality was only achieved for 1/5 complexes without CL data. This study suggests that our integrated approach can successfully use data obtained from CL experiments to distinguish between nativelike and non-nativelike models.

Covalent labeling (CL) from mass spectrometry experiments provides structural information of higher-order protein structure. Here, the authors develop an algorithm which integrates experimental CL data to predict protein complexes in the Rosetta molecular modeling suite using AlphaFold models.

[end paper 58]

[begin paper 59]

PMCID = 9234698

Title = AlphaFold Models of Small Proteins Rival the Accuracy of Solution NMR Structures

Recent advances in molecular modeling using deep learning have the potential to revolutionize the field of structural biology. In particular, AlphaFold has been observed to provide models of protein structures with accuracies rivaling medium-resolution X-ray crystal structures, and with excellent atomic coordinate matches to experimental protein NMR and cryo-electron microscopy structures. Here we assess the hypothesis that AlphaFold models of small, relatively rigid proteins have accuracies (based on comparison against experimental data) similar to experimental solution NMR structures. We selected six representative small proteins with structures determined by both NMR and X-ray crystallography, and modeled each of them using AlphaFold. Using several structure validation tools integrated under the Protein Structure Validation Software suite (PSVS), we then assessed how well these models fit to experimental NMR data, including NOESY peak lists (RPF-DP scores), comparisons between predicted rigidity and chemical shift data (ANSURR scores), and 15N-1H residual dipolar coupling data (RDC Q factors) analyzed by software tools integrated in the PSVS suite. Remarkably, the fits to NMR data for the protein structure models predicted with AlphaFold are generally similar, or better, than for the corresponding experimental NMR or X-ray crystal structures. Similar conclusions were reached in comparing AlphaFold2 predictions and NMR structures for three targets from the Critical Assessment of Protein Structure Prediction (CASP). These results contradict the widely held misperception that AlphaFold cannot accurately model solution NMR structures. They also document the value of PSVS for model vs. data assessment of protein NMR structures, and the potential for using AlphaFold models for guiding analysis of experimental NMR data and more generally in structural biology.

[end paper 59]

[begin paper 60]

PMCID = 9973325

Title = Identification and Structural Modeling of the RNA Polymerase Omega Subunits in Chlamydiae and Other Obligate Intracellular Bacteria

Gene transcription in bacteria is carried out by the multisubunit RNA polymerase (RNAP), which is composed of a catalytic core enzyme and a promoter-recognizing σ factor. The core enzyme comprises two α subunits, one β subunit, one β′ subunit, and one ω subunit. The ω subunit plays critical roles in the assembly of the core enzyme and other cellular functions, including the regulation of bacterial growth, the stress response, and biofilm formation. However, the identity of an ω subunit for the obligate intracellular bacterium Chlamydia has not previously been determined. Here, we report the identification of the hypothetical protein CTL0286 as the probable chlamydial ω subunit based on sequence, synteny, and AlphaFold and AlphaFold-Multimer three-dimensional-structure predictions. Our findings indicate that CTL0286 functions as the missing ω subunit of chlamydial RNAP. Our extended analysis also indicates that all obligate intracellular bacteria have ω orthologs.

[end paper 60]

[begin paper 61]

PMCID = 11070728

Title = HighFold: accurately predicting structures of cyclic peptides and complexes with head-to-tail and disulfide bridge constraints

In recent years, cyclic peptides have emerged as a promising therapeutic modality due to their diverse biological activities. Understanding the structures of these cyclic peptides and their complexes is crucial for unlocking invaluable insights about protein target–cyclic peptide interaction, which can facilitate the development of novel-related drugs. However, conducting experimental observations is time-consuming and expensive. Computer-aided drug design methods are not practical enough in real-world applications. To tackles this challenge, we introduce HighFold, an AlphaFold-derived model in this study. By integrating specific details about the head-to-tail circle and disulfide bridge structures, the HighFold model can accurately predict the structures of cyclic peptides and their complexes. Our model demonstrates superior predictive performance compared to other existing approaches, representing a significant advancement in structure–activity research. The HighFold model is openly accessible at https://github.com/hongliangduan/HighFold.

[end paper 61]

[begin paper 62]

PMCID = 11564622

Title = Diversification of molecular pattern recognition in bacterial NLR-like proteins

Antiviral STANDs (Avs) are bacterial anti-phage proteins evolutionarily related to immune pattern recognition receptors of the NLR family. Type 2 Avs proteins (Avs2) were suggested to recognize the phage large terminase subunit as a signature of phage infection. Here, we show that Avs2 from Klebsiella pneumoniae (KpAvs2) can recognize several different phage proteins as signature for infection. While KpAvs2 recognizes the large terminase subunit of Seuratvirus phages, we find that to protect against Dhillonvirus phages, KpAvs2 recognizes a different phage protein named KpAvs2-stimulating protein 1 (Ksap1). KpAvs2 directly binds Ksap1 to become activated, and phages mutated in Ksap1 escape KpAvs2 defense despite encoding an intact terminase. We further show that KpAvs2 protects against a third group of phages by recognizing another protein, Ksap2. Our results exemplify the evolutionary diversification of molecular pattern recognition in bacterial Avs2, and show that a single pattern recognition receptor evolved to recognize different phage-encoded proteins.

Avs proteins are bacterial anti-phage pattern recognition receptors evolutionarily related to eukaryotic NLRs. Here, Béchon et al show that a single bacterial Avs can recognize different phage proteins as a signature for infection, explaining the broad defensive range of Avs proteins.

[end paper 62]

[begin paper 63]

PMCID = 10165273

Title = Itch receptor MRGPRX4 interacts with the receptor activity–modifying proteins

Cholestatic itch is a severe and debilitating symptom in liver diseases with limited treatment options. The class A G protein-coupled receptor (GPCR) Mas-related GPCR subtype X4 (MRGPRX4) has been identified as a receptor for bile acids, which are potential cholestatic pruritogens. An increasing number of GPCRs have been shown to interact with receptor activity–modifying proteins (RAMPs), which can modulate different aspects of GPCR biology. Using a combination of multiplexed immunoassay and proximity ligation assay, we show that MRGPRX4 interacts with RAMPs. The interaction of MRGPRX4 with RAMP2, but not RAMP1 or 3, causes attenuation of basal and agonist-dependent signaling, which correlates with a decrease of MRGPRX4 cell surface expression as measured using a quantitative NanoBRET pulse-chase assay. Finally, we use AlphaFold Multimer to predict the structure of the MRGPRX4–RAMP2 complex. The discovery that RAMP2 regulates MRGPRX4 may have direct implications for future drug development for cholestatic itch.

[end paper 63]

[begin paper 64]

PMCID = 11218997

Title = Human cytomegalovirus deploys molecular mimicry to recruit VPS4A to sites of virus assembly

The AAA-type ATPase VPS4 is recruited by proteins of the endosomal sorting complex required for transport III (ESCRT-III) to catalyse membrane constriction and membrane fission. VPS4A accumulates at the cytoplasmic viral assembly complex (cVAC) of cells infected with human cytomegalovirus (HCMV), the site where nascent virus particles obtain their membrane envelope. Here we show that VPS4A is recruited to the cVAC via interaction with pUL71. Sequence analysis, deep-learning structure prediction, molecular dynamics and mutagenic analysis identify a short peptide motif in the C-terminal region of pUL71 that is necessary and sufficient for the interaction with VPS4A. This motif is predicted to bind the same groove of the N-terminal VPS4A Microtubule-Interacting and Trafficking (MIT) domain as the Type 2 MIT-Interacting Motif (MIM2) of cellular ESCRT-III components, and this viral MIM2-like motif (vMIM2) is conserved across β-herpesvirus pUL71 homologues. However, recruitment of VPS4A by pUL71 is dispensable for HCMV morphogenesis or replication and the function of the conserved vMIM2 during infection remains enigmatic. VPS4-recruitment via a vMIM2 represents a previously unknown mechanism of molecular mimicry in viruses, extending previous observations that herpesviruses encode proteins with structural and functional homology to cellular ESCRT-III components.

Enveloped viruses appropriate cellular membranes from infected cells to generate their membrane coats. Many enveloped viruses accomplish the membrane wrapping of new virus particles by recruiting the cellular membrane-remodelling ESCRT machinery to sites of virus assembly. Herpesviruses have a complex assembly pathway and molecular roles of the ESCRT machinery in the production of new herpesvirus particles remains unclear, but previous studies have shown that the ESCRT-associated ATPase VPS4A is recruited to the cytoplasmic assembly site of human cytomegalovirus (HCMV). Here we show that the HCMV protein pUL71 is necessary for this VPS4A recruitment. Using a combination of biochemical, computational and cell-based techniques we demonstrate that a C-terminal segment of pUL71 binds VPS4A via molecular mimicry of cellular ‘MIM2’ VPS4-binding motifs. Direct binding to VPS4A via a viral MIM2 (vMIM2) represents a previously unknown mechanism of molecular mimicry in viruses, extending previous observations that herpesviruses encode proteins with structural and functional homology to cellular ESCRT-III components. However, mutational analysis reveals that VPS4A recruitment by pUL71 is not required for HCMV particle assembly, indicating that this novel and conserved HCMV vMIM2 has an unknown function of during infection.

[end paper 64]

[begin paper 65]

PMCID = 9459051

Title = RL-MLZerD: Multimeric protein docking using reinforcement learning

Numerous biological processes in a cell are carried out by protein complexes. To understand the molecular mechanisms of such processes, it is crucial to know the quaternary structures of the complexes. Although the structures of protein complexes have been determined by biophysical experiments at a rapid pace, there are still many important complex structures that are yet to be determined. To supplement experimental structure determination of complexes, many computational protein docking methods have been developed; however, most of these docking methods are designed only for docking with two chains. Here, we introduce a novel method, RL-MLZerD, which builds multiple protein complexes using reinforcement learning (RL). In RL-MLZerD a multi-chain assembly process is considered as a series of episodes of selecting and integrating pre-computed pairwise docking models in a RL framework. RL is effective in correctly selecting plausible pairwise models that fit well with other subunits in a complex. When tested on a benchmark dataset of protein complexes with three to five chains, RL-MLZerD showed better modeling performance than other existing multiple docking methods under different evaluation criteria, except against AlphaFold-Multimer in unbound docking. Also, it emerged that the docking order of multi-chain complexes can be naturally predicted by examining preferred paths of episodes in the RL computation.

[end paper 65]

[begin paper 66]

PMCID = 11513016

Title = Accelerating protein–protein interaction screens with reduced AlphaFold-Multimer sampling

Discovering new protein–protein interactions (PPIs) across entire proteomes offers vast potential for understanding novel protein functions and elucidate system properties within or between an organism. While recent advances in computational structural biology, particularly AlphaFold-Multimer, have facilitated this task, scaling for large-scale screenings remains a challenge, requiring significant computational resources.

We evaluated the impact of reducing the number of models generated by AlphaFold-Multimer from five to one on the method’s ability to distinguish true PPIs from false ones. Our evaluation was conducted on a dataset containing both intra- and inter-species PPIs, which included proteins from bacterial and eukaryotic sources. We demonstrate that reducing the sampling does not compromise the accuracy of the method, offering a faster, efficient, and environmentally friendly solution for PPI predictions.

The code used in this article is available at https://github.com/MIDIfactory/AlphaFastPPi. Note that the same can be achieved using the latest version of AlphaPulldown available at https://github.com/KosinskiLab/AlphaPulldown

[end paper 66]

[begin paper 67]

PMCID = 9094657

Title = A bioactive phlebovirus-like envelope protein in a hookworm endogenous virus

Endogenous viral elements (EVEs), accounting for 15% of our genome, serve as a genetic reservoir from which new genes can emerge. Nematode EVEs are particularly diverse and informative of virus evolution. We identify Atlas virus—an intact retrovirus-like EVE in the human hookworm Ancylostoma ceylanicum, with an envelope protein genetically related to GN-GC glycoproteins from the family Phenuiviridae. A cryo-EM structure of Atlas GC reveals a class II viral membrane fusion protein fold not previously seen in retroviruses. Atlas GC has the structural hallmarks of an active fusogen. Atlas GC trimers insert into membranes with endosomal lipid compositions and low pH. When expressed on the plasma membrane, Atlas GC has cell-cell fusion activity. With its preserved biological activities, Atlas GC has the potential to acquire a cellular function. Our work reveals structural plasticity in reverse-transcribing RNA viruses.

A retroelement with a retrovirus-like genome and a phlebovirus-like envelope protein provides insights on virus evolution.

[end paper 67]

[begin paper 68]

PMCID = 10808870

Title = In silico protein function prediction: the rise of machine learning-based approaches

Proteins function as integral actors in essential life processes, rendering the realm of protein research a fundamental domain that possesses the potential to propel advancements in pharmaceuticals and disease investigation. Within the context of protein research, an imperious demand arises to uncover protein functionalities and untangle intricate mechanistic underpinnings. Due to the exorbitant costs and limited throughput inherent in experimental investigations, computational models offer a promising alternative to accelerate protein function annotation. In recent years, protein pre-training models have exhibited noteworthy advancement across multiple prediction tasks. This advancement highlights a notable prospect for effectively tackling the intricate downstream task associated with protein function prediction. In this review, we elucidate the historical evolution and research paradigms of computational methods for predicting protein function. Subsequently, we summarize the progress in protein and molecule representation as well as feature extraction techniques. Furthermore, we assess the performance of machine learning-based algorithms across various objectives in protein function prediction, thereby offering a comprehensive perspective on the progress within this field.

[end paper 68]

[begin paper 69]

PMCID = 11702703

Title = A deep learning method for predicting interactions for intrinsically disordered regions of proteins

Intrinsically disordered proteins or regions (IDPs/IDRs) adopt diverse binding modes with different partners, ranging from ordered to multivalent to fuzzy conformations in the bound state. Characterizing IDR interfaces is challenging experimentally and computationally. Alphafold-multimer and Alphafold3, the state-of-the-art structure prediction methods, are less accurate at predicting IDR binding sites at their benchmarked confidence cutoffs. Their performance improves upon lowering the confidence cutoffs. Here, we developed Disobind, a deep-learning method that predicts inter-protein contact maps and interface residues for an IDR and a partner protein, given their sequences. It outperforms AlphaFold-multimer and AlphaFold3 at multiple confidence cutoffs. Combining the Disobind and AlphaFold-multimer predictions further improves the performance. In contrast to most current methods, Disobind considers the context of the binding partner and does not depend on structures and multiple sequence alignments. Its predictions can be used to localize IDRs in integrative structures of large assemblies and characterize and modulate IDR-mediated interactions.

[end paper 69]

[begin paper 70]

PMCID = 10281963

Title = PIQLE: protein–protein interface quality estimation by deep graph learning of multimeric interaction geometries

Accurate modeling of protein–protein interaction interface is essential for high-quality protein complex structure prediction. Existing approaches for estimating the quality of a predicted protein complex structural model utilize only the physicochemical properties or energetic contributions of the interacting atoms, ignoring evolutionarily information or inter-atomic multimeric geometries, including interaction distance and orientations.

Here, we present PIQLE, a deep graph learning method for protein–protein interface quality estimation. PIQLE leverages multimeric interaction geometries and evolutionarily information along with sequence- and structure-derived features to estimate the quality of individual interactions between the interfacial residues using a multi-head graph attention network and then probabilistically combines the estimated quality for scoring the overall interface. Experimental results show that PIQLE consistently outperforms existing state-of-the-art methods including DProQA, TRScore, GNN-DOVE and DOVE on multiple independent test datasets across a wide range of evaluation metrics. Our ablation study and comparison with the self-assessment module of AlphaFold-Multimer repurposed for protein complex scoring reveal that the performance gains are connected to the effectiveness of the multi-head graph attention network in leveraging multimeric interaction geometries and evolutionary information along with other sequence- and structure-derived features adopted in PIQLE.

An open-source software implementation of PIQLE is freely available at https://github.com/Bhattacharya-Lab/PIQLE.

 Supplementary data are available at Bioinformatics Advances online.

[end paper 70]

[begin paper 71]

PMCID = 11016909

Title = Evolutionary analysis reveals the role of a non-catalytic domain of peptidyl arginine deiminase 2 in transcriptional regulation

Peptidyl arginine deiminases (PADIs) catalyze protein citrullination, a post-translational conversion of arginine to citrulline. The most widely expressed member of this family, PADI2, regulates cellular processes that impact several diseases. We hypothesized that we could gain new insights into PADI2 function through a systematic evolutionary and structural analysis. Here, we identify 20 positively selected PADI2 residues, 16 of which are structurally exposed and maintain PADI2 interactions with cognate proteins. Many of these selected residues reside in non-catalytic regions of PADI2. We validate the importance of a prominent loop in the middle domain that encompasses PADI2 L162, a residue under positive selection. This site is essential for interaction with the transcription elongation factor (P-TEFb) and mediates the active transcription of the oncogenes c-MYC, and CCNB1, as well as impacting cellular proliferation. These insights could be key to understanding and addressing the role of the PADI2 c-MYC axis in cancer progression.

Positively evolved residues are mainly in the non-catalytic domain of the PADI2

L162 of PADI2 is a positively evolved residue at the structurally exposed loop

PADI2-L162 mediates PADI2’s effective interaction with the P-TEFb complex

PADI2-L162 dictates cell proliferation and c-MYC transcription

Natural sciences; Biological sciences; Biochemistry; Molecular biology; Evolutionary biology; Bioinformatics

[end paper 71]

[begin paper 72]

PMCID = 11775587

Title = The ortholog of human DNAJC9 promotes histone H3–H4 degradation and is counteracted by Asf1 in fission yeast

Mammalian J-domain protein DNAJC9 interacts with histones H3–H4 and is important for cell proliferation. However, its exact function remains unclear. Here, we show that, in the fission yeast Schizosaccharomyces pombe, loss of Djc9, the ortholog of DNAJC9, renders the histone chaperone Asf1 no longer essential for growth. Utilizing AlphaFold-based structural prediction, we identified a histone-binding surface on Djc9 that binds to helix α3 of H3 in a manner that precludes simultaneous helix α3-binding by Asf1. Djc9 and Asf1 indeed compete for binding to the H3–H4 dimer in vitro, and an H3-α3 mutation impeding Djc9 binding also renders Asf1 non-essential, indicating that the role of Asf1 needed for growth in fission yeast is to prevent histone binding by Djc9. In the absence of Asf1, cell growth is hindered due to unrestrained Djc9-mediated downregulation of H3 and H4. In the presence of Asf1, Djc9 confers resistance to the DNA replication inhibitor hydroxyurea and dominant negative disease-related histone mutants by promoting the degradation of superfluous or dysfunctional histones. Our findings provide new insights into the function and mechanism of this conserved histone-binding protein.

[end paper 72]

[begin paper 73]

PMCID = 9907820

Title = AlphaFold-multimer predicts ATG8 protein binding motifs crucial for autophagy research

In this issue of PLOS Biology, Ibrahim and colleagues demonstrate how AlphaFold-multimer, an artificial intelligence–based structure prediction tool, can be used to identify sequence motifs binding to the ATG8 family of proteins central to autophagy.

This Primer explores the implications of a new PLOS Biology study which demonstrates how the AI-based structure prediction tool AlphaFold-multimer can be used to identify sequence motifs that bind to the ATG8 family of proteins central to autophagy.

[end paper 73]

[begin paper 74]

PMCID = 10760632

Title = Characterization of the SF3B1–SUGP1 interface reveals how numerous cancer mutations cause mRNA missplicing

In this study, Zhang et al. provide molecular and structural insights into the cancer-relevant interaction between spliceosome factors SF3B1 and SUGP1. They further elucidate the effect of cancer-associated mutations in SF3B1 and SUGP1 that weaken their interaction, disrupting the formation of a larger complex with DHX15 and causing pre-mRNA splicing defects.

The spliceosomal gene SF3B1 is frequently mutated in cancer. While it is known that SF3B1 hotspot mutations lead to loss of splicing factor SUGP1 from spliceosomes, the cancer-relevant SF3B1–SUGP1 interaction has not been characterized. To address this issue, we show by structural modeling that two regions flanking the SUGP1 G-patch make numerous contacts with the region of SF3B1 harboring hotspot mutations. Experiments confirmed that all the cancer-associated mutations in these regions, as well as mutations affecting other residues in the SF3B1–SUGP1 interface, not only weaken or disrupt the interaction but also alter splicing similarly to SF3B1 cancer mutations. Finally, structural modeling of a trimeric protein complex reveals that the SF3B1–SUGP1 interaction “loops out” the G-patch for interaction with the helicase DHX15. Our study thus provides an unprecedented molecular view of a protein complex essential for accurate splicing and also reveals that numerous cancer-associated mutations disrupt the critical SF3B1–SUGP1 interaction.

[end paper 74]

[begin paper 75]

PMCID = 10540290

Title = Disrupting Dimeric β-Amyloid by Electric Fields

The early oligomers of the amyloid Aβ peptide are implicated in Alzheimer’s disease, but their transient nature complicates the characterization of their structure and toxicity. Here, we investigate the stability of the minimal toxic species, i.e., β-amyloid dimers, in the presence of an oscillating electric field. We first use deep learning (AlphaFold-multimer) for generating initial models of Aβ42 dimers. The flexibility and secondary structure content of the models are then analyzed by multiple runs of molecular dynamics (MD). Structurally stable models are similar to ensemble representatives from microsecond-long MD sampling. Finally, we employ the validated model as the starting structure of MD simulations in the presence of an external oscillating electric field and observe a fast decay of β-sheet content at high field strengths. Control simulations using the helical dimer of the 42-residue leucine zipper peptide show higher structural stability than the Aβ42 dimer. The simulation results provide evidence that an external electric field (oscillating at 1 GHz) can disrupt amyloid oligomers which should be further investigated by experiments with brain organoids in vitro and eventually in vivo.

[end paper 75]

[begin paper 76]

PMCID = 11030396

Title = Predictomes: A classifier-curated database of AlphaFold-modeled protein-protein interactions

Protein-protein interactions (PPIs) are ubiquitous in biology, yet a comprehensive structural characterization of the PPIs underlying biochemical processes is lacking. Although AlphaFold-Multimer (AF-M) has the potential to fill this knowledge gap, standard AF-M confidence metrics do not reliably separate relevant PPIs from an abundance of false positive predictions. To address this limitation, we used machine learning on well curated datasets to train a Structure Prediction and Omics informed Classifier called SPOC that shows excellent performance in separating true and false PPIs, including in proteome-wide screens. We applied SPOC to an all-by-all matrix of nearly 300 human genome maintenance proteins, generating ~40,000 predictions that can be viewed at predictomes.org, where users can also score their own predictions with SPOC. High confidence PPIs discovered using our approach suggest novel hypotheses in genome maintenance. Our results provide a framework for interpreting large scale AF-M screens and help lay the foundation for a proteome-wide structural interactome.

[end paper 76]

[begin paper 77]

PMCID = 11281715

Title = African Swine Fever Virus Protein–Protein Interaction Prediction

The African swine fever virus (ASFV) is an often deadly disease in swine and poses a threat to swine livestock and swine producers. With its complex genome containing more than 150 coding regions, developing effective vaccines for this virus remains a challenge due to a lack of basic knowledge about viral protein function and protein–protein interactions between viral proteins and between viral and host proteins. In this work, we identified ASFV-ASFV protein–protein interactions (PPIs) using artificial intelligence-powered protein structure prediction tools. We benchmarked our PPI identification workflow on the Vaccinia virus, a widely studied nucleocytoplasmic large DNA virus, and found that it could identify gold-standard PPIs that have been validated in vitro in a genome-wide computational screening. We applied this workflow to more than 18,000 pairwise combinations of ASFV proteins and were able to identify seventeen novel PPIs, many of which have corroborating experimental or bioinformatic evidence for their protein–protein interactions, further validating their relevance. Two protein–protein interactions, I267L and I8L, I267L__I8L, and B175L and DP79L, B175L__DP79L, are novel PPIs involving viral proteins known to modulate host immune response.

[end paper 77]

[begin paper 78]

PMCID = 10707872

Title = Intelligent Protein Design and Molecular Characterization Techniques: A Comprehensive Review

In recent years, the widespread application of artificial intelligence algorithms in protein structure, function prediction, and de novo protein design has significantly accelerated the process of intelligent protein design and led to many noteworthy achievements. This advancement in protein intelligent design holds great potential to accelerate the development of new drugs, enhance the efficiency of biocatalysts, and even create entirely new biomaterials. Protein characterization is the key to the performance of intelligent protein design. However, there is no consensus on the most suitable characterization method for intelligent protein design tasks. This review describes the methods, characteristics, and representative applications of traditional descriptors, sequence-based and structure-based protein characterization. It discusses their advantages, disadvantages, and scope of application. It is hoped that this could help researchers to better understand the limitations and application scenarios of these methods, and provide valuable references for choosing appropriate protein characterization techniques for related research in the field, so as to better carry out protein research.

[end paper 78]

[begin paper 79]

PMCID = 11754096

Title = In situ analysis reveals the TRiC duty cycle and PDCD5 as an open-state cofactor

The ring-shaped chaperonin T-complex protein ring complex (TRiC; also known as chaperonin containing TCP-1, CCT) is an ATP-driven protein-folding machine that is essential for maintenance of cellular homeostasis. Its dysfunction is related to cancer and neurodegenerative disease. Despite its importance, how TRiC works in the cell remains unclear. Here we structurally analysed the architecture, conformational dynamics and spatial organization of the chaperonin TRiC in human cells using cryo-electron tomography. We resolved distinctive open, closed, substrate-bound and prefoldin-associated states of TRiC, and reconstructed its duty cycle in situ. The substrate-bound open and symmetrically closed TRiC states were equally abundant. Closed TRiC containing substrate forms distinctive clusters, indicative of spatial organization. Translation inhibition did not fundamentally change the distribution of duty cycle intermediates, but reduced substrate binding for all states as well as cluster formation. From our in-cell structures, we identified the programmed cell death protein 5 (PDCD5) as an interactor that specifically binds to almost all open but not closed TRiC, in a position that is compatible with both substrate and prefoldin binding. Our data support a model in which TRiC functions at near full occupancy to fold newly synthesized proteins inside cells. Defining the TRiC cycle and function inside cells lays the foundation to understand its dysfunction during cancer and neurodegeneration.

TRiC functions at near full occupancy to fold newly synthesized proteins inside cells.

[end paper 79]

[begin paper 80]

PMCID = 10554751

Title = Allosteric inhibition of the T cell receptor by a designed membrane ligand

The T cell receptor (TCR) is a complex molecular machine that directs the activation of T cells, allowing the immune system to fight pathogens and cancer cells. Despite decades of investigation, the molecular mechanism of TCR activation is still controversial. One of the leading activation hypotheses is the allosteric model. This model posits that binding of pMHC at the extracellular domain triggers a dynamic change in the transmembrane (TM) domain of the TCR subunits, which leads to signaling at the cytoplasmic side. We sought to test this hypothesis by creating a TM ligand for TCR. Previously we described a method to create a soluble peptide capable of inserting into membranes and binding to the TM domain of the receptor tyrosine kinase EphA2 (Alves et al., eLife, 2018). Here, we show that the approach is generalizable to complex membrane receptors, by designing a TM ligand for TCR. We observed that the designed peptide caused a reduction of Lck phosphorylation of TCR at the CD3ζ subunit in T cells. As a result, in the presence of this peptide inhibitor of TCR (PITCR), the proximal signaling cascade downstream of TCR activation was significantly dampened. Co-localization and co-immunoprecipitation in diisobutylene maleic acid (DIBMA) native nanodiscs confirmed that PITCR was able to bind to the TCR. AlphaFold-Multimer predicted that PITCR binds to the TM region of TCR, where it interacts with the two CD3ζ subunits. Our results additionally indicate that PITCR disrupts the allosteric changes in the compactness of the TM bundle that occur upon TCR activation, lending support to the allosteric TCR activation model. The TCR inhibition achieved by PITCR might be useful to treat inflammatory and autoimmune diseases and to prevent organ transplant rejection, as in these conditions aberrant activation of TCR contributes to disease.

[end paper 80]

[begin paper 81]

PMCID = 11228504

Title = Pairing interacting protein sequences using masked language modeling

Deep learning has brought major advances to the analysis of biological sequences. Self-supervised models, based on approaches from natural language processing and trained on large ensembles of protein sequences, efficiently learn statistical dependence in this data. This includes coevolution patterns between structurally or functionally coupled amino acids, which allows them to capture structural contacts. We propose a method to pair interacting protein sequences which leverages the power of a protein language model trained on multiple sequence alignments. Our method performs well for small datasets that are challenging for existing methods. It can improve structure prediction of protein complexes by supervised methods, which remains more challenging than that of single-chain proteins.

Predicting which proteins interact together from amino acid sequences is an important task. We develop a method to pair interacting protein sequences which leverages the power of protein language models trained on multiple sequence alignments (MSAs), such as MSA Transformer and the EvoFormer module of AlphaFold. We formulate the problem of pairing interacting partners among the paralogs of two protein families in a differentiable way. We introduce a method called Differentiable Pairing using Alignment-based Language Models (DiffPALM) that solves it by exploiting the ability of MSA Transformer to fill in masked amino acids in multiple sequence alignments using the surrounding context. MSA Transformer encodes coevolution between functionally or structurally coupled amino acids within protein chains. It also captures inter-chain coevolution, despite being trained on single-chain data. Relying on MSA Transformer without fine-tuning, DiffPALM outperforms existing coevolution-based pairing methods on difficult benchmarks of shallow multiple sequence alignments extracted from ubiquitous prokaryotic protein datasets. It also outperforms an alternative method based on a state-of-the-art protein language model trained on single sequences. Paired alignments of interacting protein sequences are a crucial ingredient of supervised deep learning methods to predict the three-dimensional structure of protein complexes. Starting from sequences paired by DiffPALM substantially improves the structure prediction of some eukaryotic protein complexes by AlphaFold-Multimer. It also achieves competitive performance with using orthology-based pairing.

[end paper 81]

[begin paper 82]

PMCID = 10497406

Title = PCNA recruits cohesin loader Scc2 to ensure sister chromatid cohesion

Sister chromatid cohesion, established during replication by the ring-shaped multiprotein complex cohesin, is essential for faithful chromosome segregation. Replisome-associated proteins are required to generate cohesion by two independent pathways. One mediates conversion of cohesins bound to unreplicated DNA ahead of replication forks into cohesive entities behind them, while the second promotes cohesin de novo loading onto newly replicated DNA. The latter process depends on the cohesin loader Scc2 (NIPBL in vertebrates) and the alternative PCNA loader CTF18-RFC. However, the mechanism of de novo cohesin loading during replication is unknown. Here we show that PCNA physically recruits the yeast cohesin loader Scc2 via its C-terminal PCNA-interacting protein motif. Binding to PCNA is crucial, as the scc2-pip mutant deficient in Scc2–PCNA interaction is defective in cohesion when combined with replisome mutants of the cohesin conversion pathway. Importantly, the role of NIPBL recruitment to PCNA for cohesion generation is conserved in vertebrate cells.

Here the authors report how cohesin loader Scc2 is recruited to chromatin during replication by the processivity-promoting factor PCNA to support de novo cohesin loading onto replicated sister DNA and ensure sister chromatid cohesion.

[end paper 82]

[begin paper 83]

PMCID = 11188072

Title = The Cryptic Bacterial Microproteome

Microproteins encoded by small open reading frames (smORFs) comprise the “dark matter” of proteomes. Although functional microproteins were identified in diverse organisms from all three domains of life, bacterial smORFs remain poorly characterized. In this comprehensive study of intergenic smORFs (ismORFs, 15–70 codons) in 5,668 bacterial genomes of the family Enterobacteriaceae, we identified 67,297 clusters of ismORFs subject to purifying selection. The ismORFs mainly code for hydrophobic, potentially transmembrane, unstructured, or minimally structured microproteins. Using AlphaFold Multimer, we predicted interactions of some of the predicted microproteins encoded by transcribed ismORFs with proteins encoded by neighboring genes, revealing the potential of microproteins to regulate the activity of various proteins, particularly, under stress. We compiled a catalog of predicted microprotein families with different levels of evidence from synteny analysis, structure prediction, and transcription and translation data. This study offers a resource for investigation of biological functions of microproteins.

[end paper 83]

[begin paper 84]

PMCID = 9674347

Title = Integrative modeling reveals the molecular architecture of the intraflagellar transport A (IFT-A) complex

Intraflagellar transport (IFT) is a conserved process of cargo transport in cilia that is essential for development and homeostasis in organisms ranging from algae to vertebrates. In humans, variants in genes encoding subunits of the cargo-adapting IFT-A and IFT-B protein complexes are a common cause of genetic diseases known as ciliopathies. While recent progress has been made in determining the atomic structure of IFT-B, little is known of the structural biology of IFT-A. Here, we combined chemical cross-linking mass spectrometry and cryo-electron tomography with AlphaFold2-based prediction of both protein structures and interaction interfaces to model the overall architecture of the monomeric six-subunit IFT-A complex, as well as its polymeric assembly within cilia. We define monomer-monomer contacts and membrane-associated regions available for association with transported cargo, and we also use this model to provide insights into the pleiotropic nature of human ciliopathy-associated genetic variants in genes encoding IFT-A subunits. Our work demonstrates the power of integration of experimental and computational strategies both for multi-protein structure determination and for understanding the etiology of human genetic disease.

[end paper 84]

[begin paper 85]

PMCID = 11697402

Title = State of the interactomes: an evaluation of molecular networks for generating biological insights

Advancements in genomic and proteomic technologies have powered the creation of large gene and protein networks (“interactomes”) for understanding biological systems. However, the proliferation of interactomes complicates the selection of networks for specific applications. Here, we present a comprehensive evaluation of 45 current human interactomes, encompassing protein-protein interactions as well as gene regulatory, signaling, colocalization, and genetic interaction networks. Our analysis shows that large composite networks such as HumanNet, STRING, and FunCoup are most effective for identifying disease genes, while smaller networks such as DIP, Reactome, and SIGNOR demonstrate stronger performance in interaction prediction. Our study provides a benchmark for interactomes across diverse biological applications and clarifies factors that influence network performance. Furthermore, our evaluation pipeline paves the way for continued assessment of emerging and updated interaction networks in the future.

Systematic evaluation of 45 human interactomes offers insights and tools for network selection across biological applications.

Skew toward high expression, protein abundance, and citation count is observed in all but the largest interactomes, and over- and under-enrichment of biological functions is seen across experimental networks.

The greatest utility for disease gene prioritization is demonstrated by large comprehensive interactomes, and updated parsimonious composite networks (PCNets) are developed for enhanced performance.

Signaling networks and AlphaFold-Multimer can be used for the prediction of interactions involving underrepresented functions, such as those involving transmembrane receptors.

Accessible python code is provided for ongoing analysis of interactome resources.

Systematic evaluation of 45 human interactomes offers insights and tools for network selection across biological applications.

[end paper 85]

[begin paper 86]

PMCID = 9992458

Title = Domain-Based Protein Docking with Extremely Large Conformational Changes

Proteins are key components in many processes in living cells, and physical interactions with other proteins and nucleic acids often form key parts of their functions. In many cases, large flexibility of proteins as they interact is key to their function. To understand the mechanisms of these processes, it is necessary to consider the 3D structures of such protein complexes. When such structures are not yet experimentally determined, protein docking has long been present to computationally generate useful structure models. However, protein docking has long had the limitation that the consideration of flexibility is usually limited to very small movements or very small structures. Methods have been developed which handle minor flexibility via normal mode or other structure sampling, but new methods are required to model ordered proteins which undergo large-scale conformational changes to elucidate their function at the molecular level. Here, we present Flex-LZerD, a framework for docking such complexes. Via partial assembly multidomain docking and an iterative normal mode analysis admitting curvilinear motions, we demonstrate the ability to model the assembly of a variety of protein-protein and protein-nucleic acid complexes.

[end paper 86]

[begin paper 87]

PMCID = 11294035

Title = Artificial intelligence in fusion protein three‐dimensional structure prediction: Review and perspective

Recent advancements in artificial intelligence (AI) have accelerated the prediction of unknown protein structures. However, accurately predicting the three‐dimensional (3D) structures of fusion proteins remains a difficult task because the current AI‐based protein structure predictions are focused on the WT proteins rather than on the newly fused proteins in nature. Following the central dogma of biology, fusion proteins are translated from fusion transcripts, which are made by transcribing the fusion genes between two different loci through the chromosomal rearrangements in cancer. Accurately predicting the 3D structures of fusion proteins is important for understanding the functional roles and mechanisms of action of new chimeric proteins. However, predicting their 3D structure using a template‐based model is challenging because known template structures are often unavailable in databases. Deep learning (DL) models that utilize multi‐level protein information have revolutionized the prediction of protein 3D structures. In this review paper, we highlighted the latest advancements and ongoing challenges in predicting the 3D structure of fusion proteins using DL models. We aim to explore both the advantages and challenges of employing AlphaFold2, RoseTTAFold, tr‐Rosetta and D‐I‐TASSER for modelling the 3D structures.

This review provides the overall pipeline and landscape of the prediction of the 3D structure of fusion protein.

This review provides the factors that should be considered in predicting the 3D structures of fusion proteins using AI approaches in each step.

This review highlights the latest advancements and ongoing challenges in predicting the 3D structure of fusion proteins using deep learning models.

This review explores the advantages and challenges of employing AlphaFold2, RoseTTAFold, tr-Rosetta, and D-I-TASSER to model 3D structures.

This review provides the overall pipeline and landscape of the prediction of the 3D structure of fusion protein.

This review provides the factors that should be considered in predicting the 3D structures of fusion proteins using AI approaches in each step.

This review highlights the latest advancements and ongoing challenges in predicting the 3D structure of fusion proteins using deep learning models.

This review explores the advantages and challenges of employing AlphaFold2, RoseTTAFold, tr‐Rosetta and D‐I‐TASSER to model 3D structures.

[end paper 87]

[begin paper 88]

PMCID = 9850923

Title = Design of Diverse Asymmetric Pockets in De Novo Homo-oligomeric Proteins

A challenge for design of protein–small-molecule recognition is that incorporation of cavities with size, shape, and composition suitable for specific recognition can considerably destabilize protein monomers. This challenge can be overcome through binding pockets formed at homo-oligomeric interfaces between folded monomers. Interfaces surrounding the central homo-oligomer symmetry axes necessarily have the same symmetry and so may not be well suited to binding asymmetric molecules. To enable general recognition of arbitrary asymmetric substrates and small molecules, we developed an approach to designing asymmetric interfaces at off-axis sites on homo-oligomers, analogous to those found in native homo-oligomeric proteins such as glutamine synthetase. We symmetrically dock curved helical repeat proteins such that they form pockets at the asymmetric interface of the oligomer with sizes ranging from several angstroms, appropriate for binding a single ion, to up to more than 20 Å across. Of the 133 proteins tested, 84 had soluble expression in E. coli, 47 had correct oligomeric states in solution, 35 had small-angle X-ray scattering (SAXS) data largely consistent with design models, and 8 had negative-stain electron microscopy (nsEM) 2D class averages showing the structures coming together as designed. Both an X-ray crystal structure and a cryogenic electron microscopy (cryoEM) structure are close to the computational design models. The nature of these proteins as homo-oligomers allows them to be readily built into higher-order structures such as nanocages, and the asymmetric pockets of these structures open rich possibilities for small-molecule binder design free from the constraints associated with monomer destabilization.

[end paper 88]

[begin paper 89]

PMCID = 10136805

Title = Using AlphaFold Predictions in Viral Research

Elucidation of the tertiary structure of proteins is an important task for biological and medical studies. AlphaFold, a modern deep-learning algorithm, enables the prediction of protein structure to a high level of accuracy. It has been applied in numerous studies in various areas of biology and medicine. Viruses are biological entities infecting eukaryotic and procaryotic organisms. They can pose a danger for humans and economically significant animals and plants, but they can also be useful for biological control, suppressing populations of pests and pathogens. AlphaFold can be used for studies of molecular mechanisms of viral infection to facilitate several activities, including drug design. Computational prediction and analysis of the structure of bacteriophage receptor-binding proteins can contribute to more efficient phage therapy. In addition, AlphaFold predictions can be used for the discovery of enzymes of bacteriophage origin that are able to degrade the cell wall of bacterial pathogens. The use of AlphaFold can assist fundamental viral research, including evolutionary studies. The ongoing development and improvement of AlphaFold can ensure that its contribution to the study of viral proteins will be significant in the future.

[end paper 89]

[begin paper 90]

PMCID = 11117562

Title = A Survey of Deep Learning Methods for Estimating the Accuracy of Protein Quaternary Structure Models

The quality prediction of quaternary structure models of a protein complex, in the absence of its true structure, is known as the Estimation of Model Accuracy (EMA). EMA is useful for ranking predicted protein complex structures and using them appropriately in biomedical research, such as protein–protein interaction studies, protein design, and drug discovery. With the advent of more accurate protein complex (multimer) prediction tools, such as AlphaFold2-Multimer and ESMFold, the estimation of the accuracy of protein complex structures has attracted increasing attention. Many deep learning methods have been developed to tackle this problem; however, there is a noticeable absence of a comprehensive overview of these methods to facilitate future development. Addressing this gap, we present a review of deep learning EMA methods for protein complex structures developed in the past several years, analyzing their methodologies, data and feature construction. We also provide a prospective summary of some potential new developments for further improving the accuracy of the EMA methods.

[end paper 90]

[begin paper 91]

PMCID = 9846249

Title = Mapping the FtsQBL divisome components in bacterial NTD pathogens as potential drug targets

Cytokinesis is an essential process in bacterial cell division, and it involves more than 25 essential/non-essential cell division proteins that form a protein complex known as a divisome. Central to the divisome are the proteins FtsB and FtsL binding to FtsQ to form a complex FtsQBL, which helps link the early proteins with late proteins. The FtsQBL complex is highly conserved as a component across bacteria. Pathogens like Vibrio cholerae, Mycobacterium ulcerans, Mycobacterium leprae, and Chlamydia trachomatis are the causative agents of the bacterial Neglected Tropical Diseases Cholera, Buruli ulcer, Leprosy, and Trachoma, respectively, some of which seemingly lack known homologs for some of the FtsQBL complex proteins. In the absence of experimental characterization, either due to insufficient resources or the massive increase in novel sequences generated from genomics, functional annotation is traditionally inferred by sequence similarity to a known homolog. With the advent of accurate protein structure prediction methods, features both at the fold level and at the protein interaction level can be used to identify orthologs that cannot be unambiguously identified using sequence similarity methods. Using the FtsQBL complex proteins as a case study, we report potential remote homologs using Profile Hidden Markov models and structures predicted using AlphaFold. Predicted ortholog structures show conformational similarity with corresponding E. coli proteins irrespective of their level of sequence similarity. Alphafold multimer was used to characterize remote homologs as FtsB or FtsL, when they were not sufficiently distinguishable at both the sequence or structure level, as their interactions with FtsQ and FtsW play a crucial role in their function. The structures were then analyzed to identify functionally critical regions of the proteins consistent with their homologs and delineate regions potentially useful for inhibitor discovery.

[end paper 91]

[begin paper 92]

PMCID = 10274727

Title = Structural Organization of the Retriever-CCC Endosomal Recycling Complex

The recycling of membrane proteins from endosomes to the cell surface is vital for cell signaling and survival. Retriever, a trimeric complex of VPS35L, VPS26C and VPS29, together with the CCC complex comprising CCDC22, CCDC93, and COMMD proteins, plays a crucial role in this process. The precise mechanisms underlying Retriever assembly and its interaction with CCC have remained elusive. Here, we present the first high-resolution structure of Retriever determined using cryogenic electron microscopy. The structure reveals a unique assembly mechanism, distinguishing it from its remotely related paralog, Retromer. By combining AlphaFold predictions and biochemical, cellular, and proteomic analyses, we further elucidate the structural organization of the entire Retriever-CCC complex and uncover how cancer-associated mutations disrupt complex formation and impair membrane protein homeostasis. These findings provide a fundamental framework for understanding the biological and pathological implications associated with Retriever-CCC-mediated endosomal recycling.

[end paper 92]

[begin paper 93]

PMCID = 10349958

Title = Evaluation of AlphaFold Antibody-Antigen Modeling with Implications for Improving Predictive Accuracy

High resolution antibody-antigen structures provide critical insights into immune recognition and can inform therapeutic design. The challenges of experimental structural determination and the diversity of the immune repertoire underscore the necessity of accurate computational tools for modeling antibody-antigen complexes. Initial benchmarking showed that despite overall success in modeling protein-protein complexes, AlphaFold and AlphaFold-Multimer have limited success in modeling antibody-antigen interactions. In this study, we performed a thorough analysis of AlphaFold’s antibody-antigen modeling performance on 429 nonredundant antibody-antigen complex structures, identifying useful confidence metrics for predicting model quality, and features of complexes associated with improved modeling success. We show the importance of bound-like component modeling in complex assembly accuracy, and that the current version of AlphaFold improves near-native modeling success to over 30%, versus approximately 20% for a previous version. With this improved success, AlphaFold can generate accurate antibody-antigen models in many cases, while additional training may further improve its performance.

[end paper 93]

[begin paper 94]

PMCID = 11507785

Title = Exploring voltage-gated sodium channel conformations and protein-protein interactions using AlphaFold2

Voltage-gated sodium (NaV) channels are vital regulators of electrical activity in excitable cells, playing critical roles in generating and propagating action potentials. Given their importance in physiology, NaV channels are key therapeutic targets for treating numerous conditions, yet developing subtype-selective drugs remains challenging due to the high sequence and structural conservation among NaV family members. Recent advances in cryo-electron microscopy have resolved nearly all human NaV channels, providing valuable insights into their structure and function. However, limitations persist in fully capturing the complex conformational states that underlie NaV channel gating and modulation. This study explores the capability of AlphaFold2 to sample multiple NaV channel conformations and assess AlphaFold Multimer’s accuracy in modeling interactions between the NaV α-subunit and its protein partners, including auxiliary β-subunits and calmodulin. We enhance conformational sampling to explore NaV channel conformations using a subsampled multiple sequence alignment approach and varying the number of recycles. Our results demonstrate that AlphaFold2 models multiple NaV channel conformations, including those from experimental structures, new states not yet experimentally identified, and potential intermediate states. Furthermore, AlphaFold Multimer models NaV complexes with auxiliary β-subunits and calmodulin with high accuracy, and the presence of protein partners significantly alters the conformational landscape of the NaV α-subunit. These findings highlight the potential of deep learning-based methods to expand our understanding of NaV channel structure, gating, and modulation, with significant implications for future drug discovery efforts.

[end paper 94]

[begin paper 95]

PMCID = 10246253

Title = Establishing comprehensive quaternary structural proteomes from genome sequence

A critical body of knowledge has developed through advances in protein microscopy, protein-fold modeling, structural biology software, availability of sequenced bacterial genomes, large-scale mutation databases, and genome-scale models. Based on these recent advances, we develop a computational platform that; i) computes the oligomeric structural proteome encoded by an organism’s genome; ii) maps multi-strain alleleomic variation, resulting in the structural proteome for a species; and iii) calculates the 3D orientation of proteins across subcellular compartments with angstrom-level precision. Using the platform, we; iv) compute the full quaternary E. coli K-12 MG1655 structural proteome; v) deploy structure-guided analyses to identify consequential mutations; and, in combination with a genome-scale model that computes proteome allocation, vi) obtain a draft 3D visualization of the proteome in a functioning cell. Thus, in conjunction with relevant datasets and computational models, we can now resolve genome-scale structural proteomes to obtain an angstrom-level understanding of whole-cell functions.

[end paper 95]

[begin paper 96]

PMCID = 11112044

Title = Peptriever: a Bi-Encoder approach for large-scale protein–peptide binding search

Peptide therapeutics hinge on the precise interaction between a tailored peptide and its designated receptor while mitigating interactions with alternate receptors is equally indispensable. Existing methods primarily estimate the binding score between protein and peptide pairs. However, for a specific peptide without a corresponding protein, it is challenging to identify the proteins it could bind due to the sheer number of potential candidates.

We propose a transformers-based protein embedding scheme in this study that can quickly identify and rank millions of interacting proteins. Furthermore, the proposed approach outperforms existing sequence- and structure-based methods, with a mean AUC-ROC and AUC-PR of 0.73.

Training data, scripts, and fine-tuned parameters are available at https://github.com/RoniGurvich/Peptriever. The proposed method is linked with a web application available for customized prediction at https://peptriever.app/.

[end paper 96]

[begin paper 97]

PMCID = 10356834

Title = GPCRome-wide analysis of G-protein-coupling diversity using a computational biology approach

GPCRs are master regulators of cell signaling by transducing extracellular stimuli into the cell via selective coupling to intracellular G-proteins. Here we present a computational analysis of the structural determinants of G-protein-coupling repertoire of experimental and predicted 3D GPCR-G-protein complexes. Interface contact analysis recapitulates structural hallmarks associated with G-protein-coupling specificity, including TM5, TM6 and ICLs. We employ interface contacts as fingerprints to cluster Gs vs Gi complexes in an unsupervised fashion, suggesting that interface residues contribute to selective coupling. We experimentally confirm on a promiscuous receptor (CCKAR) that mutations of some of these specificity-determining positions bias the coupling selectivity. Interestingly, Gs-GPCR complexes have more conserved interfaces, while Gi/o proteins adopt a wider number of alternative docking poses, as assessed via structural alignments of representative 3D complexes. Binding energy calculations demonstrate that distinct structural properties of the complexes are associated to higher stability of Gs than Gi/o complexes. AlphaFold2 predictions of experimental binary complexes confirm several of these structural features and allow us to augment the structural coverage of poorly characterized complexes such as G12/13.

Selective GPCR-G protein complexes formation is critical for signal transduction regulation. Here, the authors use a data-driven approach to show that the structures of experimental and predicted complex interfaces inform, at least partially, on G protein binding preferences.

[end paper 97]

[begin paper 98]

PMCID = 11321591

Title = Arl2 GTPase associates with the centrosomal protein Cdk5rap2 to regulate cortical development via microtubule organization

ADP ribosylation factor-like GTPase 2 (Arl2) is crucial for controlling mitochondrial fusion and microtubule assembly in various organisms. Arl2 regulates the asymmetric division of neural stem cells in Drosophila via microtubule growth. However, the function of mammalian Arl2 during cortical development was unknown. Here, we demonstrate that mouse Arl2 plays a new role in corticogenesis via regulating microtubule growth, but not mitochondria functions. Arl2 knockdown (KD) leads to impaired proliferation of neural progenitor cells (NPCs) and neuronal migration. Arl2 KD in mouse NPCs significantly diminishes centrosomal microtubule growth and delocalization of centrosomal proteins Cdk5rap2 and γ-tubulin. Moreover, Arl2 physically associates with Cdk5rap2 by in silico prediction using AlphaFold multimer, which was validated by co-immunoprecipitation and proximity ligation assay. Remarkably, Cdk5rap2 overexpression significantly rescues the neurogenesis defects caused by Arl2 KD. Therefore, Arl2 plays an important role in mouse cortical development through microtubule growth via the centrosomal protein Cdk5rap2.

Arl2 regulates asymmetric division in fly neural stem cells through the control of microtubule growth. This study shows that, during mammalian cortical development, Arl2 associates with Cdk5rap2 to control microtubule growth, thereby regulating corticogenesis.

[end paper 98]

[begin paper 99]

PMCID = 10285254

Title = The FAM86 domain of FAM86A confers substrate specificity to promote EEF2-Lys525 methylation

FAM86A is a class I lysine methyltransferase (KMT) that generates trimethylation on the eukaryotic translation elongation factor 2 (EEF2) at Lys525. Publicly available data from The Cancer Dependency Map project indicate high dependence of hundreds of human cancer cell lines on FAM86A expression. This classifies FAM86A among numerous other KMTs as potential targets for future anticancer therapies. However, selective inhibition of KMTs by small molecules can be challenging due to high conservation within the S-adenosyl methionine (SAM) cofactor binding domain among KMT subfamilies. Therefore, understanding the unique interactions within each KMT–substrate pair can facilitate developing highly specific inhibitors. The FAM86A gene encodes an N-terminal FAM86 domain of unknown function in addition to its C-terminal methyltransferase domain. Here, we used a combination of X-ray crystallography, the AlphaFold algorithms, and experimental biochemistry to identify an essential role of the FAM86 domain in mediating EEF2 methylation by FAM86A. To facilitate our studies, we also generated a selective EEF2K525 methyl antibody. Overall, this is the first report of a biological function for the FAM86 structural domain in any species and an example of a noncatalytic domain participating in protein lysine methylation. The interaction between the FAM86 domain and EEF2 provides a new strategy for developing a specific FAM86A small molecule inhibitor, and our results provide an example in which modeling a protein–protein interaction with AlphaFold expedites experimental biology.

[end paper 99]

[begin paper 100]

PMCID = 11116810

Title = ERLIN1/2 scaffolds bridge TMUB1 and RNF170 and restrict cholesterol esterification to regulate the secretory pathway

Nanodomains composed of ERLIN1 and ERLIN2 at the endoplasmic reticulum membrane mediate the interaction of RNF170 with TMUB1 and regulate the secretory pathway by limiting cholesterol esterification.

Complexes of ERLIN1 and ERLIN2 (ER lipid raft–associated 1 and 2) form large ring-like cup-shaped structures on the endoplasmic reticulum (ER) membrane and serve as platforms to bind cholesterol and E3 ubiquitin ligases, potentially defining functional nanodomains. Here, we show that ERLIN scaffolds mediate the interaction between the full-length isoform of TMUB1 (transmembrane and ubiquitin-like domain–containing 1) and RNF170 (RING finger protein 170). We identify a luminal N-terminal conserved region in TMUB1 and RNF170, which is required for this interaction. Three-dimensional modelling shows that this conserved motif binds the stomatin/prohibitin/flotillin/HflKC domain of two adjacent ERLIN subunits at different interfaces. Protein variants that preclude these interactions have been previously linked to hereditary spastic paraplegia. Using omics-based approaches in combination with phenotypic characterization of HeLa cells lacking both ERLINs, we demonstrate a role of ERLIN scaffolds in limiting cholesterol esterification, thereby favouring cholesterol transport from the ER to the Golgi apparatus and regulating Golgi morphology and the secretory pathway.

[end paper 100]



