## User

This is the beginning of an individual meeting with Immunologist to discuss your research project.

Here is the agenda for the meeting:

You are working on a research project to use machine learning to develop antibodies or nanobodies for the newest variant of the SARS-CoV-2 spike protein that also, ideally, have activity against other circulating minor variants and past variants. Your team previous decided to modify existing nanobodies to improve their binding to the newest variant of the SARS-CoV-2 spike protein. You are responsible for understanding the topic "nanobodies" in the context of designing nanobody binders for SARS-CoV-2. You need to fine-tune yourself on the relevant literature on nanobodies to improve your ability to design SARS-CoV-2 nanobody binders. Please use PubMed Central and search for relevant papers on nanobodies using the query "nanobody engineering AND cross-reactivity AND viral variants" and request 100 articles with abstracts only. Read all of the abstracts and based on each abstract individually, decide whether you want to fine-tune yourself on the full text of that paper. Include as many papers as possible, but only include papers that are directly relevant to nanobodies. Please provide the PMCIDs and titles of all the papers that you wish to fine-tune yourself on as a Python dictionary mapping PMCID as a double-quoted string to title as a double-quoted string.

Here are the agenda questions that must be answered:

1. What are the PMCIDs and titles of the papers you wish to fine-tune yourself on (as a Python dictionary mapping PMCID as a double-quoted string to title as double-quoted string)?

Immunologist, please provide your response to the agenda.

## Immunologist

Based on the abstracts provided, I have selected the following papers that are directly relevant to the topic of nanobodies in the context of SARS-CoV-2 and their cross-reactivity with viral variants. Here is the Python dictionary mapping PMCID to title:

```python
{
    "8651292": "Highly synergistic combinations of nanobodies that target SARS-CoV-2 and are resistant to escape",
    "10986514": "Nanobody engineering for SARS-CoV-2 neutralization and detection",
    "8043454": "Nanobody Repertoires for Exposing Vulnerabilities of SARS-CoV-2",
    "8951411": "Structural and Computational Studies of the SARS-CoV-2 Spike Protein Binding Mechanisms with Nanobodies: From Structure and Dynamics to Avidity-Driven Nanobody Engineering",
    "9365158": "A humanized nanobody phage display library yields potent binders of SARS CoV-2 spike",
    "9656363": "Screening, Expression, and Identification of Nanobody against SARS-CoV-2 Spike Protein",
    "9529347": "Biparatopic nanobodies targeting the receptor binding domain efficiently neutralize SARS-CoV-2",
    "8448731": "A cell-free nanobody engineering platform rapidly generates SARS-CoV-2 neutralizing nanobodies",
    "11140877": "Therapeutic nanobodies against SARS-CoV-2 and other pathogenic human coronaviruses",
    "11091055": "Enhanced potency of an IgM-like nanobody targeting conserved epitope in SARS-CoV-2 spike N-terminal domain",
    "8458290": "A potent SARS-CoV-2 neutralising nanobody shows therapeutic efficacy in the Syrian golden hamster model of COVID-19",
    "8928829": "Nebulized delivery of a broadly neutralizing SARS-CoV-2 RBD-specific nanobody prevents clinical, virological, and pathological disease in a Syrian hamster model of COVID-19",
    "9461429": "Arsenal of nanobodies shows broad-spectrum neutralization against SARS-CoV-2 variants of concern in vitro and in vivo in hamster models",
    "9792944": "Broadly neutralizing and protective nanobodies against SARS-CoV-2 Omicron subvariants BA.1, BA.2, and BA.4/5 and diverse sarbecoviruses",
    "7199733": "Structural Basis for Potent Neutralization of Betacoronaviruses by Single-Domain Camelid Antibodies",
    "9170159": "Dromedary camel nanobodies broadly neutralize SARS-CoV-2 variants",
    "8646660": "Uncovering a conserved vulnerability site in SARS‐CoV‐2 by a human antibody",
    "9288967": "Accelerating PERx reaction enables covalent nanobodies for potent neutralization of SARS-CoV-2 and variants",
    "9132424": "Deep mutational engineering of broadly-neutralizing nanobodies accommodating SARS-CoV-1 and 2 antigenic drift",
    "9734137": "A bispecific nanobody dimer broadly neutralizes SARS-CoV-1 & 2 variants of concern and offers substantial protection against Omicron via low-dose intranasal administration",
    "7932109": "Structure-guided multivalent nanobodies block SARS-CoV-2 infection and suppress mutational escape",
    "9491202": "Sensitivity to Vaccines, Therapeutic Antibodies, and Viral Entry Inhibitors and Advances To Counter the SARS-CoV-2 Omicron Variant",
    "11245638": "ISEV2024 Abstract Book",
    "8061467": "Integrative overview of antibodies against SARS-CoV-2 and their possible applications in COVID-19 prophylaxis and treatment",
    "9174178": "Superimmunity by pan-sarbecovirus nanobodies",
    "10715488": "Identification of nurse shark VNAR single-domain antibodies targeting the spike S2 subunit of SARS-CoV-2",
    "10200341": "Rapid covalent labeling of membrane proteins on living cells using a nanobody-epitope tag pair",
    "11515141": "Discovery of nanobodies: a comprehensive review of their applications and potential over the past five years",
    "10213069": "Generation of nanobodies acting as silent and positive allosteric modulators of the α7 nicotinic acetylcholine receptor",
    "10844482": "Nanobodies: a promising approach to treatment of viral diseases",
    "9392344": "K205R specific nanobody-horseradish peroxidase fusions as reagents of competitive ELISA to detect African swine fever virus serum antibodies"
}
```

These papers were chosen based on their relevance to nanobody engineering, cross-reactivity, and application against SARS-CoV-2 and its variants.

## User

Tool Output:

Here are the top 100 articles on PubMed Central for the query "nanobody engineering AND cross-reactivity AND viral variants":

[begin paper 1]

PMCID = 11112410

Title = Trimeric Bet v 1-specific nanobodies cause strong suppression of IgE binding

Around 20% of the population in Northern and Central Europe is affected by birch pollen allergy, with the major birch pollen allergen Bet v 1 as the main elicitor of allergic reactions. Together with its cross-reactive allergens from related trees and foods, Bet v 1 causes an impaired quality of life. Hence, new treatment strategies were elaborated, demonstrating the effectiveness of blocking IgG antibodies on Bet v 1-induced IgE-mediated reactions. A recent study provided evidence for the first time that Bet v 1-specific nanobodies reduce patients´ IgE binding to Bet v 1. In order to increase the potential to outcompete IgE recognition of Bet v 1 and to foster cross-reactivity and cross-protection, we developed Bet v 1-specific nanobody trimers and evaluated their capacity to suppress polyclonal IgE binding to corresponding allergens and allergen-induced basophil degranulation.

Nanobody trimers were engineered by adding isoleucine zippers, thus enabling trimeric formation. Trimers were analyzed for their cross-reactivity, binding kinetics to Bet v 1, and related allergens, and patients’ IgE inhibition potential. Finally, their efficacy to prevent basophil degranulation was investigated.

Trimers showed enhanced recognition of cross-reactive allergens and increased efficiency to reduce IgE-allergen binding compared to nanobody monomers. Furthermore, trimers displayed slow dissociation rates from allergens and suppressed allergen-induced mediator release.

We generated high-affine nanobody trimers that target Bet v 1 and related allergens. Trimers blocked IgE-allergen interaction by competing with IgE for allergen binding. They inhibited IgE-mediated release of biological mediators, demonstrating a promising potential to prevent allergic reactions caused by Bet v 1 and relatives.

[end paper 1]

[begin paper 2]

PMCID = 8651292

Title = Highly synergistic combinations of nanobodies that target SARS-CoV-2 and are resistant to escape

The emergence of SARS-CoV-2 variants threatens current vaccines and therapeutic antibodies and urgently demands powerful new therapeutics that can resist viral escape. We therefore generated a large nanobody repertoire to saturate the distinct and highly conserved available epitope space of SARS-CoV-2 spike, including the S1 receptor binding domain, N-terminal domain, and the S2 subunit, to identify new nanobody binding sites that may reflect novel mechanisms of viral neutralization. Structural mapping and functional assays show that indeed these highly stable monovalent nanobodies potently inhibit SARS-CoV-2 infection, display numerous neutralization mechanisms, are effective against emerging variants of concern, and are resistant to mutational escape. Rational combinations of these nanobodies that bind to distinct sites within and between spike subunits exhibit extraordinary synergy and suggest multiple tailored therapeutic and prophylactic strategies.

[end paper 2]

[begin paper 3]

PMCID = 10986514

Title = Nanobody engineering for SARS-CoV-2 neutralization and detection

In response to the ongoing COVID-19 pandemic, the quest for coronavirus inhibitors has inspired research on a variety of small proteins beyond conventional antibodies, including robust single-domain antibody fragments, i.e., “nanobodies.” Here, we explore the potential of nanobody engineering in the development of antivirals and diagnostic tools. Through fusion of nanobody domains that target distinct binding sites, we engineered multimodular nanobody constructs that neutralize wild-type SARS-CoV-2 and the Alpha and Delta variants at high potency, with IC50 values as low as 50 pM. Despite simultaneous binding to distinct epitopes, Beta and Omicron variants were more resistant to neutralization by the multimodular nanobodies, which highlights the importance of accounting for antigenic drift in the design of biologics. To further explore the applications of nanobody engineering in outbreak management, we present an assay based on fusions of nanobodies with fragments of NanoLuc luciferase that can detect sub-nanomolar quantities of the SARS-CoV-2 spike protein in a single step. Our work showcases the potential of nanobody engineering to combat emerging infectious diseases.

Nanobodies, small protein binders derived from the camelid antibody, are highly potent inhibitors of respiratory viruses that offer several advantages over conventional antibodies as candidates for specific therapies, including high stability and low production costs. In this work, we leverage the unique properties of nanobodies and apply them as building blocks for new therapeutic and diagnostic tools. We report ultra-potent SARS-CoV-2 inhibition by engineered nanobodies comprising multiple modules in structure-guided combinations and develop nanobodies that carry signal molecules, allowing rapid detection of the SARS-CoV-2 spike protein. Our results highlight the potential of engineered nanobodies in the development of effective countermeasures, both therapeutic and diagnostic, to manage outbreaks of emerging viruses.

[end paper 3]

[begin paper 4]

PMCID = 8043454

Title = Nanobody Repertoires for Exposing Vulnerabilities of SARS-CoV-2

Despite the great promise of vaccines, the COVID-19 pandemic is ongoing and future serious outbreaks are highly likely, so that multi-pronged containment strategies will be required for many years. Nanobodies are the smallest naturally occurring single domain antigen binding proteins identified to date, possessing numerous properties advantageous to their production and use. We present a large repertoire of high affinity nanobodies against SARS-CoV-2 Spike protein with excellent kinetic and viral neutralization properties, which can be strongly enhanced with oligomerization. This repertoire samples the epitope landscape of the Spike ectodomain inside and outside the receptor binding domain, recognizing a multitude of distinct epitopes and revealing multiple neutralization targets of pseudoviruses and authentic SARS-CoV-2, including in primary human airway epithelial cells. Combinatorial nanobody mixtures show highly synergistic activities, and are resistant to mutational escape and emerging viral variants of concern. These nanobodies establish an exceptional resource for superior COVID-19 prophylactics and therapeutics.

[end paper 4]

[begin paper 5]

PMCID = 7273724

Title = Protection From Influenza by Intramuscular Gene Vector Delivery of a Broadly Neutralizing Nanobody Does Not Depend on Antibody Dependent Cellular Cytotoxicity

Cross-subtype neutralizing single domain antibodies against influenza present new opportunities for immunoprophylaxis and pandemic preparedness. Their simple modular structure and single open reading frame format are highly amenable to gene therapy-mediated delivery. We have previously described R1a-B6, an alpaca-derived single domain antibody (nanobody), that is capable of potent cross-subtype neutralization in vitro of H1N1, H5N1, H2N2, and H9N2 influenza viruses, through binding to a highly conserved epitope in the influenza hemagglutinin stem region. To evaluate the potential of R1a-B6 for immunoprophylaxis, we have reformatted it as an Fc fusion for adeno-associated viral (AAV) vector delivery. Our findings demonstrate that a single intramuscular injection in mice of AAV encoding R1a-B6 fused to Fc fragments of different isotypes equipped either, with or without antibody dependent cellular cytotoxicity (ADCC) activity, was able to drive sustained high-level expression (0.5–1.1 mg/mL) in sera with no evidence of reduction for up to 6 months. R1a-B6-Fc fusions of both isotypes gave complete protection against lethal challenge with both pandemic A/California/07/2009 (H1N1)pdm09 and avian influenza A/Vietnam/1194/2004 (H5N1). This data suggests that R1a-B6 is capable of cross-subtype protection and ADCC was not essential for R1a-B6 efficacy. Our findings demonstrate AAV delivery of cross-subtype neutralizing nanobodies may be an effective strategy to prevent influenza infection and provide long-term protection independent of a host induced immune response.

[end paper 5]

[begin paper 6]

PMCID = 11277559

Title = Single-Domain Antibodies—Novel Tools to Study and Treat Allergies

IgE-mediated allergies represent a major health problem in the modern world. Apart from allergen-specific immunotherapy (AIT), the only disease-modifying treatment, researchers focus on biologics that target different key molecules such as allergens, IgE, or type 2 cytokines to ameliorate allergic symptoms. Single-domain antibodies, or nanobodies, are the newcomers in biotherapeutics, and their huge potential is being investigated in various research fields since their discovery 30 years ago. While they are dominantly applied for theranostics of cancer and treatment of infectious diseases, nanobodies have become increasingly substantial in allergology over the last decade. In this review, we discuss the prerequisites that we consider to be important for generating useful nanobody-based drug candidates for treating allergies. We further summarize the available research data on nanobodies used as allergen monitoring and detection probes and for therapeutic approaches. We reflect on the limitations that have to be addressed during the development process, such as in vivo half-life and immunogenicity. Finally, we speculate about novel application formats for allergy treatment that might be available in the future.

[end paper 6]

[begin paper 7]

PMCID = 8951411

Title = Structural and Computational Studies of the SARS-CoV-2 Spike Protein Binding Mechanisms with Nanobodies: From Structure and Dynamics to Avidity-Driven Nanobody Engineering

Nanobodies provide important advantages over traditional antibodies, including their smaller size and robust biochemical properties such as high thermal stability, high solubility, and the ability to be bioengineered into novel multivalent, multi-specific, and high-affinity molecules, making them a class of emerging powerful therapies against SARS-CoV-2. Recent research efforts on the design, protein engineering, and structure-functional characterization of nanobodies and their binding with SARS-CoV-2 S proteins reflected a growing realization that nanobody combinations can exploit distinct binding epitopes and leverage the intrinsic plasticity of the conformational landscape for the SARS-CoV-2 S protein to produce efficient neutralizing and mutation resistant characteristics. Structural and computational studies have also been instrumental in quantifying the structure, dynamics, and energetics of the SARS-CoV-2 spike protein binding with nanobodies. In this review, a comprehensive analysis of the current structural, biophysical, and computational biology investigations of SARS-CoV-2 S proteins and their complexes with distinct classes of nanobodies targeting different binding sites is presented. The analysis of computational studies is supplemented by an in-depth examination of mutational scanning simulations and identification of binding energy hotspots for distinct nanobody classes. The review is focused on the analysis of mechanisms underlying synergistic binding of multivalent nanobodies that can be superior to single nanobodies and conventional nanobody cocktails in combating escape mutations by effectively leveraging binding avidity and allosteric cooperativity. We discuss how structural insights and protein engineering approaches together with computational biology tools can aid in the rational design of synergistic combinations that exhibit superior binding and neutralization characteristics owing to avidity-mediated mechanisms.

[end paper 7]

[begin paper 8]

PMCID = 9365158

Title = A humanized nanobody phage display library yields potent binders of SARS CoV-2 spike

Neutralizing antibodies targeting the SARS-CoV-2 spike protein have shown a great preventative/therapeutic potential. Here, we report a rapid and efficient strategy for the development and design of SARS-CoV-2 neutralizing humanized nanobody constructs with sub-nanomolar affinities and nanomolar potencies. CryoEM-based structural analysis of the nanobodies in complex with spike revealed two distinct binding modes. The most potent nanobody, RBD-1-2G(NCATS-BL8125), tolerates the N501Y RBD mutation and remains capable of neutralizing the B.1.1.7 (Alpha) variant. Molecular dynamics simulations provide a structural basis for understanding the neutralization process of nanobodies exclusively focused on the spike-ACE2 interface with and without the N501Y mutation on RBD. A primary human airway air-lung interface (ALI) ex vivo model showed that RBD-1-2G-Fc antibody treatment was effective at reducing viral burden following WA1 and B.1.1.7 SARS-CoV-2 infections. Therefore, this presented strategy will serve as a tool to mitigate the threat of emerging SARS-CoV-2 variants.

[end paper 8]

[begin paper 9]

PMCID = 9656363

Title = Screening, Expression, and Identification of Nanobody against SARS-CoV-2 Spike Protein

Coronavirus disease 2019 (COVID-19) is caused by the severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), an infectious disease that has become a serious burden on global public health. This study screened and yielded specific nanobodies (Nbs) against SARS-CoV-2 spike protein receptor binding domain (RBD), following testing its basic characteristics. A nanobody phage library was established by immunizing a camel with RBD protein. After three rounds of panning, the positive colonies were screened by enzyme-linked immunosorbent assay (ELISA). By sequencing, four different sequences of nanobody gene fragments were selected. The four nanobody fusion proteins were expressed and purified, respectively. The specificity and affinity of the four nanobodies were identified by ELISA. Our results showed that an immune phage display library against SARS-CoV-2 has been successfully constructed with a library capacity of which was 4.7 × 108 CFU. The four purified nanobodies showed specific high-affinity binding SARS-CoV-2 S-RBD. Among these, the antigen binding affinity of Nb61 was more comparable to that of commercial rabbit anti-SARS-CoV-2 S-RBD antibodies. In sum, our study has obtained four nanobody strains against SARS-CoV-2 S-RBD with significant affinity and specificity, therefore laying an essential foundation for further research as well as the applications of diagnostic and therapeutic tools of SARS-CoV-2.

[end paper 9]

[begin paper 10]

PMCID = 9529347

Title = Biparatopic nanobodies targeting the receptor binding domain efficiently neutralize SARS-CoV-2

The development of therapeutics to prevent or treat COVID-19 remains an area of intense focus. Protein biologics, including monoclonal antibodies and nanobodies that neutralize virus, have potential for the treatment of active disease. Here, we have used yeast display of a synthetic nanobody library to isolate nanobodies that bind the receptor-binding domain (RBD) of SARS-CoV-2 and neutralize the virus. We show that combining two clones with distinct binding epitopes within the RBD into a single protein construct to generate biparatopic reagents dramatically enhances their neutralizing capacity. Furthermore, the biparatopic nanobodies exhibit enhanced control over clinically relevant RBD variants that escaped recognition by the individual nanobodies. Structural analysis of biparatopic binding to spike (S) protein revealed a unique binding mode whereby the two nanobody paratopes bridge RBDs encoded by distinct S trimers. Accordingly, biparatopic nanobodies offer a way to rapidly generate powerful viral neutralizers with enhanced ability to control viral escape mutants.

Yeast display was used to generate SARS-CoV-2 RBD-specific nanobodies

Distinct nanobody clones were fused to create biparatopic reagents

Biparatopic nanobodies protect mice from SARS-CoV-2 infection

The biparatopics bridge distinct S proteins in a unique “twinned-S” conformation

Immunology; Microbiology; Structural biology.

[end paper 10]

[begin paper 11]

PMCID = 8419096

Title = CAPSID-SPECIFIC NANOBODY EFFECTS ON HIV-1 ASSEMBLY AND INFECTIVITY

The capsid (CA) domain of the HIV-1 precursor Gag (PrGag) protein plays multiple roles in HIV-1 replication, and is central to the assembly of immature virions, and mature virus cores. CA proteins themselves are composed of N-terminal domains (NTDs) and C-terminal domains (CTDs). We have investigated the interactions of CA with anti-CA nanobodies, which derive from the antigen recognition regions of camelid heavy chain-only antibodies. The one CA NTD-specific and two CTD-specific nanobodies we analyzed proved sensitive and specific HIV-1 CA detection reagents in immunoassays. When co-expressed with HIV-1 Gag proteins in cells, the NTD-specific nanobody was efficiently assembled into virions and did not perturb virus assembly. In contrast, the two CTD-specific nanobodies reduced PrGag processing, virus release and HIV-1 infectivity. Our results demonstrate the feasibility of Gag-targeted nanobody inhibition of HIV-1.

[end paper 11]

[begin paper 12]

PMCID = 7997558

Title = Nanobody: A Small Antibody with Big Implications for Tumor Therapeutic Strategy

The development of monoclonal antibody treatments for successful tumor-targeted therapies took several decades. However, the efficacy of antibody-based therapy is still confined and desperately needs further improvement. Nanobodies are the recombinant variable domains of heavy-chain-only antibodies, with many unique properties such as small size (~15kDa), excellent solubility, superior stability, ease of manufacture, quick clearance from blood, and deep tissue penetration, which gain increasing acceptance as therapeutical tools and are considered also as building blocks for chimeric antigen receptors as well as for targeted drug delivery. Thus, one of the promising novel developments that may address the deficiency of monoclonal antibody-based therapies is the utilization of nanobodies. This article provides readers the significant factors that the structural and biochemical properties of nanobodies and the research progress on nanobodies in the fields of tumor treatment, as well as their application prospect.

[end paper 12]

[begin paper 13]

PMCID = 8448731

Title = A cell-free nanobody engineering platform rapidly generates SARS-CoV-2 neutralizing nanobodies

Antibody engineering technologies face increasing demands for speed, reliability and scale. We develop CeVICA, a cell-free nanobody engineering platform that uses ribosome display for in vitro selection of nanobodies from a library of 1011 randomized sequences. We apply CeVICA to engineer nanobodies against the Receptor Binding Domain (RBD) of SARS-CoV-2 spike protein and identify >800 binder families using a computational pipeline based on CDR-directed clustering. Among 38 experimentally-tested families, 30 are true RBD binders and 11 inhibit SARS-CoV-2 pseudotyped virus infection. Affinity maturation and multivalency engineering increase nanobody binding affinity and yield a virus neutralizer with picomolar IC50. Furthermore, the capability of CeVICA for comprehensive binder prediction allows us to validate the fitness of our nanobody library. CeVICA offers an integrated solution for rapid generation of divergent synthetic nanobodies with tunable affinities in vitro and may serve as the basis for automated and highly parallel nanobody engineering.

Faster, higher throughput antibody engineering methods are needed. Here the authors present CeVICA, a cell-free nanobody engineering platform using ribosome display and computational clustering analysis for in vitro selection; they use this to develop nanobodies against the RBD of SARS-CoV-2 spike protein.

[end paper 13]

[begin paper 14]

PMCID = 11140877

Title = Therapeutic nanobodies against SARS-CoV-2 and other pathogenic human coronaviruses

Nanobodies, single-domain antibodies derived from variable domain of camelid or shark heavy-chain antibodies, have unique properties with small size, strong binding affinity, easy construction in versatile formats, high neutralizing activity, protective efficacy, and manufactural capacity on a large-scale. Nanobodies have been arisen as an effective research tool for development of nanobiotechnologies with a variety of applications. Three highly pathogenic coronaviruses (CoVs), SARS-CoV-2, SARS-CoV, and MERS-CoV, have caused serious outbreaks or a global pandemic, and continue to post a threat to public health worldwide. The viral spike (S) protein and its cognate receptor-binding domain (RBD), which initiate viral entry and play a critical role in virus pathogenesis, are important therapeutic targets. This review describes pathogenic human CoVs, including viral structures and proteins, and S protein-mediated viral entry process. It also summarizes recent advances in development of nanobodies targeting these CoVs, focusing on those targeting the S protein and RBD. Finally, we discuss potential strategies to improve the efficacy of nanobodies against emerging SARS-CoV-2 variants and other CoVs with pandemic potential. It will provide important information for rational design and evaluation of therapeutic agents against emerging and reemerging pathogens.

[end paper 14]

[begin paper 15]

PMCID = 11091055

Title = Enhanced potency of an IgM-like nanobody targeting conserved epitope in SARS-CoV-2 spike N-terminal domain

Almost all the neutralizing antibodies targeting the receptor-binding domain (RBD) of spike (S) protein show weakened or lost efficacy against severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) emerged or emerging variants, such as Omicron and its sub-variants. This suggests that highly conserved epitopes are crucial for the development of neutralizing antibodies. Here, we present one nanobody, N235, displaying broad neutralization against the SARS-CoV-2 prototype and multiple variants, including the newly emerged Omicron and its sub-variants. Cryo-electron microscopy demonstrates N235 binds a novel, conserved, cryptic epitope in the N-terminal domain (NTD) of the S protein, which interferes with the RBD in the neighboring S protein. The neutralization mechanism interpreted via flow cytometry and Western blot shows that N235 appears to induce the S1 subunit shedding from the trimeric S complex. Furthermore, a nano-IgM construct (MN235), engineered by fusing N235 with the human IgM Fc region, displays prevention via inducing S1 shedding and cross-linking virus particles. Compared to N235, MN235 exhibits varied enhancement in neutralization against pseudotyped and authentic viruses in vitro. The intranasal administration of MN235 in low doses can effectively prevent the infection of Omicron sub-variant BA.1 and XBB in vivo, suggesting that it can be developed as a promising prophylactic antibody to cope with the ongoing and future infection.

[end paper 15]

[begin paper 16]

PMCID = 10028217

Title = Efficient virus detection utilizing chitin-immobilized nanobodies synthesized in Ustilago maydis

The COVID-19 pandemic has greatly impacted the global economy and health care systems, illustrating the urgent need for timely and inexpensive responses to pandemic threats in the form of vaccines and antigen tests. Currently, antigen testing is mostly conducted by qualitative flow chromatography or via quantitative ELISA-type assays. The latter mostly utilize materials like protein-adhesive polymers and gold or latex particles. Here we present an alternative ELISA approach using inexpensive, biogenic materials and permitting quick detection based on components produced in the microbial model Ustilago maydis. In this fungus, heterologous proteins like biopharmaceuticals can be exported by fusion to unconventionally secreted chitinase Cts1. As a unique feature, the carrier chitinase binds to chitin allowing its additional use as a purification or immobilization tag. Recent work has demonstrated that nanobodies are suitable target proteins. These proteins represent a very versatile alternative antibody format and can quickly be adapted to detect novel antigens by camelidae immunization or synthetic libraries. In this study, we exemplarily produced different mono- and bivalent SARS-CoV-2 nanobodies directed against the spike protein receptor binding domain (RBD) as Cts1 fusions and screened their antigen binding affinity in vitro and in vivo. Functional nanobody-Cts1 fusions were immobilized on chitin forming an RBD tethering surface. This provides a solid base for future development of inexpensive antigen tests utilizing unconventionally secreted nanobodies as antigen trap and a matching ubiquitous and biogenic surface for immobilization.

[end paper 16]

[begin paper 17]

PMCID = 8458290

Title = A potent SARS-CoV-2 neutralising nanobody shows therapeutic efficacy in the Syrian golden hamster model of COVID-19

SARS-CoV-2 remains a global threat to human health particularly as escape mutants emerge. There is an unmet need for effective treatments against COVID-19 for which neutralizing single domain antibodies (nanobodies) have significant potential. Their small size and stability mean that nanobodies are compatible with respiratory administration. We report four nanobodies (C5, H3, C1, F2) engineered as homotrimers with pmolar affinity for the receptor binding domain (RBD) of the SARS-CoV-2 spike protein. Crystal structures show C5 and H3 overlap the ACE2 epitope, whilst C1 and F2 bind to a different epitope. Cryo Electron Microscopy shows C5 binding results in an all down arrangement of the Spike protein. C1, H3 and C5 all neutralize the Victoria strain, and the highly transmissible Alpha (B.1.1.7 first identified in Kent, UK) strain and C1 also neutralizes the Beta (B.1.35, first identified in South Africa). Administration of C5-trimer via the respiratory route showed potent therapeutic efficacy in the Syrian hamster model of COVID-19 and separately, effective prophylaxis. The molecule was similarly potent by intraperitoneal injection.

Neutralizing nanobodies (Nb) are of considerable interest as therapeutic agents for COVID-19 treatment. Here, the authors functionally and structurally characterize Nbs that bind with high affinity to the receptor binding domain of the SARS-CoV-2 spike protein and show that an engineered homotrimeric Nb prevents disease progression in a Syrian hamster model of COVID-19 when administered intranasally.

[end paper 17]

[begin paper 18]

PMCID = 9393763

Title = Antibody-mediated immunity to SARS-CoV-2 spike

Despite effective spike-based vaccines and monoclonal antibodies, the SARS-CoV-2 pandemic continues more than two and a half years post-onset. Relentless investigation has outlined a causative dynamic between host-derived antibodies and reciprocal viral subversion. Integration of this paradigm into the architecture of next generation antiviral strategies, predicated on a foundational understanding of the virology and immunology of SARS-CoV-2, will be critical for success. This review aims to serve as a primer on the immunity endowed by antibodies targeting SARS-CoV-2 spike protein through a structural perspective. We begin by introducing the structure and function of spike, polyclonal immunity to SARS-CoV-2 spike, and the emergence of major SARS-CoV-2 variants that evade immunity. The remainder of the article comprises an in-depth dissection of all major epitopes on SARS-CoV-2 spike in molecular detail, with emphasis on the origins, neutralizing potency, mechanisms of action, cross-reactivity, and variant resistance of representative monoclonal antibodies to each epitope.

[end paper 18]

[begin paper 19]

PMCID = 4118869

Title = The Breadth of Cross Sub-Type Neutralisation Activity of a Single Domain Antibody to Influenza Hemagglutinin Can Be Increased by Antibody Valency

The response to the 2009 A(H1N1) influenza pandemic has highlighted the need for additional strategies for intervention which preclude the prior availability of the influenza strain. Here, 18 single domain VHH antibodies against the 2009 A(H1N1) hemagglutinin (HA) have been isolated from a immune alpaca phage displayed library. These antibodies have been grouped as having either (i) non-neutralising, (ii) H1N1 restricted neutralising or (iii) broad cross-subtype neutralising activity. The ability to neutralise different viral subtypes, including highly pathogenic avian influenza (H5N1), correlated with the absence of hemagglutination inhibition activity, loss of binding to HA at acid pH and the absence of binding to the head domain containing the receptor binding site. This data supports their binding to epitopes in the HA stem region and a mechanism of action other than blocking viral attachment to cell surface receptors. After conversion of cross-neutralising antibodies R1a-B6 and R1a-A5 into a bivalent format, no significant enhancement in neutralisation activity was seen against A(H1N1) and A(H5N1) viruses. However, bivalent R1a-B6 showed an 18 fold enhancement in potency against A(H9N2) virus and, surprisingly, gained the ability to neutralise an A(H2N2) virus. This demonstrates that cross-neutralising antibodies, which make lower affinity interactions with the membrane proximal stem region of more divergent HA sub-types, can be optimised by bivalency so increasing their breadth of anti-viral activity. The broad neutralising activity and favourable characteristics, such as high stability, simple engineering into bivalent molecules and low cost production make these single domain antibodies attractive candidates for diagnostics and immunotherapy of pandemic influenza.

[end paper 19]

[begin paper 20]

PMCID = 8928829

Title = Nebulized delivery of a broadly neutralizing SARS-CoV-2 RBD-specific nanobody prevents clinical, virological, and pathological disease in a Syrian hamster model of COVID-19

There remains an unmet need for globally deployable, low-cost therapeutics for the ongoing severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) pandemic. Previously, we reported on the isolation and in vitro characterization of a potent single-domain nanobody, NIH-CoVnb-112, specific for the receptor-binding domain (RBD) of SARS-CoV-2. Here, we report on the molecular basis for the observed broad in vitro neutralization capability of NIH-CoVnb-112 against variant SARS-CoV-2 pseudoviruses. The structure of NIH-CoVnb-112 bound to SARS-CoV-2 RBD reveals a large contact surface area overlapping the angiotensin converting enzyme 2 (ACE2) binding site, which is largely unencumbered by the common RBD mutations. In an in vivo pilot study, we demonstrate effective reductions in weight loss, viral burden, and lung pathology in a Syrian hamster model of COVID-19 following nebulized delivery of NIH-CoVnb-112. These findings support the further development of NIH-CoVnb-112 as a potential adjunct preventative therapeutic for the treatment of SARS-CoV-2 infection.

Abbreviations: ACE2 – angiotensin converting enzyme 2BSA – buried surface areaCDR – complementary determining regionRBD – receptor binding domainRBM – receptor-binding motifSARS-CoV-2 - severe acute respiratory syndrome coronavirus 2

[end paper 20]

[begin paper 21]

PMCID = 9461429

Title = Arsenal of nanobodies shows broad-spectrum neutralization against SARS-CoV-2 variants of concern in vitro and in vivo in hamster models

Nanobodies offer several potential advantages over mAbs for the control of SARS-CoV-2. Their ability to access cryptic epitopes conserved across SARS-CoV-2 variants of concern (VoCs) and feasibility to engineer modular, multimeric designs, make these antibody fragments ideal candidates for developing broad-spectrum therapeutics against current and continually emerging SARS-CoV-2 VoCs. Here we describe a diverse collection of 37 anti-SARS-CoV-2 spike glycoprotein nanobodies extensively characterized as both monovalent and IgG Fc-fused bivalent modalities. The nanobodies were collectively shown to have high intrinsic affinity; high thermal, thermodynamic and aerosolization stability; broad subunit/domain specificity and cross-reactivity across existing VoCs; wide-ranging epitopic and mechanistic diversity and high and broad in vitro neutralization potencies. A select set of Fc-fused nanobodies showed high neutralization efficacies in hamster models of SARS-CoV-2 infection, reducing viral burden by up to six orders of magnitude to below detectable levels. In vivo protection was demonstrated with anti-RBD and previously unreported anti-NTD and anti-S2 nanobodies. This collection of nanobodies provides a potential therapeutic toolbox from which various cocktails or multi-paratopic formats could be built to combat multiple SARS-CoV-2 variants.

Isolation and extensive characterization of a collection of 37 anti-SARS-CoV-2 spike glycoprotein nanobodies show broad neutralization efficacies in vitro and in vivo in a hamster model of SARS-CoV-2 infection.

[end paper 21]

[begin paper 22]

PMCID = 11723632

Title = Discovery of Nanosota-EB1 and -EB2 as Novel Nanobody Inhibitors Against Ebola Virus Infection

The Ebola filovirus (EBOV) poses a serious threat to global health and national security. Nanobodies, a type of single-domain antibody, have demonstrated promising therapeutic potential. We identified two anti-EBOV nanobodies, Nanosota-EB1 and Nanosota-EB2, which specifically target the EBOV glycoprotein (GP). Cryo-EM and biochemical data revealed that Nanosota-EB1 binds to the glycan cap of GP1, preventing its protease cleavage, while Nanosota-EB2 binds to critical membrane-fusion elements in GP2, stabilizing it in the pre-fusion state. Nanosota-EB2 is a potent neutralizer of EBOV infection in vitro and offers excellent protection in a mouse model of EBOV challenge, while Nanosota-EB1 provides moderate neutralization and protection. Nanosota-EB1 and Nanosota-EB2 are the first nanobodies shown to inhibit authentic EBOV. Combined with our newly developed structure-guided in vitro evolution approach, they lay the foundation for nanobody-based therapies against EBOV and other viruses within the ebolavirus genus.

Ebola virus (EBOV) poses a serious global health threat due to its high fatality rate. Current antibody-based treatments face significant challenges, including reducing mortality, addressing viral persistence in some survivors, and ensuring affordability and accessibility. Furthermore, adapting existing human antibody therapies to combat EBOV-related viruses, such as Bundibugyo ebolavirus (BDBV) and Sudan ebolavirus (SUDV), remains difficult. Nanobodies, with their modular single-domain structure, present a promising solution to these challenges. In this study, we identified two anti-EBOV nanobodies, Nanosota-EB1 and Nanosota-EB2, which specifically target the EBOV GP protein. Cryo-EM and biochemical analyses revealed that Nanosota-EB1 binds to the glycan cap of GP1, preventing protease cleavage essential for viral infection. In contrast, Nanosota-EB2 binds to GP2, stabilizing it in the pre-fusion state and blocking its ability to mediate fusion between viral and host membranes. Nanosota-EB2 demonstrates strong neutralizing activity against EBOV infection in vitro and provides robust protection in a mouse model of EBOV challenge, while Nanosota-EB1 offers moderate neutralization and protection. As the first nanobodies shown to be effective against authentic EBOV, Nanosota-EB1 and Nanosota-EB2, together with our recently developed nanobody evolution strategy, establish a foundation for nanobody-based therapies targeting EBOV and related filoviruses.

[end paper 22]

[begin paper 23]

PMCID = 8608585

Title = Anti-SARS-CoV-1 and −2 nanobody engineering towards avidity-inspired therapeutics

In the past two decades, the emergence of coronavirus diseases has been dire distress on both continental and global fronts and has resulted in the search for potent treatment strategies. One crucial challenge in this search is the recurrent mutations in the causative virus spike protein, which lead to viral escape issues. Among the current promising therapeutic discoveries is the use of nanobodies and nanobody-like molecules. While these nanobodies have demonstrated high-affinity interaction with the virus, the unpredictable spike mutations have warranted the need for avidity-inspired therapeutics of potent inhibitors such as nanobodies. This article discusses novel approaches for the design of anti-SARS-CoV-1 and −2 nanobodies to facilitate advanced innovations in treatment technologies. It further discusses molecular interactions and suggests multivalent protein nanotechnology and chemistry approaches to translate mere molecular affinity into avidity.

[end paper 23]

[begin paper 24]

PMCID = 11758808

Title = Genetic Code Expansion: Recent Developments and Emerging Applications

The concept of genetic code expansion (GCE) has revolutionized the field of chemical and synthetic biology, enabling the site-specific incorporation of noncanonical amino acids (ncAAs) into proteins, thus opening new avenues in research and applications across biology and medicine. In this review, we cover the principles of GCE, including the optimization of the aminoacyl-tRNA synthetase (aaRS)/tRNA system and the advancements in translation system engineering. Notable developments include the refinement of aaRS/tRNA pairs, enhancements in screening methods, and the biosynthesis of noncanonical amino acids. The applications of GCE technology span from synthetic biology, where it facilitates gene expression regulation and protein engineering, to medicine, with promising approaches in drug development, vaccine production, and gene editing. The review concludes with a perspective on the future of GCE, underscoring its potential to further expand the toolkit of biology and medicine. Through this comprehensive review, we aim to provide a detailed overview of the current state of GCE technology, its challenges, opportunities, and the frontier it represents in the expansion of the genetic code for novel biological research and therapeutic applications.

[end paper 24]

[begin paper 25]

PMCID = 9792944

Title = Broadly neutralizing and protective nanobodies against SARS-CoV-2 Omicron subvariants BA.1, BA.2, and BA.4/5 and diverse sarbecoviruses

As SARS-CoV-2 Omicron and other variants of concern (VOCs) continue spreading worldwide, development of antibodies and vaccines to confer broad and protective activity is a global priority. Here, we report on the identification of a special group of nanobodies from immunized alpaca with potency against diverse VOCs including Omicron subvariants BA.1, BA.2 and BA.4/5, SARS-CoV-1, and major sarbecoviruses. Crystal structure analysis of one representative nanobody, 3-2A2-4, discovers a highly conserved epitope located between the cryptic and the outer face of the receptor binding domain (RBD), distinctive from the receptor ACE2 binding site. Cryo-EM and biochemical evaluation reveal that 3-2A2-4 interferes structural alteration of RBD required for ACE2 binding. Passive delivery of 3-2A2-4 protects K18-hACE2 mice from infection of authentic SARS-CoV-2 Delta and Omicron. Identification of these unique nanobodies will inform the development of next generation antibody therapies and design of pan-sarbecovirus vaccines.

The authors identify nanobodies from immunized alpaca with broadly neutralizing activity against SARS-CoV-1, SARS-CoV-2 variants, and major sarbecoviruses. One representative nanobody binds to a highly conserved epitope on RBD and protects K18-hACE2 mice from Omicron and Delta infection.

[end paper 25]

[begin paper 26]

PMCID = 7199733

Title = Structural Basis for Potent Neutralization of Betacoronaviruses by Single-Domain Camelid Antibodies

Coronaviruses make use of a large envelope protein called spike (S) to engage host cell receptors and catalyze membrane fusion. Because of the vital role that these S proteins play, they represent a vulnerable target for the development of therapeutics. Here, we describe the isolation of single-domain antibodies (VHHs) from a llama immunized with prefusion-stabilized coronavirus spikes. These VHHs neutralize MERS-CoV or SARS-CoV-1 S pseudotyped viruses, respectively. Crystal structures of these VHHs bound to their respective viral targets reveal two distinct epitopes, but both VHHs interfere with receptor binding. We also show cross-reactivity between the SARS-CoV-1 S-directed VHH and SARS-CoV-2 S and demonstrate that this cross-reactive VHH neutralizes SARS-CoV-2 S pseudotyped viruses as a bivalent human IgG Fc-fusion. These data provide a molecular basis for the neutralization of pathogenic betacoronaviruses by VHHs and suggest that these molecules may serve as useful therapeutics during coronavirus outbreaks.

VHHs isolated from a llama immunized with prefusion-stabilized coronavirus spikes

Structural characterization of VHHs reveals conserved mechanism of neutralization

SARS-CoV-1 S-directed VHH cross-reacts with SARS-CoV-2 S

Bivalent VHH neutralizes SARS-CoV-2 pseudoviruses

Using llamas immunized with prefusion-stabilized betacoronavirus spike proteins, Wrapp et al. identify neutralizing cross-reactive single-domain camelid antibodies, which may serve not only as useful reagents for researchers studying the viruses causing MERS, SARS, and COVID-19, but also potential therapeutic candidates. Crystal structures further reveal how these antibodies bind spike proteins to prevent virus entry into cells.

[end paper 26]

[begin paper 27]

PMCID = 11250487

Title = Unlocking precision gene therapy: harnessing AAV tropism with nanobody swapping at capsid hotspots

Adeno-associated virus (AAV) has been remarkably successful in the clinic, but its broad tropism is a practical limitation of precision gene therapy. A promising path to engineer AAV tropism is the addition of binding domains to the AAV capsid that recognize cell surface markers present on a targeted cell type. We have recently identified two previously unexplored capsid regions near the 2/5-fold wall and 5-fold pore of the AAV capsid that are amenable to insertion of larger protein domains, including nanobodies. Here, we demonstrate that these hotspots facilitate AAV tropism switching through simple nanobody replacement without extensive optimization in both VP1 and VP2. Our data suggest that engineering VP2 is the preferred path for maintaining both virus production yield and infectivity. We demonstrate highly specific targeting of human cancer cells expressing fibroblast activating protein (FAP). Furthermore, we found that the combination of FAP nanobody insertion plus ablation of the heparin binding domain can reduce off-target infection to a minimum, while maintaining a strong infection of FAP receptor-positive cells. Taken together, our study shows that nanobody swapping at multiple capsid locations is a viable strategy for nanobody-directed cell-specific AAV targeting.

[end paper 27]

[begin paper 28]

PMCID = 9043892

Title = Passive Immunotherapy Against SARS-CoV-2: From Plasma-Based Therapy to Single Potent Antibodies in the Race to Stay Ahead of the Variants

The COVID-19 pandemic is now approaching 2 years old, with more than 440 million people infected and nearly six million dead worldwide, making it the most significant pandemic since the 1918 influenza pandemic. The severity and significance of SARS-CoV-2 was recognized immediately upon discovery, leading to innumerable companies and institutes designing and generating vaccines and therapeutic antibodies literally as soon as recombinant SARS-CoV-2 spike protein sequence was available. Within months of the pandemic start, several antibodies had been generated, tested, and moved into clinical trials, including Eli Lilly’s bamlanivimab and etesevimab, Regeneron’s mixture of imdevimab and casirivimab, Vir’s sotrovimab, Celltrion’s regdanvimab, and Lilly’s bebtelovimab. These antibodies all have now received at least Emergency Use Authorizations (EUAs) and some have received full approval in select countries. To date, more than three dozen antibodies or antibody combinations have been forwarded into clinical trials. These antibodies to SARS-CoV-2 all target the receptor-binding domain (RBD), with some blocking the ability of the RBD to bind human ACE2, while others bind core regions of the RBD to modulate spike stability or ability to fuse to host cell membranes. While these antibodies were being discovered and developed, new variants of SARS-CoV-2 have cropped up in real time, altering the antibody landscape on a moving basis. Over the past year, the search has widened to find antibodies capable of neutralizing the wide array of variants that have arisen, including Alpha, Beta, Gamma, Delta, and Omicron. The recent rise and dominance of the Omicron family of variants, including the rather disparate BA.1 and BA.2 variants, demonstrate the need to continue to find new approaches to neutralize the rapidly evolving SARS-CoV-2 virus. This review highlights both convalescent plasma- and polyclonal antibody-based approaches as well as the top approximately 50 antibodies to SARS-CoV-2, their epitopes, their ability to bind to SARS-CoV-2 variants, and how they are delivered. New approaches to antibody constructs, including single domain antibodies, bispecific antibodies, IgA- and IgM-based antibodies, and modified ACE2-Fc fusion proteins, are also described. Finally, antibodies being developed for palliative care of COVID-19 disease, including the ramifications of cytokine release syndrome (CRS) and acute respiratory distress syndrome (ARDS), are described.

The online version contains supplementary material available at 10.1007/s40259-022-00529-7.

[end paper 28]

[begin paper 29]

PMCID = 8495465

Title = Arterial Thromboembolism



[end paper 29]

[begin paper 30]

PMCID = 9170159

Title = Dromedary camel nanobodies broadly neutralize SARS-CoV-2 variants

Due to their small size, nanobodies can recognize protein cavities that are not accessible to conventional antibodies. In this report, we built dromedary camel (Camelus dromedarius) VHH phage libraries for the isolation of high-affinity nanobodies that broadly neutralize SARS-CoV-2 variants. Cryo-EM complex structures reveal that one dromedary camel VHH nanobody (8A2) binds the S1 subunit of the viral spike protein, and the other (7A3) targets a deeply buried region that uniquely extends to the S2 subunit beyond the S1 subunit. These nanobodies can protect mice from the lethal challenge of variants B.1.351 or B.1.617.2, suggesting the therapeutic potential of these nanobodies against COVID-19. The dromedary camel VHH libraries could be helpful to isolate neutralizing nanobodies against future emerging viruses quickly.

The severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) spike is a trimer of S1/S2 heterodimers with three receptor-binding domains (RBDs) at the S1 subunit for human angiotensin-converting enzyme 2 (hACE2). Due to their small size, nanobodies can recognize protein cavities that are not accessible to conventional antibodies. To isolate high-affinity nanobodies, large libraries with great diversity are highly desirable. Dromedary camels (Camelus dromedarius) are natural reservoirs of coronaviruses like Middle East respiratory syndrome CoV (MERS-CoV) that are transmitted to humans. Here, we built large dromedary camel VHH phage libraries to isolate nanobodies that broadly neutralize SARS-CoV-2 variants. We isolated two VHH nanobodies, NCI-CoV-7A3 (7A3) and NCI-CoV-8A2 (8A2), which have a high affinity for the RBD via targeting nonoverlapping epitopes and show broad neutralization activity against SARS-CoV-2 and its emerging variants of concern. Cryoelectron microscopy (cryo-EM) complex structures revealed that 8A2 binds the RBD in its up mode with a long CDR3 loop directly involved in the ACE2 binding residues and that 7A3 targets a deeply buried region that uniquely extends from the S1 subunit to the apex of the S2 subunit regardless of the conformational state of the RBD. At a dose of ≥5 mg/kg, 7A3 efficiently protected transgenic mice expressing hACE2 from the lethal challenge of variants B.1.351 or B.1.617.2, suggesting its therapeutic use against COVID-19 variants. The dromedary camel VHH phage libraries could be helpful as a unique platform ready for quickly isolating potent nanobodies against future emerging viruses.

[end paper 30]

[begin paper 31]

PMCID = 8646660

Title = Uncovering a conserved vulnerability site in SARS‐CoV‐2 by a human antibody

An essential step for SARS‐CoV‐2 infection is the attachment to the host cell receptor by its Spike receptor‐binding domain (RBD). Most of the existing RBD‐targeting neutralizing antibodies block the receptor‐binding motif (RBM), a mutable region with the potential to generate neutralization escape mutants. Here, we isolated and structurally characterized a non‐RBM‐targeting monoclonal antibody (FD20) from convalescent patients. FD20 engages the RBD at an epitope distal to the RBM with a K D of 5.6 nM, neutralizes SARS‐CoV‐2 including the current Variants of Concern such as B.1.1.7, B.1.351, P.1, and B.1.617.2 (Delta), displays modest cross‐reactivity against SARS‐CoV, and reduces viral replication in hamsters. The epitope coincides with a predicted “ideal” vulnerability site with high functional and structural constraints. Mutation of the residues of the conserved epitope variably affects FD20‐binding but confers little or no resistance to neutralization. Finally, in vitro mode‐of‐action characterization and negative‐stain electron microscopy suggest a neutralization mechanism by which FD20 destructs the Spike. Our results reveal a conserved vulnerability site in the SARS‐CoV‐2 Spike for the development of potential antiviral drugs.

A monoclonal antibody (FD20) from convalescent COVID‐19 patients has been isolated and structurally and biologically characterized. Various SARS‐CoV‐2 strains, including the Alpha, Beta, Gamma, and Delta variants, and naturally occurring epitope mutants, can be neutralized by FD20 with similar potency. 

[end paper 31]

[begin paper 32]

PMCID = 9288967

Title = Accelerating PERx reaction enables covalent nanobodies for potent neutralization of SARS-CoV-2 and variants

The long-lasting COVID-19 pandemic and increasing SARS-CoV-2 variants demand effective drugs for prophylactics and treatment. Protein-based biologics offer high specificity, yet their noncovalent interactions often lead to drug dissociation and incomplete inhibition. Here, we have developed covalent nanobodies capable of binding with SARS-CoV-2 irreversibly via a proximity-enabled reactive therapeutic (PERx) mechanism. A latent bioreactive amino acid (FFY) was designed and genetically encoded into nanobodies to accelerate the PERx reaction rate. Compared with the noncovalent wild-type nanobody, the FFY-incorporated covalent nanobodies neutralized both wild-type SARS-CoV-2 and its Alpha, Delta, Epsilon, Lambda, and Omicron variants with drastically higher potency. This PERx-enabled covalent-nanobody strategy and the related insights into increased potency can be valuable to developing effective therapeutics for various viral infections.

The long-lasting COVID-19 pandemic and increasing SARS-CoV-2 variants demand effective therapeutic strategies. Herein, we have engineered unique unnatural amino acids into proteins and developed covalent nanobodies capable of irreversibly binding with the viral spike protein. These covalent nanobodies are able to neutralize both SARS-CoV-2 and its different variants with drastic higher potency than traditional nanobodies, affording a potential prophylactic and medication for COVID-19. In addition, our strategy could provide an original general route to developing effective therapeutics for various other viral infections, such as influenza, hepatitis, AIDS, and anthrax.

Traditional nanobodies interact with their targets reversibly, leading to dissociation and incomplete inhibition. Wang and co-workers have designed a latent bioreactive amino acid (FFY) and genetically encoded it into nanobodies. Upon nanobody binding to the viral spike protein, FFY covalently reacts with spike via a proximity-enabled reactive therapeutic (PERx) mechanism, enabling the nanobody to bind to SARS-CoV-2 irreversibly and effectively block viral infection. The FFY-incorporated covalent nanobodies neutralized both wild-type SARS-CoV-2 and its various variants with drastically higher potency than the noncovalent wild-type nanobodies.

[end paper 32]

[begin paper 33]

PMCID = 9834170

Title = Two pan-SARS-CoV-2 nanobodies and their multivalent derivatives effectively prevent Omicron infections in mice

With the widespread vaccinations against coronavirus disease 2019 (COVID-19), we are witnessing gradually waning neutralizing antibodies and increasing cases of breakthrough infections, necessitating the development of drugs aside from vaccines, particularly ones that can be administered outside of hospitals. Here, we present two cross-reactive nanobodies (R14 and S43) and their multivalent derivatives, including decameric ones (fused to the immunoglobulin M [IgM] Fc) that maintain potent neutralizing activity against severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) after aerosolization and display not only pan-SARS-CoV-2 but also varied pan-sarbecovirus activities. Through respiratory administration to mice, monovalent and decameric R14 significantly reduce the lung viral RNAs at low dose and display potent pre- and post-exposure protection. Furthermore, structural studies reveal the neutralizing mechanisms of R14 and S43 and the multiple inhibition effects that the multivalent derivatives exert. Our work demonstrates promising convenient drug candidates via respiratory administration against SARS-CoV-2 infection, which can contribute to containing the COVID-19 pandemic.

Nanobodies R14 and S43 display not just pan-SARS-CoV-2 but varied pan-sarbecovirus activities

The multivalent nanobodies show enhanced neutralization, fitting the avidity effect

Intranasally administered MR14 potently prevents Omicron infections in mice

Structural studies reveal the neutralizing mechanisms of R14 and S43 as well as MR14

Liu et al. isolate two nanobodies (R14 and S43) and construct their multivalent derivatives with increased potencies and breadth against SARS-CoV-2 variants. Particularly, decameric R14, maintaining similar activities after aerosolization, efficiently prevents BA.2 infections in both pre- and post-exposure prophylactic settings in vivo and deserves further development.

[end paper 33]

[begin paper 34]

PMCID = 9132424

Title = Deep mutational engineering of broadly-neutralizing nanobodies accommodating SARS-CoV-1 and 2 antigenic drift

Here, we report the molecular engineering of nanobodies that bind with picomolar affinity to both SARS-CoV-1 and SARS-CoV-2 receptor-binding domains (RBD) and are highly neutralizing. We applied deep mutational engineering to VHH72, a nanobody initially specific for SARS-CoV-1 RBD with little cross-reactivity to SARS-CoV-2 antigen. We first identified all the individual VHH substitutions that increase binding to SARS-CoV-2 RBD and then screened highly focused combinatorial libraries to isolate engineered nanobodies with improved properties. The corresponding VHH-Fc molecules show high affinities for SARS-CoV-2 antigens from various emerging variants and SARS-CoV-1, block the interaction between ACE2 and RBD, and neutralize the virus with high efficiency. Its rare specificity across sarbecovirus relies on its peculiar epitope outside the immunodominant regions. The engineered nanobodies share a common motif of three amino acids, which contribute to the broad specificity of recognition. Our results show that deep mutational engineering is a very powerful method, especially to rapidly adapt existing antibodies to new variants of pathogens.

[end paper 34]

[begin paper 35]

PMCID = 9734137

Title = A bispecific nanobody dimer broadly neutralizes SARS-CoV-1 & 2 variants of concern and offers substantial protection against Omicron via low-dose intranasal administration

Current SARS-CoV-2 Omicron subvariants impose a heavy burden on global health systems by evading immunity from most developed neutralizing antibodies and vaccines. Here, we identified a nanobody (aSA3) that strongly cross-reacts with the receptor binding domain (RBD) of both SARS-CoV-1 and wild-type (WT) SARS-CoV-2. The dimeric construct of aSA3 (aSA3-Fc) tightly binds and potently neutralizes both SARS-CoV-1 and WT SARS-CoV-2. Based on X-ray crystallography, we engineered a bispecific nanobody dimer (2-3-Fc) by fusing aSA3-Fc to aRBD-2, a previously identified broad-spectrum nanobody targeting an RBD epitope distinct from aSA3. 2-3-Fc exhibits single-digit ng/mL neutralizing potency against all major variants of concerns including BA.5. In hamsters, a single systemic dose of 2-3-Fc at 10 mg/kg conferred substantial efficacy against Omicron infection. More importantly, even at three low doses of 0.5 mg/kg, 2-3-Fc prophylactically administered through the intranasal route drastically reduced viral RNA loads and completely eliminated infectious Omicron particles in the trachea and lungs. Finally, we discovered that 2(Y29G)-3-Fc containing a Y29G substitution in aRBD-2 showed better activity than 2-3-Fc in neutralizing BA.2.75, a recent Omicron subvariant that emerged in India. This study expands the arsenal against SARS-CoV-1, provides potential therapeutic and prophylactic candidates that fully cover major SARS-CoV-2 variants, and may offer a simple preventive approach against Omicron and its subvariants.

[end paper 35]

[begin paper 36]

PMCID = 7494754

Title = Viral Emerging Diseases: Challenges in Developing Vaccination Strategies

In the last decades, a number of infectious viruses have emerged from wildlife or re-emerged, generating serious threats to the global health and to the economy worldwide. Ebola and Marburg hemorrhagic fevers, Lassa fever, Dengue fever, Yellow fever, West Nile fever, Zika, and Chikungunya vector-borne diseases, Swine flu, Severe acute respiratory syndrome (SARS), Middle East respiratory syndrome (MERS), and the recent Coronavirus disease 2019 (COVID-19) are examples of zoonoses that have spread throughout the globe with such a significant impact on public health that the scientific community has been called for a rapid intervention in preventing and treating emerging infections. Vaccination is probably the most effective tool in helping the immune system to activate protective responses against pathogens, reducing morbidity and mortality, as proven by historical records. Under health emergency conditions, new and alternative approaches in vaccine design and development are imperative for a rapid and massive vaccination coverage, to manage a disease outbreak and curtail the epidemic spread. This review gives an update on the current vaccination strategies for some of the emerging/re-emerging viruses, and discusses challenges and hurdles to overcome for developing efficacious vaccines against future pathogens.

[end paper 36]

[begin paper 37]

PMCID = 7932109

Title = Structure-guided multivalent nanobodies block SARS-CoV-2 infection and suppress mutational escape

Monoclonal antibodies are an important weapon in the battle against COVID-19. However, these large proteins are difficult to produce in the needed quantities and at low cost. Attention has turned to nanobodies, which are aptly named, single-domain antibodies that are easier to produce and have the potential to be administered by inhalation. Koenig et al. describe four nanobodies that bind to the severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) spike protein and prevent infection of cells (see the Perspective by Saelens and Schepens). Structures show that the nanobodies target two distinct epitopes on the SARS-CoV-2 spike protein. Multivalent nanobodies neutralize virus much more potently than single nanobodies, and multivalent nanobodies that bind two epitopes prevent the emergence of viral escape mutants.

Science, this issue p. eabe6230; see also p. 

SARS-CoV-2–neutralizing nanobodies were combined to design potent multivalent nanobodies.

The global scale and rapid spread of severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) pose unprecedented challenges to society, health care systems, and science. In addition to effective and safe vaccines, passive immunization by antibody-related molecules offers an opportunity to harness the vertebrate immune system to fight viral infections in high-risk patients. Variable domains of heavy-chain–only antibodies (VHHs), also known as nanobodies, are suitable lead molecules in such efforts, as they are small, extremely stable, easy to engineer, and economic to produce in simple expression systems.

We engineered improved multivalent nanobodies neutralizing SARS-CoV-2 on the basis of two principles: (i) detailed structural information of their epitopes and binding modes to the viral spike protein and (ii) mechanistic insights into viral fusion with cellular membranes catalyzed by the spike.

Nanobodies specific for the receptor binding domain (RBD) of SARS-CoV-2 spike were identified by phage display using nanobody libraries from an alpaca and a llama immunized with the RBD and inactivated virus. Four of the resulting nanobodies—VHHs E, U, V, and W—potently neutralize SARS-CoV-2 and SARS-CoV-2–pseudotyped vesicular stomatitis virus. X-ray crystallography revealed that the nanobodies bind to two distinct epitopes on the RBD, interfaces “E” and “UVW,” which can be synergistically targeted by combinations of nanobodies to inhibit infection. Cryo–electron microscopy (cryo-EM) of trimeric spike in complex with VHH E and VHH V revealed that VHH E stabilizes a conformation of the spike with all three RBDs in the “up” conformation (3-up), a state that is typically associated with activation by receptor binding. In line with this observation, we found that VHH E triggers the fusion activity of spike in the absence of the cognate receptor ACE2. VHH V, by contrast, stabilizes spike in a 2-up conformation and does not induce fusion. On the basis of the structural information, we designed bi- and trivalent nanobodies with improved neutralizing properties. VHH EEE most potently inhibited infection, did not activate fusion, and likely inactivated virions by outcompeting interaction of the virus with its receptor. Yet evolution experiments revealed emergence of escape mutants in the spike with single–amino acid changes that were completely insensitive to inhibition by VHH EEE. VHH VE also neutralized more efficiently than VHH E or VHH V alone; stabilized the 3-up conformation of spike, as determined by cryo-EM; and more strongly induced the spike fusogenic activity. We conclude that the premature activation of the fusion machinery on virions was an unexpected mechanism of neutralization, as enhanced neutralization could not be attributed simply to better blocking of virus-receptor interactions. Activation of spike in the absence of target membranes likely induces irreversible conformational changes to assume the energetically favorable postfusion conformation without catalyzing fusion per se. Simultaneous targeting of two independent epitopes by VHH VE largely prevented the emergence of resistant escape mutants in evolution experiments.

Our results demonstrate the strength of the modular combination of nanobodies for neutralization. Premature activation of spike by nanobodies reveals an unusual mode of neutralization and yields insights into the mechanism of fusion.

On virions, SARS-CoV-2 spike trimers are mostly in an inactive configuration with all RBDs in the down conformation (left). Binding of bivalent nanobody VE stabilizes the spike in an active conformation with all RBDs up (middle), triggering premature induction of the postfusion conformation, which irreversibly inactivates the spike protein (right).

The pandemic caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) continues to spread, with devastating consequences. For passive immunization efforts, nanobodies have size and cost advantages over conventional antibodies. In this study, we generated four neutralizing nanobodies that target the receptor binding domain of the SARS-CoV-2 spike protein. We used x-ray crystallography and cryo–electron microscopy to define two distinct binding epitopes. On the basis of these structures, we engineered multivalent nanobodies with more than 100 times the neutralizing activity of monovalent nanobodies. Biparatopic nanobody fusions suppressed the emergence of escape mutants. Several nanobody constructs neutralized through receptor binding competition, whereas other monovalent and biparatopic nanobodies triggered aberrant activation of the spike fusion machinery. These premature conformational changes in the spike protein forestalled productive fusion and rendered the virions noninfectious.

[end paper 37]

[begin paper 38]

PMCID = 9126950

Title = SARS-CoV-2 detection using a nanobody-functionalized voltammetric device

An ongoing need during the COVID-19 pandemic has been the requirement for accurate and efficient point-of-care testing platforms to distinguish infected from non-infected people, and to differentiate SARS-CoV-2 infections from other viruses. Electrochemical platforms can detect the virus via its envelope spike protein by recording changes in voltammetric signals between samples. However, this remains challenging due to the limited sensitivity of these sensing platforms.

Here, we report on a nanobody-functionalized electrochemical platform for the rapid detection of whole SARS-CoV-2 viral particles in complex media such as saliva and nasopharyngeal swab samples. The sensor relies on the functionalization of gold electrode surface with highly-oriented Llama nanobodies specific to the spike protein receptor binding domain (RBD). The device provides results in 10 min of exposure to 200 µL of unprocessed samples with high specificity to SARS-CoV-2 viral particles in human saliva and nasopharyngeal swab samples.

The developed sensor could discriminate between different human coronavirus strains and other respiratory viruses, with 90% positive and 90% negative percentage agreement on 80 clinical samples, as compared to RT-qPCR.

We believe this diagnostic concept, also validated for RBD mutants and successfully tested on Delta variant samples, to be a powerful tool to detect patients’ infection status, easily extendable to other viruses and capable of overcoming sensing-related mutation effects.

Our ability to control SARS-CoV-2, the virus that causes COVID-19, relies on access to rapid and sensitive technologies that can detect the virus, and differentiate SARS-CoV-2 infections from infections with other viruses. One potential approach to detect SARS-CoV-2 is to use sensors comprising specific surfaces and chemicals that can detect changes in the properties of samples containing the virus. Here, we develop one of these sensors and test it in samples from people with COVID-19 or other viral infections, or people without. We show that our portable device can quickly and accurately detect SARS-CoV-2 and can also detect different variants. We also show that signals from our device are specific for SARS-CoV-2. Our device may provide another tool for COVID-19 diagnosis, with benefits being portability and speed compared with current gold-standard approaches.

Pagneux et al. developed a nanobody-functionalized electrochemical platform with smartphone readout to detect SARS-CoV-2 in clinical samples. They demonstrated that their device can rapidly and accurately detect SARS-CoV-2 in saliva and nasopharyngeal swab samples and discriminate between this virus and other respiratory viruses.

[end paper 38]

[begin paper 39]

PMCID = 6344588

Title = An improved yeast surface display platform for the screening of nanobody immune libraries

Fusions to the C-terminal end of the Aga2p mating adhesion of Saccharomyces cerevisiae have been used in many studies for the selection of affinity reagents by yeast display followed by flow cytometric analysis. Here we present an improved yeast display system for the screening of Nanobody immune libraries where we fused the Nanobody to the N-terminal end of Aga2p to avoid steric hindrance between the fused Nanobody and the antigen. Moreover, the display level of a cloned Nanobody on the surface of an individual yeast cell can be monitored through a covalent fluorophore that is attached in a single enzymatic step to an orthogonal acyl carrier protein (ACP). Additionally, the displayed Nanobody can be easily released from the yeast surface and immobilised on solid surfaces for rapid analysis. To prove the generic nature of this novel Nanobody discovery platform, we conveniently selected Nanobodies against three different antigens, including two membrane proteins.

[end paper 39]

[begin paper 40]

PMCID = 8001348

Title = Broad Reactivity Single Domain Antibodies against Influenza Virus and Their Applications to Vaccine Potency Testing and Immunotherapy

The antigenic variability of influenza presents many challenges to the development of vaccines and immunotherapeutics. However, it is apparent that there are epitopes on the virus that have evolved to remain largely constant due to their functional importance. These more conserved regions are often hidden and difficult to access by the human immune system but recent efforts have shown that these may be the Achilles heel of the virus through development and delivery of appropriate biological drugs. Amongst these, single domain antibodies (sdAbs) are equipped to target these vulnerabilities of the influenza virus due to their preference for concave epitopes on protein surfaces, their small size, flexible reformatting and high stability. Single domain antibodies are well placed to provide a new generation of robust analytical reagents and therapeutics to support the constant efforts to keep influenza in check.

[end paper 40]

[begin paper 41]

PMCID = 9201380

Title = Structural Characterization of a Neutralizing Nanobody With Broad Activity Against SARS-CoV-2 Variants

SARS-CoV-2 and its variants, such as the Omicron continue to threaten public health. The virus recognizes the host cell by attaching its Spike (S) receptor-binding domain (RBD) to the host receptor, ACE2. Therefore, RBD is a primary target for neutralizing antibodies and vaccines. Here, we report the isolation and biological and structural characterization of a single-chain antibody (nanobody) from RBD-immunized alpaca. The nanobody, named DL28, binds to RBD tightly with a KD of 1.56 nM and neutralizes the original SARS-CoV-2 strain with an IC50 of 0.41 μg mL−1. Neutralization assays with a panel of variants of concern (VOCs) reveal its wide-spectrum activity with IC50 values ranging from 0.35 to 1.66 μg mL−1 for the Alpha/Beta/Gamma/Delta and an IC50 of 0.66 μg mL−1 for the currently prevalent Omicron. Competition binding assays show that DL28 blocks ACE2-binding. However, structural characterizations and mutagenesis suggest that unlike most antibodies, the blockage by DL28 does not involve direct competition or steric hindrance. Rather, DL28 may use a “conformation competition” mechanism where it excludes ACE2 by keeping an RBD loop in a conformation incompatible with ACE2-binding.

[end paper 41]

[begin paper 42]

PMCID = 9343077

Title = Evaluation of SARS-CoV-2-Neutralizing Nanobody Using Virus Receptor Binding Domain-Administered Model Mice

Due to the rapid spread of coronavirus disease 2019 (COVID-19), there is an urgent requirement for the development of additional diagnostic tools for further analysis of the disease. The isolated nanobody Nb11-59 binds to the severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) receptor-binding domain (RBD) with high affinity to neutralize the virus and block the angiotensin-converting enzyme 2- (ACE2-) RBD interaction. Here, we introduce a novel nanobody-based radiotracer named 68Ga-Nb1159. The radiotracer retained high affinity for the RBD and showed reliable radiochemical characteristics both in vitro and in vivo. Preclinical positron emission tomography (PET) studies of 68Ga-Nb1159 in mice revealed its rapid clearance from circulation and robust uptake into the renal and urinary systems. Fortunately, 68Ga-Nb1159 could specifically reveal the distribution of the RBD in mice. This study also helped to evaluate the pharmacodynamic effects of the neutralizing nanobody. Moreover, 68Ga-Nb1159 may be a promising tool to explore the distribution of the RBD and improve the understanding of the virus. In particular, this study identified a novel molecular radioagent and established a reliable evaluation method for specifically investigating the RBD through noninvasive and visual PET technology.

[end paper 42]

[begin paper 43]

PMCID = 9429973

Title = Abstract Book for the 27th Congress of the European Hematology Association



[end paper 43]

[begin paper 44]

PMCID = 11793827

Title = Nanobody screening and machine learning guided identification of cross-variant anti-SARS-CoV-2 neutralizing heavy-chain only antibodies

Severe Acute Respiratory Syndrome Coronavirus 2 (SARS-CoV-2) continues to persist, demonstrating the risks posed by emerging infectious diseases to national security, public health, and the economy. Development of new vaccines and antibodies for emerging viral threats requires substantial resources and time, and traditional development platforms for vaccines and antibodies are often too slow to combat continuously evolving immunological escape variants, reducing their efficacy over time. Previously, we designed a next-generation synthetic humanized nanobody (Nb) phage display library and demonstrated that this library could be used to rapidly identify highly specific and potent neutralizing heavy chain-only antibodies (HCAbs) with prophylactic and therapeutic efficacy in vivo against the original SARS-CoV-2. In this study, we used a combination of high throughput screening and machine learning (ML) models to identify HCAbs with potent efficacy against SARS-CoV-2 viral variants of interest (VOIs) and concern (VOCs). To start, we screened our highly diverse Nb phage display library against several pre-Omicron VOI and VOC receptor binding domains (RBDs) to identify panels of cross-reactive HCAbs. Using HCAb affinity for SARS-CoV-2 VOI and VOCs (pre-Omicron variants) and model features from other published data, we were able to develop a ML model that successfully identified HCAbs with efficacy against Omicron variants, independent of our experimental biopanning workflow. This biopanning informed ML approach reduced the experimental screening burden by 78% to 90% for the Omicron BA.5 and Omicron BA.1 variants, respectively. The combined approach can be applied to other emerging viruses with pandemic potential to rapidly identify effective therapeutic antibodies against emerging variants.

We leveraged a robust experimental pipeline for heavy chain-only (HCAb) library screening to identify 59 potent HCAbs that can cross-neutralize different SARS-CoV-2 variants. Several of these HCAbs with efficacy against different variants also bind to different SARS-CoV-2 epitopes, suggesting they could be used in antibody cocktails or be engineered as bispecific antibodies that are cross-variant and resistant to viral escape. Using existing published data and data generated from our library of HCAbs against diverse pre-Omicron SARS-CoV-2 variants, we developed an ML model to rapidly predict which HCAbs would be effective against Omicron BA.1 and Omicron BA.5. Using this ML model three HCAbs with efficacy against Omicron BA.1 and BA.5 were identified without requiring additional biopanning. This integrated computational and experimental pipeline can be leveraged for future outbreaks to rapidly identify cross-variant countermeasures that are effective against potential emerging variants.

[end paper 44]

[begin paper 45]

PMCID = 9860306

Title = Synthetic nanobodies as tools to distinguish IgG Fc glycoforms

On proteins, glycans are attached through stepwise construction of linear or branched structures, ultimately leading to families of highly related but nonequivalent glycoproteins known as glycoforms. On immunoglobulin G (IgG), differences in glycosylation of the fragment crystallizable domain (Fc) modulate its ability to signal to leukocytes to execute critical antibody effector functions. Despite burgeoning interest in understanding the complexities of IgG Fc glycoforms, there is an evident scarcity of tools available to study them. Here, we identify nanobodies which we use to study and manipulate specific IgG glycoforms in vitro and in vivo.

Protein glycosylation is a crucial mediator of biological functions and is tightly regulated in health and disease. However, interrogating complex protein glycoforms is challenging, as current lectin tools are limited by cross-reactivity while mass spectrometry typically requires biochemical purification and isolation of the target protein. Here, we describe a method to identify and characterize a class of nanobodies that can distinguish glycoforms without reactivity to off-target glycoproteins or glycans. We apply this technology to immunoglobulin G (IgG) Fc glycoforms and define nanobodies that specifically recognize either IgG lacking its core-fucose or IgG bearing terminal sialic acid residues. By adapting these tools to standard biochemical methods, we can clinically stratify dengue virus and SARS-CoV-2 infected individuals based on their IgG glycan profile, selectively disrupt IgG–Fcγ receptor binding both in vitro and in vivo, and interrogate the B cell receptor (BCR) glycan structure on living cells. Ultimately, we provide a strategy for the development of reagents to identify and manipulate IgG Fc glycoforms.

[end paper 45]

[begin paper 46]

PMCID = 7755911

Title = High affinity nanobodies block SARS-CoV-2 spike receptor binding domain interaction with human angiotensin converting enzyme

There are currently few approved effective treatments for SARS-CoV-2, the virus responsible for the COVID-19 pandemic. Nanobodies are 12–15 kDa single-domain antibody fragments that can be delivered by inhalation and are amenable to relatively inexpensive large scale production compared to other biologicals. We have isolated nanobodies that bind to the SARS-CoV-2 spike protein receptor binding domain and block spike protein interaction with the angiotensin converting enzyme 2 (ACE2) with 1–5 nM affinity. The lead nanobody candidate, NIH-CoVnb-112, blocks SARS-CoV-2 spike pseudotyped lentivirus infection of HEK293 cells expressing human ACE2 with an EC50 of 0.3 µg/mL. NIH-CoVnb-112 retains structural integrity and potency after nebulization. Furthermore, NIH-CoVnb-112 blocks interaction between ACE2 and several high affinity variant forms of the spike protein. These nanobodies and their derivatives have therapeutic, preventative, and diagnostic potential.

[end paper 46]

[begin paper 47]

PMCID = 9797221

Title = Therapeutic applications of nanobodies against SARS-CoV-2 and other viral infections: Current update

In the last two years, the world encountered the SARS-CoV-2 virus, which is still dominating the population due to the absence of a viable treatment. To eradicate the global pandemic, scientists, doctors, and researchers took an exceptionally significant initiative towards the development of effective therapeutics to save many lifes. This review discusses about the single-domain antibodies (sdAbs), also called nanobodies, their structure, and their types against the infections of dreadful SARS-CoV-2 virus. A precise description highlights the nanobodies and their therapeutic application against the other selected viruses. It aims to focus on the extraordinary features of these antibodies compared to the conventional therapeutics like mAbs, convalescent plasma therapy, and vaccines. The stable structure of these nanobodies along with the suitable mechanism of action also confers greater resistance to the evolving variants with numerous mutations. The nanobodies developed against SARS-CoV-2 and its mutant variants have shown the greater neutralization potential than the primitive ones. Engineering of these specialized antibodies by modern biotechnological approaches will surely be more beneficial in treating this COVID-19 pandemic along with certain other viral infections.

[end paper 47]

[begin paper 48]

PMCID = 9585390

Title = Abstract



[end paper 48]

[begin paper 49]

PMCID = 8982573

Title = Biparatopic sybodies neutralize SARS‐CoV‐2 variants of concern and mitigate drug resistance

The ongoing COVID‐19 pandemic represents an unprecedented global health crisis. Here, we report the identification of a synthetic nanobody (sybody) pair, Sb#15 and Sb#68, that can bind simultaneously to the SARS‐CoV‐2 spike RBD and efficiently neutralize pseudotyped and live viruses by interfering with ACE2 interaction. Cryo‐EM confirms that Sb#15 and Sb#68 engage two spatially discrete epitopes, influencing rational design of bispecific and tri‐bispecific fusion constructs that exhibit up to 100‐ and 1,000‐fold increase in neutralization potency, respectively. Cryo‐EM of the sybody‐spike complex additionally reveals a novel up‐out RBD conformation. While resistant viruses emerge rapidly in the presence of single binders, no escape variants are observed in the presence of the bispecific sybody. The multivalent bispecific constructs further increase the neutralization potency against globally circulating SARS‐CoV‐2 variants of concern. Our study illustrates the power of multivalency and biparatopic nanobody fusions for the potential development of therapeutic strategies that mitigate the emergence of new SARS‐CoV‐2 escape mutants.

Sybodies Sb#15 and Sb#68 inhibit SARS‐CoV‐2 infectivity by targeting non‐overlapping epitopes on the spike glycoprotein. Covalent sybody fusion and valency engineering enhances neutralization potency against variants and impedes emergence of escape mutants.  

[end paper 49]

[begin paper 50]

PMCID = 10487883

Title = NANOBODY® Molecule, a Giga Medical Tool in Nanodimensions

Although antibodies remain the most widely used tool for biomedical research, antibody technology is not flawless. Innovative alternatives, such as Nanobody® molecules, were developed to address the shortcomings of conventional antibodies. Nanobody® molecules are antigen-binding variable-domain fragments derived from the heavy-chain-only antibodies of camelids (VHH) and combine the advantageous properties of small molecules and monoclonal antibodies. Nanobody® molecules present a small size (~15 kDa, 4 nm long and 2.5 nm wide), high solubility, stability, specificity, and affinity, ease of cloning, and thermal and chemical resistance. Recombinant production in microorganisms is cost-effective, and VHH are also building blocks for multidomain constructs. These unique features led to numerous applications in fundamental research, diagnostics, and therapy. Nanobody® molecules are employed as biomarker probes and, when fused to radioisotopes or fluorophores, represent ideal non-invasive in vivo imaging agents. They can be used as neutralizing agents, receptor-ligand antagonists, or in targeted vehicle-based drug therapy. As early as 2018, the first Nanobody®, Cablivi (caplacizumab), a single-domain antibody (sdAb) drug developed by French pharmaceutical giant Sanofi for the treatment of adult patients with acquired thrombocytopenic purpura (aTTP), was launched. Nanobody® compounds are ideal tools for further development in clinics for diagnostic and therapeutic purposes.

[end paper 50]

[begin paper 51]

PMCID = 8863326

Title = A potent alpaca-derived nanobody that neutralizes SARS-CoV-2 variants

The spike glycoprotein of SARS-CoV-2 engages with human ACE 2 to facilitate infection. Here, we describe an alpaca-derived heavy chain antibody fragment (VHH), saRBD-1, that disrupts this interaction by competitively binding to the spike protein receptor-binding domain. We further generated an engineered bivalent nanobody construct engineered by a flexible linker and a dimeric Fc conjugated nanobody construct. Both multivalent nanobodies blocked infection at picomolar concentrations and demonstrated no loss of potency against emerging variants of concern including Alpha (B.1.1.7), Beta (B.1.351), Gamma (P.1), Epsilon (B.1.427/429), and Delta (B.1.617.2). saRBD-1 tolerates elevated temperature, freeze-drying, and nebulization, making it an excellent candidate for further development into a therapeutic approach for COVID-19.

SARS-CoV-2 variants effectively neutralized by saRBD-1 VHH with picomolar affinity

saRBD-1 neutralization increases when expressed as a bivalent or Fc construct

saRBD-1 binds SARS-CoV-2 RBD as a likely class 1 neutralizing antibody

saRBD-1 retains binding, neutralization after heat and nebulization treatments

Applied microbiology; Bioengineering; Nanotechnology

[end paper 51]

[begin paper 52]

PMCID = 10275483

Title = Emerging Landscape of Nanobodies and Their Neutralizing Applications against SARS-CoV-2 Virus

The new severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) that causes the coronavirus disease 2019 (COVID-19) has significantly altered people’s way of life. Despite widespread knowledge of vaccination, mask use, and avoidance of close contact, COVID-19 is still spreading around the world. Numerous research teams are examining the SARS-CoV-2 infection process to discover strategies to identify, prevent, and treat COVID-19 to limit the spread of this chronic coronavirus illness and restore lives to normalcy. Nanobodies have advantages over polyclonal and monoclonal antibodies (Ab) and Ab fragments, including reduced size, high stability, simplicity in manufacture, compatibility with genetic engineering methods, and lack of solubility and aggregation issues. Recent studies have shown that nanobodies that target the SARS-CoV-2 receptor-binding domain and disrupt ACE2 interactions are helpful in the prevention and treatment of SARS-CoV-2-infected animal models, despite the lack of evidence in human patients. The creation and evaluation of nanobodies, as well as their diagnostic and therapeutic applications against COVID-19, are discussed in this paper.

[end paper 52]

[begin paper 53]

PMCID = 8575140

Title = Camel nanobodies broadly neutralize SARS-CoV-2 variants

With the emergence of SARS-CoV-2 variants, there is urgent need to develop broadly neutralizing antibodies. Here, we isolate two VHH nanobodies (7A3 and 8A2) from dromedary camels by phage display, which have high affinity for the receptor-binding domain (RBD) and broad neutralization activities against SARS-CoV-2 and its emerging variants. Cryo-EM complex structures reveal that 8A2 binds the RBD in its up mode and 7A3 inhibits receptor binding by uniquely targeting a highly conserved and deeply buried site in the spike regardless of the RBD conformational state. 7A3 at a dose of ≥5 mg/kg efficiently protects K18-hACE2 transgenic mice from the lethal challenge of B.1.351 or B.1.617.2, suggesting that the nanobody has promising therapeutic potentials to curb the COVID-19 surge with emerging SARS-CoV-2 variants.

Dromedary camel (Camelus dromedarius) VHH phage libraries were built for isolation of the nanobodies that broadly neutralize SARS-CoV-2 variants.

[end paper 53]

[begin paper 54]

PMCID = 9784173

Title = Anti-SOD1 Nanobodies That Stabilize Misfolded SOD1 Proteins Also Promote Neurite Outgrowth in Mutant SOD1 Human Neurons

ALS-linked mutations induce aberrant conformations within the SOD1 protein that are thought to underlie the pathogenic mechanism of SOD1-mediated ALS. Although clinical trials are underway for gene silencing of SOD1, these approaches reduce both wild-type and mutated forms of SOD1. Here, we sought to develop anti-SOD1 nanobodies with selectivity for mutant and misfolded forms of human SOD1 over wild-type SOD1. Characterization of two anti-SOD1 nanobodies revealed that these biologics stabilize mutant SOD1 in vitro. Further, SOD1 expression levels were enhanced and the physiological subcellular localization of mutant SOD1 was restored upon co-expression of anti-SOD1 nanobodies in immortalized cells. In human motor neurons harboring the SOD1 A4V mutation, anti-SOD1 nanobody expression promoted neurite outgrowth, demonstrating a protective effect of anti-SOD1 nanobodies in otherwise unhealthy cells. In vitro assays revealed that an anti-SOD1 nanobody exhibited selectivity for human mutant SOD1 over endogenous murine SOD1, thus supporting the preclinical utility of anti-SOD1 nanobodies for testing in animal models of ALS. In sum, the anti-SOD1 nanobodies developed and presented herein represent viable biologics for further preclinical testing in human and mouse models of ALS.

[end paper 54]

[begin paper 55]

PMCID = 8465657

Title = A Small Virus to Deliver Small Antibodies: New Targeted Therapies Based on AAV Delivery of Nanobodies

Nanobodies are camelid-derived single-domain antibodies that present some advantages versus conventional antibodies, such as a smaller size, and higher tissue penetrability, stability, and hydrophilicity. Although nanobodies can be delivered as proteins, in vivo expression from adeno-associated viral (AAV) vectors represents an attractive strategy. This is due to the fact that AAV vectors, that can provide long-term expression of recombinant genes, have shown an excellent safety profile, and can accommodate genes for one or several nanobodies. In fact, several studies showed that AAV vectors can provide sustained nanobody expression both locally or systemically in preclinical models of human diseases. Some of the pathologies addressed with this technology include cancer, neurological, cardiovascular, infectious, and genetic diseases. Depending on the indication, AAV-delivered nanobodies can be expressed extracellularly or inside cells. Intracellular nanobodies or “intrabodies” carry out their function by interacting with cell proteins involved in disease and have also been designed to help elucidate cellular mechanisms by interfering with normal cell processes. Finally, nanobodies can also be used to retarget AAV vectors, when tethered to viral capsid proteins. This review covers applications in which AAV vectors have been used to deliver nanobodies, with a focus on their therapeutic use.

[end paper 55]

[begin paper 56]

PMCID = 8164649

Title = Nanobodies in cancer

For treatment and diagnosis of cancer, antibodies have proven their value and now serve as a first line of therapy for certain cancers. A unique class of antibody fragments called nanobodies, derived from camelid heavy chain-only antibodies, are gaining increasing acceptance as diagnostic tools and are considered also as building blocks for chimeric antigen receptors as well as for targeted drug delivery. The small size of nanobodies (~15 kDa), their stability, ease of manufacture and modification for diverse formats, short circulatory half-life, and high tissue penetration, coupled with excellent specificity and affinity, account for their attractiveness. Here we review applications of nanobodies in the sphere of tumor biology.

[end paper 56]

[begin paper 57]

PMCID = 6247239

Title = Award Winners and Abstracts of the 32nd Annual Symposium of The Protein Society; Boston, MA, July 9–12, 2018



[end paper 57]

[begin paper 58]

PMCID = 8130932

Title = Breadth and function of antibody response to acute SARS-CoV-2 infection in humans

Serological and plasmablast responses and plasmablast-derived IgG monoclonal antibodies (MAbs) have been analysed in three COVID-19 patients with different clinical severities. Potent humoral responses were detected within 3 weeks of onset of illness in all patients and the serological titre was elicited soon after or concomitantly with peripheral plasmablast response. An average of 13.7% and 3.5% of plasmablast-derived MAbs were reactive with virus spike glycoprotein or nucleocapsid, respectively. A subset of anti-spike (10 of 32) antibodies cross-reacted with other betacoronaviruses tested and harboured extensive somatic mutations, indicative of an expansion of memory B cells upon SARS-CoV-2 infection. Fourteen of 32 anti-spike MAbs, including five anti-receptor-binding domain (RBD), three anti-non-RBD S1 and six anti-S2, neutralised wild-type SARS-CoV-2 in independent assays. Anti-RBD MAbs were further grouped into four cross-inhibiting clusters, of which six antibodies from three separate clusters blocked the binding of RBD to ACE2 and five were neutralising. All ACE2-blocking anti-RBD antibodies were isolated from two recovered patients with prolonged fever, which is compatible with substantial ACE2-blocking response in their sera. Finally, the identification of non-competing pairs of neutralising antibodies would offer potential templates for the development of prophylactic and therapeutic agents against SARS-CoV-2.

The global COVID-19 outbreak poses a serious threat to human health and antibody-mediated immunity plays a key role in controlling acute viral infection in humans. We report the complete mapping of antibody responses, from serology through to single plasmablast-derived antibody clone, in three COVID-19 patients with different severities. The data show that a subset of anti-spike plasmablast-derived antibodies cross-react with other betacoronaviruses including human coronavirus OC43, which suggests an expansion of memory B cells upon SARS-CoV-2 infection. Anti-SARS-CoV-2 spike antibody clones target a diverse spectrum of epitopes on the receptor-binding domain (RBD), non-RBD S1 and S2 regions of the spike glycoprotein, 40% of them neutralise wild-type SARS-CoV-2. Anti-RBD antibodies constitute a major part of neutralising antibody response. Potent antibodies target three non-overlapping epitopes on the RBD, and the neutralising activity is linked to ACE2-binding blockade. Combinations of multiple antibody clones targeting non-overlapping epitopes offer a potential avenue to combat the global outbreak.

[end paper 58]

[begin paper 59]

PMCID = 9512078

Title = Mucosal nanobody IgA as inhalable and affordable prophylactic and therapeutic treatment against SARS-CoV-2 and emerging variants

Anti-COVID antibody therapeutics have been developed but not widely used due to their high cost and escape of neutralization from the emerging variants. Here, we describe the development of VHH-IgA1.1, a nanobody IgA fusion molecule as an inhalable, affordable and less invasive prophylactic and therapeutic treatment against SARS-CoV-2 Omicron variants. VHH-IgA1.1 recognizes a conserved epitope of SARS-CoV-2 spike protein Receptor Binding Domain (RBD) and potently neutralizes major global SARS-CoV-2 variants of concern (VOC) including the Omicron variant and its sub lineages BA.1.1, BA.2 and BA.2.12.1. VHH-IgA1.1 is also much more potent against Omicron variants as compared to an IgG Fc fusion construct, demonstrating the importance of IgA mediated mucosal protection for Omicron infection. Intranasal administration of VHH-IgA1.1 prior to or after challenge conferred significant protection from severe respiratory disease in K18-ACE2 transgenic mice infected with SARS-CoV-2 VOC. More importantly, for cost-effective production, VHH-IgA1.1 produced in Pichia pastoris had comparable potency to mammalian produced antibodies. Our study demonstrates that intranasal administration of affordably produced VHH-IgA fusion protein provides effective mucosal immunity against infection of SARS-CoV-2 including emerging variants.

[end paper 59]

[begin paper 60]

PMCID = 10878242

Title = A novel strategy for an anti-idiotype vaccine: nanobody mimicking neutralization epitope of porcine circovirus type 2

Vaccination is the most effective method to protect humans and animals from diseases. Anti-idiotype vaccines are safer due to their absence of pathogens. However, the commercial production of traditional anti-idiotype vaccines using monoclonal and polyclonal antibodies (mAb and pAb) is complex and has a high failure rate. The present study designed a novel, simple, low-cost strategy for developing anti-idiotype vaccines with nanobody technology. We used porcine circovirus type 2 (PCV2) as a viral model, which can result in serious economic loss in the pig industry. The neutralizing mAb-1E7 (Ab1) against PCV2 capsid protein (PCV2-Cap) was immunized in the camel. And 12 nanobodies against mAb-1E7 were screened. Among them, Nb61 (Ab2) targeted the idiotype epitope of mAb-1E7 and blocked mAb-1E7’s binding to PCV2-Cap. Additionally, a high-dose Nb61 vaccination can also protect mice and pigs from PCV2 infection. Epitope mapping showed that mAb-1E7 recognized the 75NINDFL80 of PCV2-Cap and 101NYNDFLG107 of Nb61. Subsequently, the mAb-3G4 (Ab3) against Nb61 was produced and can neutralize PCV2 infection in the PK-15 cells. Structure analysis showed that the amino acids of mAb-1E7 and mAb-3G4 respective binding to PCV2-Cap and Nb61 were also similar on the amino acids sequences and spatial conformation. Collectively, our study first provided a strategy for producing nanobody-based anti-idiotype vaccines and identified that anti-idiotype nanobodies could mimic the antigen on amino acids and structures. Importantly, as more and more neutralization mAbs against different pathogens are prepared, anti-idiotype nanobody vaccines can be easily produced against the disease with our strategy, especially for dangerous pathogens.

Anti-idiotype vaccines utilize idiotype-anti-idiotype network theory, eliminating the need for external antigens as vaccine candidates. Especially for dangerous pathogens, they were safer because they did not contact the live pathogenic microorganisms. However, developing anti-idiotype vaccines with traditional monoclonal and polyclonal antibodies is complex and has a high failure rate. We present a novel, universal, simple, low-cost strategy for producing anti-idiotype vaccines with nanobody technology. Using a neutralization antibody against PCV2-Cap, a nanobody (Ab2) was successfully produced and could mimic the neutralizing epitope of PCV2-Cap. The nanobody can induce protective immune responses against PCV2 infection in mice and pigs. It highlighted that the anti-idiotype vaccine using nanobody has a very good application in the future, especially for dangerous pathogens.

[end paper 60]

[begin paper 61]

PMCID = 9491202

Title = Sensitivity to Vaccines, Therapeutic Antibodies, and Viral Entry Inhibitors and Advances To Counter the SARS-CoV-2 Omicron Variant

Severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) keeps evolving and mutating into newer variants over time, which gain higher transmissibility, disease severity, and spread in communities at a faster rate, resulting in multiple waves of surge in Coronavirus Disease 2019 (COVID-19) cases. A highly mutated and transmissible SARS-CoV-2 Omicron variant has recently emerged, driving the extremely high peak of infections in almost all continents at an unprecedented speed and scale. The Omicron variant evades the protection rendered by vaccine-induced antibodies and natural infection, as well as overpowers the antibody-based immunotherapies, raising the concerns of current effectiveness of available vaccines and monoclonal antibody-based therapies. This review outlines the most recent advancements in studying the virology and biology of the Omicron variant, highlighting its increased resistance to current antibody-based therapeutics and its immune escape against vaccines. However, the Omicron variant is highly sensitive to viral fusion inhibitors targeting the HR1 motif in the spike protein, enzyme inhibitors, involving the endosomal fusion pathway, and ACE2-based entry inhibitors. Omicron variant-associated infectivity and entry mechanisms of Omicron variant are essentially distinct from previous characterized variants. Innate sensing and immune evasion of SARS-CoV-2 and T cell immunity to the virus provide new perspectives of vaccine and drug development. These findings are important for understanding SARS-CoV-2 viral biology and advances in developing vaccines, antibody-based therapies, and more effective strategies to mitigate the transmission of the Omicron variant or the next SARS-CoV-2 variant of concern.

[end paper 61]

[begin paper 62]

PMCID = 11240265

Title = Biospecific Chemistry for Covalent Linking of Biomacromolecules

Interactions among biomacromolecules, predominantly noncovalent, underpin biological processes. However, recent advancements in biospecific chemistry have enabled the creation of specific covalent bonds between biomolecules, both in vitro and in vivo. This Review traces the evolution of biospecific chemistry in proteins, emphasizing the role of genetically encoded latent bioreactive amino acids. These amino acids react selectively with adjacent natural groups through proximity-enabled bioreactivity, enabling targeted covalent linkages. We explore various latent bioreactive amino acids designed to target different protein residues, ribonucleic acids, and carbohydrates. We then discuss how these novel covalent linkages can drive challenging protein properties and capture transient protein–protein and protein–RNA interactions in vivo. Additionally, we examine the application of covalent peptides as potential therapeutic agents and site-specific conjugates for native antibodies, highlighting their capacity to form stable linkages with target molecules. A significant focus is placed on proximity-enabled reactive therapeutics (PERx), a pioneering technology in covalent protein therapeutics. We detail its wide-ranging applications in immunotherapy, viral neutralization, and targeted radionuclide therapy. Finally, we present a perspective on the existing challenges within biospecific chemistry and discuss the potential avenues for future exploration and advancement in this rapidly evolving field.

[end paper 62]

[begin paper 63]

PMCID = 7506210

Title = Receptor-binding domain-specific human neutralizing monoclonal antibodies against SARS-CoV and SARS-CoV-2

The outbreaks of severe acute respiratory syndrome (SARS) and Coronavirus Disease 2019 (COVID-19) caused by SARS-CoV and SARS-CoV-2, respectively, have posed severe threats to global public health and the economy. Treatment and prevention of these viral diseases call for the research and development of human neutralizing monoclonal antibodies (NMAbs). Scientists have screened neutralizing antibodies using the virus receptor-binding domain (RBD) as an antigen, indicating that RBD contains multiple conformational neutralizing epitopes, which are the main structural domains for inducing neutralizing antibodies and T-cell immune responses. This review summarizes the structure and function of RBD and RBD-specific NMAbs against SARS-CoV and SARS-CoV-2 currently under development.

[end paper 63]

[begin paper 64]

PMCID = 10359190

Title = CRISPR-Cas-amplified urinary biomarkers for multiplexed and portable cancer diagnostics

Synthetic biomarkers, bioengineered sensors that generate molecular reporters in diseased microenvironments, represent an emerging paradigm in precision diagnostics. Despite the utility of DNA barcodes as a multiplexing tool, their susceptibility to nucleases in vivo has limited their utility. Here we exploit chemically stabilized nucleic acids to multiplex synthetic biomarkers and produce diagnostic signals in biofluids that can be ‘read out’ via CRISPR nucleases. The strategy relies on microenvironmental endopeptidase to trigger the release of nucleic acid barcodes and polymerase-amplification-free, CRISPR-Cas-mediated barcode detection in unprocessed urine. Our data suggest that DNA-encoded nanosensors can non-invasively detect and differentiate disease states in transplanted and autochthonous murine cancer models. We also demonstrate that CRISPR-Cas amplification can be harnessed to convert the readout to a point-of-care paper diagnostic tool. Finally, we employ a microfluidic platform for densely multiplexed, CRISPR-mediated DNA barcode readout that can potentially evaluate complex human diseases rapidly and guide therapeutic decisions.

Protease profiling of the tumour microenvironment using DNA-barcoded nanosensors and CRISPR amplification enables portable cancer diagnosis in preclinical models.

[end paper 64]

[begin paper 65]

PMCID = 10060843

Title = Review of therapeutic mechanisms and applications based on SARS-CoV-2 neutralizing antibodies

COVID-19 pandemic is a global public health emergency. Despite extensive research, there are still few effective treatment options available today. Neutralizing-antibody-based treatments offer a broad range of applications, including the prevention and treatment of acute infectious diseases. Hundreds of SARS-CoV-2 neutralizing antibody studies are currently underway around the world, with some already in clinical applications. The development of SARS-CoV-2 neutralizing antibody opens up a new therapeutic option for COVID-19. We intend to review our current knowledge about antibodies targeting various regions (i.e., RBD regions, non-RBD regions, host cell targets, and cross-neutralizing antibodies), as well as the current scientific evidence for neutralizing-antibody-based treatments based on convalescent plasma therapy, intravenous immunoglobulin, monoclonal antibodies, and recombinant drugs. The functional evaluation of antibodies (i.e., in vitro or in vivo assays) is also discussed. Finally, some current issues in the field of neutralizing-antibody-based therapies are highlighted.

[end paper 65]

[begin paper 66]

PMCID = 9411660

Title = Research progress and applications of nanobody in human infectious diseases

Infectious diseases, caused by pathogenic microorganisms, are capable of affecting crises. In addition to persistent infectious diseases such as malaria and dengue fever, the vicious outbreaks of infectious diseases such as Neocon, Ebola and SARS-CoV-2 in recent years have prompted the search for more efficient and convenient means for better diagnosis and treatment. Antibodies have attracted a lot of attention due to their good structural characteristics and applications. Nanobodies are the smallest functional single-domain antibodies known to be able to bind stably to antigens, with the advantages of high stability, high hydrophilicity, and easy expression and modification. They can directly target antigen epitopes or be constructed as multivalent nanobodies or nanobody fusion proteins to exert therapeutic effects. This paper focuses on the construction methods and potential functions of nanobodies, outlines the progress of their research, and highlights their various applications in human infectious diseases.

[end paper 66]

[begin paper 67]

PMCID = 9892613

Title = Thermodynamic analysis of an entropically driven, high-affinity nanobody-HIV p24 interaction

Protein-protein interactions are fundamental to life processes. Complementary computational, structural, and biophysical studies of these interactions enable the forces behind their specificity and strength to be understood. Antibody fragments such as single-chain antibodies have the specificity and affinity of full antibodies but a fraction of their size, expediting whole molecule studies and distal effects without exceeding the computational capacity of modeling systems. We previously reported the crystal structure of a high-affinity nanobody 59H10 bound to HIV-1 capsid protein p24 and deduced key interactions using all-atom molecular dynamics simulations. We studied the properties of closely related medium (37E7) and low (48G11) affinity nanobodies, to understand how changes of three (37E7) or one (48G11) amino acids impacted these interactions; however, the contributions of enthalpy and entropy were not quantified. Here, we report the use of qualitative and quantitative experimental and in silico approaches to separate the contributions of enthalpy and entropy. We used complementary circular dichroism spectroscopy and molecular dynamics simulations to qualitatively delineate changes between nanobodies in isolation and complexed with p24. Using quantitative techniques such as isothermal titration calorimetry alongside WaterMap and Free Energy Perturbation protocols, we found the difference between high (59H10) and medium (37E7) affinity nanobodies on binding to HIV-1 p24 is entropically driven, accounted for by the release of unstable waters from the hydrophobic surface of 59H10. Our results provide an exemplar of the utility of parallel in vitro and in silico studies and highlight that differences in entropic interactions between amino acids and water molecules are sufficient to drive orders of magnitude differences in affinity.

[end paper 67]

[begin paper 68]

PMCID = 11222437

Title = Mammalian cell display with automated oligo design and library assembly allows for rapid residue level conformational epitope mapping

Precise epitope determination of therapeutic antibodies is of great value as it allows for further comprehension of mechanism of action, therapeutic responsiveness prediction, avoidance of unwanted cross reactivity, and vaccine design. The golden standard for discontinuous epitope determination is the laborious X-ray crystallography method. Here, we present a combinatorial method for rapid mapping of discontinuous epitopes by mammalian antigen display, eliminating the need for protein expression and purification. The method is facilitated by automated workflows and tailored software for antigen analysis and oligonucleotide design. These oligos are used in automated mutagenesis to generate an antigen receptor library displayed on mammalian cells for direct binding analysis by flow cytometry. Through automated analysis of 33930 primers an optimized single condition cloning reaction was defined allowing for mutation of all surface-exposed residues of the receptor binding domain of SARS-CoV-2. All variants were functionally expressed, and two reference binders validated the method. Furthermore, epitopes of three novel therapeutic antibodies were successfully determined followed by evaluation of binding also towards SARS-CoV-2 Omicron BA.2. We find the method to be highly relevant for rapid construction of antigen libraries and determination of antibody epitopes, especially for the development of therapeutic interventions against novel pathogens.

Using a new method for rapid, residue-level epitope determination combining flow cytometry and surface displayed alanine substitution variants the authors determined the epitopes of three new antibodies binding the SARS-CoV-2 Spike protein.

[end paper 68]

[begin paper 69]

PMCID = 10588750

Title = A Naïve Phage Display Library-Derived Nanobody Neutralizes SARS-CoV-2 and Three Variants of Concern

The emergence of the coronavirus disease 2019 (COVID-19) pandemic and the new severe acute respiratory syndrome coronavirus-2 (SARS-CoV-2) variants of concern (VOCs) requires the continuous development of safe, effective, and affordable prevention and therapeutics. Nanobodies have demonstrated antiviral activity against a variety of viruses, providing a new candidate for the prevention and treatment of SARS-CoV-2 and its variants.

SARS-CoV-2 glycoprotein spike 1 subunit (S1) was selected as the target antigen for nanobody screening of a naïve phage display library. We obtained a nanobody, named Nb-H6, and then determined its affinity, inhibition, and stability by ELISA, Competitive ELISA, and Biolayer Interferometry (BLI). Infection assays of authentic and pseudotyped SARS-CoV-2 were performed to evaluate the neutralization of Nb-H6. The structure and mechanism of action were investigated by AlphaFold, docking, and residue mutation assays.

We isolated and characterized a nanobody, Nb-H6, which exhibits a broad affinity for S1 and the receptor binding domain (RBD) of SARS-CoV-2, or Alpha (B.1.1.7), Delta (B.1.617.2), Lambda (C.37), and Omicron (BA.2 and BA.5), and blocks receptor angiotensin-converting enzyme 2 (ACE2) binding. Moreover, Nb-H6 can retain its binding capability after pH or thermal treatment and effectively neutralize both pseudotyped and authentic SARS-CoV-2, as well as VOC Alpha (B.1.1.7), Delta (B.1.617.2), and Omicron (BA.2 and BA.5) pseudoviruses. We also confirmed that Nb-H6 binds two distinct amino acid residues of the RBD, preventing SARS-CoV-2 from interacting with the host receptor.

Our study highlights a novel nanobody, Nb-H6, that may be useful therapeutically in SARS-CoV-2 and VOC outbreaks and pandemics. These findings also provide a molecular foundation for further studies into how nanobodies neutralize SARS-CoV-2 and variants and imply potential therapeutic targets for the treatment of COVID-19.

[end paper 69]

[begin paper 70]

PMCID = 11546159

Title = CAR-T cells based on a TCR mimic nanobody targeting HPV16 E6 exhibit antitumor activity against cervical cancer

The E6 and E7 oncoproteins of human papillomavirus (HPV) are considered promising targets for HPV-related cancers. In this study, we evaluated novel T cell receptor mimic (TCRm) nanobodies targeting the E629-38 peptide complexed with human leukocyte antigen (HLA)-A∗02:01 in the chimeric antigen receptor (CAR) format. We isolated two dromedary camel nanobodies, F5 and G9, through phage display screening. F5 bound more efficiently to the complex expressed on cells, including peptide-pulsed T2, overexpressed 293E6, and cervical cancer lines CaSki and SS4050, compared to G9. CAR-T cells based on the F5 nanobody specifically killed target cells, including 293E6, CaSki, and SS4050 in vitro, through activation of nuclear factor of activated T cells (NFAT) and nuclear factor κB (NF-κB) signaling. Importantly, F5 CAR-T cells inhibited the growth of CaSki and SS4050 tumor xenografts in mice. These findings demonstrate that HPV-16+ cervical cancer can be targeted by F5 nanobody-based CAR-T cells, offering a valuable alternative strategy for treating HPV-16+ malignancies.

Ho and colleagues discovered a camel single-domain antibody, known as F5, that targets the oncogenic E6 of human papillomavirus (HPV) 16. Chimeric antigen receptor (CAR) T cells based on this nanobody showed promising efficacy in killing HPV16+ tumor lines and inhibiting tumor growth in cervical cancer xenografts in mice.

[end paper 70]

[begin paper 71]

PMCID = 5993540

Title = Viral GPCR US28 can signal in response to chemokine agonists of nearly unlimited structural degeneracy

Human cytomegalovirus has hijacked and evolved a human G-protein-coupled receptor into US28, which functions as a promiscuous chemokine 'sink’ to facilitate evasion of host immune responses. To probe the molecular basis of US28’s unique ligand cross-reactivity, we deep-sequenced CX3CL1 chemokine libraries selected on ‘molecular casts’ of the US28 active-state and find that US28 can engage thousands of distinct chemokine sequences, many of which elicit diverse signaling outcomes. The structure of a G-protein-biased CX3CL1-variant in complex with US28 revealed an entirely unique chemokine amino terminal peptide conformation and remodeled constellation of receptor-ligand interactions. Receptor signaling, however, is remarkably robust to mutational disruption of these interactions. Thus, US28 accommodates and functionally discriminates amongst highly degenerate chemokine sequences by sensing the steric bulk of the ligands, which distort both receptor extracellular loops and the walls of the ligand binding pocket to varying degrees, rather than requiring sequence-specific bonding chemistries for recognition and signaling.

[end paper 71]

[begin paper 72]

PMCID = 4034163

Title = The past, present and future of neutralizing antibodies for hepatitis C virus

Recent studies have provided insight into the protective role of neutralizing antibodies in hepatitis C.

Neutralizing antibodies show broad reactivity for diverse HCV genotypes.

Recombinant HCV glycoproteins can elicit neutralizing antibodies.

The HCV E2 core structure can inform rational design of immunogens.

Hepatitis C virus (HCV) is a major cause of liver disease and hepatocellular carcinoma worldwide. HCV establishes a chronic infection in the majority of cases. However, some individuals clear the virus, demonstrating a protective role for the host immune response. Although new all-oral drug combinations may soon replace traditional ribavirin–interferon therapy, the emerging drug cocktails will be expensive and associated with side-effects and resistance, making a global vaccine an urgent priority. T cells are widely accepted to play an essential role in clearing acute HCV infection, whereas the role antibodies play in resolution and disease pathogenesis is less well understood. Recent studies have provided an insight into viral neutralizing determinants and the protective role of antibodies during infection. This review provides a historical perspective of the role neutralizing antibodies play in HCV infection and discusses the therapeutic benefits of antibody-based therapies. This article forms part of a symposium in Antiviral Research on “Hepatitis C: next steps toward global eradication.”

[end paper 72]

[begin paper 73]

PMCID = 5959942

Title = Targeting G protein-coupled receptor signaling at the G protein level with a selective nanobody inhibitor

G protein-coupled receptors (GPCRs) activate heterotrimeric G proteins by mediating a GDP to GTP exchange in the Gα subunit. This leads to dissociation of the heterotrimer into Gα-GTP and Gβγ dimer. The Gα-GTP and Gβγ dimer each regulate a variety of downstream pathways to control various aspects of human physiology. Dysregulated Gβγ-signaling is a central element of various neurological and cancer-related anomalies. However, Gβγ also serves as a negative regulator of Gα that is essential for G protein inactivation, and thus has the potential for numerous side effects when targeted therapeutically. Here we report a llama-derived nanobody (Nb5) that binds tightly to the Gβγ dimer. Nb5 responds to all combinations of β-subtypes and γ-subtypes and competes with other Gβγ-regulatory proteins for a common binding site on the Gβγ dimer. Despite its inhibitory effect on Gβγ-mediated signaling, Nb5 has no effect on Gαq-mediated and Gαs-mediated signaling events in living cells.

G protein-coupled receptors (GPCRs) activate and dissociate the G protein heterotrimer into Gα-GTP and Gβγ dimer, which facilitate distinct signalling events. Here authors develop a nanobody, Nb5 that modulates Gβγ-mediated signaling without affecting GTP-bound Gαq and Gαs-mediated signaling events.

[end paper 73]

[begin paper 74]

PMCID = 11209230

Title = Comprehensive Overview of Broadly Neutralizing Antibodies against SARS-CoV-2 Variants

Currently, SARS-CoV-2 has evolved into various variants, including the numerous highly mutated Omicron sub-lineages, significantly increasing immune evasion ability. The development raises concerns about the possibly diminished effectiveness of available vaccines and antibody-based therapeutics. Here, we describe those representative categories of broadly neutralizing antibodies (bnAbs) that retain prominent effectiveness against emerging variants including Omicron sub-lineages. The molecular characteristics, epitope conservation, and resistance mechanisms of these antibodies are further detailed, aiming to offer suggestion or direction for the development of therapeutic antibodies, and facilitate the design of vaccines with broad-spectrum potential.

[end paper 74]

[begin paper 75]

PMCID = 10187017

Title = ISEV2023 Abstract Book



[end paper 75]

[begin paper 76]

PMCID = 7151424

Title = Recombinant expression of nanobodies and nanobody-derived immunoreagents

Antibody fragments for which the sequence is available are suitable for straightforward engineering and expression in both eukaryotic and prokaryotic systems. When produced as fusions with convenient tags, they become reagents which pair their selective binding capacity to an orthogonal function. Several kinds of immunoreagents composed by nanobodies and either large proteins or short sequences have been designed for providing inexpensive ready-to-use biological tools. The possibility to choose among alternative expression strategies is critical because the fusion moieties might require specific conditions for correct folding or post-translational modifications. In the case of nanobody production, the trend is towards simpler but reliable (bacterial) methods that can substitute for more cumbersome processes requiring the use of eukaryotic systems. The use of these will not disappear, but will be restricted to those cases in which the final immunoconstructs must have features that cannot be obtained in prokaryotic cells. At the same time, bacterial expression has evolved from the conventional procedure which considered exclusively the nanobody and nanobody-fusion accumulation in the periplasm. Several reports show the advantage of cytoplasmic expression, surface-display and secretion for at least some applications. Finally, there is an increasing interest to use as a model the short nanobody sequence for the development of in silico methodologies aimed at optimizing the yields, stability and affinity of recombinant antibodies.

There is an increasing request for immunoreagents based on nanobodies.

The multiplicity of their applications requires constructs with different structural complexity.

Alternative expression methods are necessary to achieve such structural requirements.

 In silico optimization of nanobody biophysical characteristics becomes more and more reliable.

[end paper 76]

[begin paper 77]

PMCID = 10968798

Title = Aptamers and Nanobodies as New Bioprobes for SARS-CoV-2 Diagnostic and Therapeutic System Applications

The global challenges posed by the severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) pandemic have underscored the critical importance of innovative and efficient control systems for addressing future pandemics. The most effective way to control the pandemic is to rapidly suppress the spread of the virus through early detection using a rapid, accurate, and easy-to-use diagnostic platform. In biosensors that use bioprobes, the binding affinity of molecular recognition elements (MREs) is the primary factor determining the dynamic range of the sensing platform. Furthermore, the sensitivity relies mainly on bioprobe quality with sufficient functionality. This comprehensive review investigates aptamers and nanobodies recently developed as advanced MREs for SARS-CoV-2 diagnostic and therapeutic applications. These bioprobes might be integrated into organic bioelectronic materials and devices, with promising enhanced sensitivity and specificity. This review offers valuable insights into advancing biosensing technologies for infectious disease diagnosis and treatment using aptamers and nanobodies as new bioprobes.

[end paper 77]

[begin paper 78]

PMCID = 10195009

Title = Mechanism of glycoform specificity and in vivo protection by an anti-afucosylated IgG nanobody

Immunoglobulin G (IgG) antibodies contain a complex N-glycan embedded in the hydrophobic pocket between its heavy chain protomers. This glycan contributes to the structural organization of the Fc domain and determines its specificity for Fcγ receptors, thereby dictating distinct cellular responses. The variable construction of this glycan structure leads to highly-related, but non-equivalent glycoproteins known as glycoforms. We previously reported synthetic nanobodies that distinguish IgG glycoforms. Here, we present the structure of one such nanobody, X0, in complex with the Fc fragment of afucosylated IgG1. Upon binding, the elongated CDR3 loop of X0 undergoes a conformational shift to access the buried N-glycan and acts as a ‘glycan sensor’, forming hydrogen bonds with the afucosylated IgG N-glycan that would otherwise be sterically hindered by the presence of a core fucose residue. Based on this structure, we designed X0 fusion constructs that disrupt pathogenic afucosylated IgG1-FcγRIIIa interactions and rescue mice in a model of dengue virus infection.

Afucosylated immunoglobulins have been shown to enhance certain viral infections. Here, the authors generate and structurally characterize a unique nanobody that inhibits these pathogenic glycoforms.

[end paper 78]

[begin paper 79]

PMCID = 10057852

Title = Nanobodies: A Review of Generation, Diagnostics and Therapeutics

Nanobodies, also referred to as single domain-based VHHs, are antibody fragments derived from heavy-chain only IgG antibodies found in the Camelidae family. Due to their small size, simple structure, high antigen binding affinity, and remarkable stability in extreme conditions, nanobodies possess the potential to overcome several of the limitations of conventional monoclonal antibodies. For many years, nanobodies have been of great interest in a wide variety of research fields, particularly in the diagnosis and treatment of diseases. This culminated in the approval of the world’s first nanobody based drug (Caplacizumab) in 2018 with others following soon thereafter. This review will provide an overview, with examples, of (i) the structure and advantages of nanobodies compared to conventional monoclonal antibodies, (ii) methods used to generate and produce antigen-specific nanobodies, (iii) applications for diagnostics, and (iv) ongoing clinical trials for nanobody therapeutics as well as promising candidates for clinical development.

[end paper 79]

[begin paper 80]

PMCID = 8417108

Title = A Methodological Approach Using rAAV Vectors Encoding Nanobody-Based Biologics to Evaluate ARTC2.2 and P2X7 In Vivo

On murine T cells, mono-ADP ribosyltransferase ARTC2.2 catalyzes ADP-ribosylation of various surface proteins when nicotinamide adenine dinucleotide (NAD+) is released into the extracellular compartment. Covalent ADP-ribosylation of the P2X7 receptor by ARTC2.2 thereby represents an additional mechanism of activation, complementary to its triggering by extracellular ATP. P2X7 is a multifaceted receptor that may represents a potential target in inflammatory, and neurodegenerative diseases, as well as in cancer. We present herein an experimental approach using intramuscular injection of recombinant AAV vectors (rAAV) encoding nanobody-based biologics targeting ARTC2.2 or P2X7. We demonstrate the ability of these in vivo generated biologics to potently and durably block P2X7 or ARTC2.2 activities in vivo, or in contrast, to potentiate NAD+- or ATP-induced activation of P2X7. We additionally demonstrate the ability of rAAV-encoded functional heavy chain antibodies to elicit long-term depletion of T cells expressing high levels of ARTC2.2 or P2X7. Our approach of using rAAV to generate functional nanobody-based biologics in vivo appears promising to evaluate the role of ARTC2.2 and P2X7 in murine acute as well as chronic disease models.

[end paper 80]

[begin paper 81]

PMCID = 11082534

Title = Half‐life extension of single‐domain antibody–drug conjugates by albumin binding moiety enhances antitumor efficacy

Single‐domain antibody–drug conjugates (sdADCs) have been proven to have deeper solid tumor penetration and intratumor accumulation capabilities due to their smaller size compared with traditional IgG format ADCs. However, one of the key challenges for improving clinical outcomes of sdADCs is their abbreviated in vivo half‐life. In this study, we innovatively fused an antihuman serum albumin (αHSA) nanobody to a sdADCs targeting oncofetal antigen 5T4, conferring serum albumin binding to enhance the pharmacokinetic profiles of sdADCs. The fusion protein was conjugated with monomethyl auristatin E (MMAE) at s224c site mutation. The conjugate exhibited potent cytotoxicity against various tumor cells. Compared with the nonalbumin‐binding counterparts, the conjugate exhibited a 10‐fold extended half‐life in wild‐type mice and fivefold prolonged serum half‐life in BxPC‐3 xenograft tumor models as well as enhanced tumor accumulation and retention in mice. Consequently, n501–αHSA–MMAE showed potent antitumor effects, which were comparable to n501–MMAE in pancreatic cancer BxPC‐3 xenograft tumor models; however, in human ovarian teratoma PA‐1 xenograft tumor models, n501–αHSA–MMAE significantly improved antitumor efficacy. Moreover, the conjugate showed mitigated hepatotoxicity. In summary, our results suggested that fusion to albumin‐binding moiety as a viable strategy can enhance the therapeutic potential of sdADCs through optimized pharmacokinetics.

We innovatively fused an antihuman serum albumin nanobody to a single‐domain antibody–drug conjugates targeting oncofetal antigen 5T4, conferring serum albumin binding, exhibiting a 10‐fold extended half‐life and enhanced tumor accumulation and retention in mice. Consequently, it demonstrated significantly improved antitumor efficacy in tumor xenografts despite less frequent dosing with mitigated hepatotoxicity.  

[end paper 81]

[begin paper 82]

PMCID = 8260353

Title = Nanobodies from camelid mice and llamas neutralize SARS-CoV-2 variants

Since the start of the COVID-19 pandemic, SARS-CoV-2 has caused millions of deaths worldwide. Although a number of vaccines have been deployed, the continual evolution of the receptor-binding domain (RBD) of the virus has challenged their efficacy. In particular, the emerging variants B.1.1.7, B.1.351 and P.1 (first detected in the UK, South Africa and Brazil, respectively) have compromised the efficacy of sera from patients who have recovered from COVID-19 and immunotherapies that have received emergency use authorization. One potential alternative to avert viral escape is the use of camelid VHHs (variable heavy chain domains of heavy chain antibody (also known as nanobodies)), which can recognize epitopes that are often inaccessible to conventional antibodies. Here, we isolate anti-RBD nanobodies from llamas and from mice that we engineered to produce VHHs cloned from alpacas, dromedaries and Bactrian camels. We identified two groups of highly neutralizing nanobodies. Group 1 circumvents antigenic drift by recognizing an RBD region that is highly conserved in coronaviruses but rarely targeted by human antibodies. Group 2 is almost exclusively focused to the RBD–ACE2 interface and does not neutralize SARS-CoV-2 variants that carry E484K or N501Y substitutions. However, nanobodies in group 2 retain full neutralization activity against these variants when expressed as homotrimers, and—to our knowledge—rival the most potent antibodies against SARS-CoV-2 that have been produced to date. These findings suggest that multivalent nanobodies overcome SARS-CoV-2 mutations through two separate mechanisms: enhanced avidity for the ACE2-binding domain and recognition of conserved epitopes that are largely inaccessible to human antibodies. Therefore, although new SARS-CoV-2 mutants will continue to emerge, nanobodies represent promising tools to prevent COVID-19 mortality when vaccines are compromised.

Multivalent nanobodies against SARS-CoV-2 from mice engineered to produce camelid nanobodies recognize conserved epitopes that are inaccessible to human antibodies and show promise as a strategy for dealing with viral escape mutations.

[end paper 82]

[begin paper 83]

PMCID = 6032109

Title = Abstract



[end paper 83]

[begin paper 84]

PMCID = 7922070

Title = Nanobodies Provide Insight into the Molecular Mechanisms of the Complement Cascade and Offer New Therapeutic Strategies

The complement system is part of the innate immune response, where it provides immediate protection from infectious agents and plays a fundamental role in homeostasis. Complement dysregulation occurs in several diseases, where the tightly regulated proteolytic cascade turns offensive. Prominent examples are atypical hemolytic uremic syndrome, paroxysmal nocturnal hemoglobinuria and Alzheimer’s disease. Therapeutic intervention targeting complement activation may allow treatment of such debilitating diseases. In this review, we describe a panel of complement targeting nanobodies that allow modulation at different steps of the proteolytic cascade, from the activation of the C1 complex in the classical pathway to formation of the C5 convertase in the terminal pathway. Thorough structural and functional characterization has provided a deep mechanistic understanding of the mode of inhibition for each of the nanobodies. These complement specific nanobodies are novel powerful probes for basic research and offer new opportunities for in vivo complement modulation.

[end paper 84]

[begin paper 85]

PMCID = 8863144

Title = Reduced antigenicity of Omicron lowers host serologic response

SARS-CoV-2 Omicron variant of concern (VOC) contains fifteen mutations on the receptor binding domain (RBD), evading most neutralizing antibodies from vaccinated sera. Emerging evidence suggests that Omicron breakthrough cases are associated with substantially lower antibody titers than other VOC cases. However, the mechanism remains unclear. Here, using a novel geometric deep-learning model, we discovered that the antigenic profile of Omicron RBD is distinct from the prior VOCs, featuring reduced antigenicity in its remodeled receptor binding sites (RBS). To substantiate our deep-learning prediction, we immunized mice with different recombinant RBD variants and found that the Omicron’s extensive mutations can lead to a drastically attenuated serologic response with limited neutralizing activity in vivo, while the T cell response remains potent. Analyses of serum cross-reactivity and competitive ELISA with epitope-specific nanobodies revealed that the antibody response to Omicron was reduced across RBD epitopes, including both the variable RBS and epitopes without any known VOC mutations. Moreover, computational modeling confirmed that the RBS is highly versatile with a capacity to further decrease antigenicity while retaining efficient receptor binding. Longitudinal analysis showed that this evolutionary trend of decrease in antigenicity was also found in hCoV229E, a common cold coronavirus that has been circulating in humans for decades. Thus, our study provided unprecedented insights into the reduced antibody titers associated with Omicron infection, revealed a possible trajectory of future viral evolution and may inform the vaccine development against future outbreaks.

[end paper 85]

[begin paper 86]

PMCID = 11100608

Title = Bispecific antibodies with broad neutralization potency against SARS-CoV-2 variants of concern

The ongoing emergence of SARS-CoV-2 variants of concern (VOCs) that reduce the effectiveness of antibody therapeutics necessitates development of next-generation antibody modalities that are resilient to viral evolution. Here, we characterized N-terminal domain (NTD) and receptor binding domain (RBD)-specific monoclonal antibodies previously isolated from COVID-19 convalescent donors for their activity against emergent SARS-CoV-2 VOCs. Among these, the NTD-specific antibody C1596 displayed the greatest breadth of binding to VOCs, with cryo-EM structural analysis revealing recognition of a distinct NTD epitope outside of the site i antigenic supersite. Given C1596’s favorable binding profile, we designed a series of bispecific antibodies (bsAbs) termed CoV2-biRNs, that featured both NTD and RBD specificities. Notably, two of the C1596-inclusive bsAbs, CoV2-biRN5 and CoV2-biRN7, retained potent in vitro neutralization activity against all Omicron variants tested, including XBB.1.5, EG.5.1, and BA.2.86, contrasting the diminished potency of parental antibodies delivered as monotherapies or as a cocktail. Furthermore, prophylactic delivery of CoV2-biRN5 significantly reduced the viral load within the lungs of K18-hACE2 mice following challenge with SARS-CoV-2 XBB.1.5. In conclusion, our NTD-RBD bsAbs offer promising potential for the design of resilient, next-generation antibody therapeutics against SARS-CoV-2 VOCs.

Bispecific antibodies with a highly cross-reactive NTD antibody demonstrate resilience to SARS-CoV-2 variants of concern.

[end paper 86]

[begin paper 87]

PMCID = 11201815

Title = Fit‐for‐purpose heterodivalent single‐domain antibody for gastrointestinal targeting of toxin B from Clostridium difficile 

Single‐domain antibodies (sdAbs), such as VHHs, are increasingly being developed for gastrointestinal (GI) applications against pathogens to strengthen gut health. However, what constitutes a suitable developability profile for applying these proteins in a gastrointestinal setting remains poorly explored. Here, we describe an in vitro methodology for the identification of sdAb derivatives, more specifically divalent VHH constructs, that display extraordinary developability properties for oral delivery and functionality in the GI environment. We showcase this by developing a heterodivalent VHH construct that cross‐inhibits the toxic activity of the glycosyltransferase domains (GTDs) from three different toxinotypes of cytotoxin B (TcdB) from lineages of Clostridium difficile. We show that the VHH construct possesses high stability and binding activity under gastric conditions, in the presence of bile salts, and at high temperatures. We suggest that the incorporation of early developability assessment could significantly aid in the efficient discovery of VHHs and related constructs fit for oral delivery and GI applications.

[end paper 87]

[begin paper 88]

PMCID = 5838514

Title = Nanobodies: Chemical Functionalization Strategies and Intracellular Applications

Nanobodies can be seen as next‐generation tools for the recognition and modulation of antigens that are inaccessible to conventional antibodies. Due to their compact structure and high stability, nanobodies see frequent usage in basic research, and their chemical functionalization opens the way towards promising diagnostic and therapeutic applications. In this Review, central aspects of nanobody functionalization are presented, together with selected applications. While early conjugation strategies relied on the random modification of natural amino acids, more recent studies have focused on the site‐specific attachment of functional moieties. Such techniques include chemoenzymatic approaches, expressed protein ligation, and amber suppression in combination with bioorthogonal modification strategies. Recent applications range from sophisticated imaging and mass spectrometry to the delivery of nanobodies into living cells for the visualization and manipulation of intracellular antigens.

[end paper 88]

[begin paper 89]

PMCID = 11071268

Title = Engineering PD-1-targeted small protein variants for in vitro diagnostics and in vivo PET imaging

Programmed cell death 1 (PD-1) belongs to immune checkpoint proteins ensuring negative regulation of the immune response. In non-small cell lung cancer (NSCLC), the sensitivity to treatment with anti-PD-1 therapeutics, and its efficacy, mostly correlated with the increase of tumor infiltrating PD-1+ lymphocytes. Due to solid tumor heterogeneity of PD-1+ populations, novel low molecular weight anti-PD-1 high-affinity diagnostic probes can increase the reliability of expression profiling of PD-1+ tumor infiltrating lymphocytes (TILs) in tumor tissue biopsies and in vivo mapping efficiency using immune-PET imaging.

We designed a 13 kDa β-sheet Myomedin scaffold combinatorial library by randomization of 12 mutable residues, and in combination with ribosome display, we identified anti-PD-1 Myomedin variants (MBA ligands) that specifically bound to human and murine PD-1-transfected HEK293T cells and human SUP-T1 cells spontaneously overexpressing cell surface PD-1.

Binding affinity to cell-surface expressed human and murine PD-1 on transfected HEK293T cells was measured by fluorescence with LigandTracer and resulted in the selection of most promising variants MBA066 (hPD-1 KD = 6.9 nM; mPD-1 KD = 40.5 nM), MBA197 (hPD-1 KD = 29.7 nM; mPD-1 KD = 21.4 nM) and MBA414 (hPD-1 KD = 8.6 nM; mPD-1 KD = 2.4 nM). The potential of MBA proteins for imaging of PD-1+ populations in vivo was demonstrated using deferoxamine-conjugated MBA labeled with 68Galium isotope. Radiochemical purity of 68Ga-MBA proteins reached values 94.7–99.3% and in vitro stability in human serum after 120 min was in the range 94.6–98.2%. The distribution of 68Ga-MBA proteins in mice was monitored using whole-body positron emission tomography combined with computerized tomography (PET/CT) imaging up to 90 min post-injection and post mortem examined in 12 mouse organs. The specificity of MBA proteins was proven by co-staining frozen sections of human tonsils and NSCLC tissue biopsies with anti-PD-1 antibody, and demonstrated their potential for mapping PD-1+ populations in solid tumors.

Using directed evolution, we developed a unique set of small binding proteins that can improve PD-1 diagnostics in vitro as well as in vivo using PET/CT imaging.

The online version contains supplementary material available at 10.1186/s12967-024-05210-x.

[end paper 89]

[begin paper 90]

PMCID = 11148044

Title = Generation of nanobodies from transgenic ‘LamaMice’ lacking an endogenous immunoglobulin repertoire

Due to their exceptional solubility and stability, nanobodies have emerged as powerful building blocks for research tools and therapeutics. However, their generation in llamas is cumbersome and costly. Here, by inserting an engineered llama immunoglobulin heavy chain (IgH) locus into IgH-deficient mice, we generate a transgenic mouse line, which we refer to as ‘LamaMouse’. We demonstrate that LamaMice solely express llama IgH molecules without association to Igκ or λ light chains. Immunization of LamaMice with AAV8, the receptor-binding domain of the SARS-CoV-2 spike protein, IgE, IgG2c, and CLEC9A enabled us to readily select respective target-specific nanobodies using classical hybridoma and phage display technologies, single B cell screening, and direct cloning of the nanobody-repertoire into a mammalian expression vector. Our work shows that the LamaMouse represents a flexible and broadly applicable platform for a facilitated selection of target-specific nanobodies.

Nanobodies are normally made from immunized camelids, Ig transgenic mice or synthetic libraries. In this study, the authors introduce the llama Ig heavy chain locus into mice lacking this locus, thereby generating a line in which nanobodies can be made by direct immunization in the absence of an endogenous antibody repertoire.

[end paper 90]

[begin paper 91]

PMCID = 7561424

Title = Nanobodies—Useful Tools for Allergy Treatment?

In the last decade single domain antibodies (nanobodies, VHH) qualified through their unique characteristics have emerged as accepted and even advantageous alternative to conventional antibodies and have shown great potential as diagnostic and therapeutic tools. Currently nanobodies find their main medical application area in the fields of oncology and neurodegenerative diseases. According to late-breaking information, nanobodies specific for coronavirus spikes have been generated these days to test their suitability as useful therapeutics for future outbreaks. Their superior properties such as chemical stability, high affinity to a broad spectrum of epitopes, low immunogenicity, ease of their generation, selection and production proved nanobodies also to be remarkable to investigate their efficacy for passive treatment of type I allergy, an exaggerated immune reaction to foreign antigens with increasing global prevalence.

[end paper 91]

[begin paper 92]

PMCID = 11245638

Title = ISEV2024 Abstract Book



[end paper 92]

[begin paper 93]

PMCID = 8061467

Title = Integrative overview of antibodies against SARS-CoV-2 and their possible applications in COVID-19 prophylaxis and treatment

SARS-CoV-2 is a novel β-coronavirus that caused the COVID-19 pandemic disease, which spread rapidly, infecting more than 134 million people, and killing almost 2.9 million thus far. Based on the urgent need for therapeutic and prophylactic strategies, the identification and characterization of antibodies has been accelerated, since they have been fundamental in treating other viral diseases. Here, we summarized in an integrative manner the present understanding of the immune response and physiopathology caused by SARS-CoV-2, including the activation of the humoral immune response in SARS-CoV-2 infection and therefore, the synthesis of antibodies. Furthermore, we also discussed about the antibodies that can be generated in COVID-19 convalescent sera and their associated clinical studies, including a detailed characterization of a variety of human antibodies and identification of antibodies from other sources, which have powerful neutralizing capacities. Accordingly, the development of effective treatments to mitigate COVID-19 is expected. Finally, we reviewed the challenges faced in producing potential therapeutic antibodies and nanobodies by cell factories at an industrial level while ensuring their quality, efficacy, and safety.

The online version contains supplementary material available at 10.1186/s12934-021-01576-5.

[end paper 93]

[begin paper 94]

PMCID = 9174178

Title = Superimmunity by pan-sarbecovirus nanobodies

Vaccine boosters and infection can facilitate the development of SARS-CoV-2 antibodies with improved potency and breadth. Here, we observe superimmunity in a camelid extensively immunized with the SARS-CoV-2 receptor-binding domain (RBD). We rapidly isolate a large repertoire of specific ultra-high-affinity nanobodies that bind strongly to all known sarbecovirus clades using integrative proteomics. These pan-sarbecovirus nanobodies (psNbs) are highly effective against SARS-CoV and SARS-CoV-2 variants, including Omicron, with the best median neutralization potency at single-digit nanograms per milliliter. A highly potent, inhalable, and bispecific psNb (PiN-31) is also developed. Structural determinations of 13 psNbs with the SARS-CoV-2 spike or RBD reveal five epitope classes, providing insights into the mechanisms and evolution of their broad activities. The highly evolved psNbs target small, flat, and flexible epitopes that contain over 75% of conserved RBD surface residues. Their potencies are strongly and negatively correlated with the distance of the epitopes from the receptor binding sites.

Xiang et al. characterize 100 diverse pan-sarbecovirus nanobodies (psNbs) that are highly effective against SARS-CoV and SARS-CoV-2 variants, including Omicron, with the best median neutralization potency at single-digit nanograms per milliliter. Structural determinations of 13 psNbs reveal five RBD epitope classes, providing insights into the mechanisms and evolution of their broad activities.

[end paper 94]

[begin paper 95]

PMCID = 10715488

Title = Identification of nurse shark VNAR single-domain antibodies targeting the spike S2 subunit of SARS-CoV-2

SARS-CoV-2 is the etiological agent of the COVID-19 pandemic. Antibody-based therapeutics targeting the spike protein, specifically the S1 subunit or the receptor binding domain (RBD) of SARS-CoV-2, have gained attention due to their clinical efficacy in treating patients diagnosed with COVID-19. An alternative to conventional antibody therapeutics is the use of shark new antigen variable receptor domain (VNAR) antibodies. VNARs are small (<15 kDa) and can reach deep into the pockets or grooves of the target antigen. Here, we have isolated 53 VNARs that bind to the S2 subunit by phage panning from a naïve nurse shark VNAR phage display library constructed in our laboratory. Among those binders, S2A9 showed the best neutralization activity against the original pseudotyped SARS-CoV-2 virus. Several binders, including S2A9, showed cross-reactivity against S2 subunits from other β coronaviruses. Furthermore, S2A9 showed neutralization activity against all variants of concern (VOCs) from alpha to omicron (including BA1, BA2, BA4, and BA5) in both pseudovirus and live virus neutralization assays. Our findings suggest that S2A9 could be a promising lead molecule for the development of broadly neutralizing antibodies against SARS-CoV-2 and emerging variants. The nurse shark VNAR phage library offers a novel platform that can be used to rapidly isolate single-domain antibodies against emerging viral pathogens.

[end paper 95]

[begin paper 96]

PMCID = 10200341

Title = Rapid covalent labeling of membrane proteins on living cells using a nanobody-epitope tag pair

Synthetic molecules that form a covalent bond upon binding to a targeted biomolecule (proximity-induced reactivity) are the subject of intense biomedical interest for the unique pharmacological properties imparted by irreversible binding. However, off-target covalent labeling and the lack of molecules with sufficient specificity limit more widespread applications. We describe the first example of a crosslinking platform that uses a synthetic peptide epitope and a single domain antibody (or nanobody) pair to form a covalent linkage rapidly and specifically. The rate of the crosslinking reaction between peptide and nanobody is faster than most other biocompatible crosslinking reactions, and it can be used to label live cells expressing receptor-nanobody fusions. The rapid kinetics of this system allowed us to probe the consequences on signaling for ligand crosslinking to the A2A-adenosine receptor. Our method may be generally useful to site-specifically link synthetic molecules to receptors on mammalian cell surfaces.

[end paper 96]

[begin paper 97]

PMCID = 11515141

Title = Discovery of nanobodies: a comprehensive review of their applications and potential over the past five years

Nanobodies (Nbs) are antibody fragments derived from heavy-chain-only IgG antibodies found in the Camelidae family as well as cartilaginous fish. Their unique structural and functional properties, such as their small size, the ability to be engineered for high antigen-binding affinity, stability under extreme conditions, and ease of production, have made them promising tools for diagnostics and therapeutics. This potential was realized in 2018 with the approval of caplacizumab, the world’s first Nb-based drug. Currently, Nbs are being investigated in clinical trials for a broad range of treatments, including targeted therapies against PDL1 and Epidermal Growth Factor Receptor (EGFR), cardiovascular diseases, inflammatory conditions, and neurodegenerative disorders such as Alzheimer’s disease, Parkinson’s disease, and amyotrophic lateral sclerosis. They are also being studied for their potential for detecting and imaging autoimmune conditions and infectious diseases such as severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2). A variety of methods are now available to generate target-specific Nbs quickly and efficiently at low costs, increasing their accessibility. This article examines these diverse applications of Nbs and their promising roles. Only the most recent articles published in the last five years have been used to summarize the most advanced developments in the field.

[end paper 97]

[begin paper 98]

PMCID = 10213069

Title = Generation of nanobodies acting as silent and positive allosteric modulators of the α7 nicotinic acetylcholine receptor

The α7 nicotinic acetylcholine receptor (nAChR), a potential drug target for treating cognitive disorders, mediates communication between neuronal and non-neuronal cells. Although many competitive antagonists, agonists, and partial-agonists have been found and synthesized, they have not led to effective therapeutic treatments. In this context, small molecules acting as positive allosteric modulators binding outside the orthosteric, acetylcholine, site have attracted considerable interest. Two single-domain antibody fragments, C4 and E3, against the extracellular domain of the human α7-nAChR were generated through alpaca immunization with cells expressing a human α7-nAChR/mouse 5-HT3A chimera, and are herein described. They bind to the α7-nAChR but not to the other major nAChR subtypes, α4β2 and α3β4. E3 acts as a slowly associating positive allosteric modulator, strongly potentiating the acetylcholine-elicited currents, while not precluding the desensitization of the receptor. An E3–E3 bivalent construct shows similar potentiating properties but displays very slow dissociation kinetics conferring quasi-irreversible properties. Whereas, C4 does not alter the receptor function, but fully inhibits the E3-evoked potentiation, showing it is a silent allosteric modulator competing with E3 binding. Both nanobodies do not compete with α-bungarotoxin, localizing at an allosteric extracellular binding site away from the orthosteric site. The functional differences of each nanobody, as well as the alteration of functional properties through nanobody modifications indicate the importance of this extracellular site. The nanobodies will be useful for pharmacological and structural investigations; moreover, they, along with the extracellular site, have a direct potential for clinical applications.

The online version contains supplementary material available at 10.1007/s00018-023-04779-8.

[end paper 98]

[begin paper 99]

PMCID = 10844482

Title = Nanobodies: a promising approach to treatment of viral diseases

Since their discovery in the 1990s, heavy chain antibodies have garnered significant interest in the scientific community. These antibodies, found in camelids such as llamas and alpacas, exhibit distinct characteristics from conventional antibodies due to the absence of a light chain in their structure. Furthermore, they possess a single antigen-binding domain known as VHH or Nanobody (Nb). With a small size of approximately 15 kDa, these Nbs demonstrate improved characteristics compared to conventional antibodies, including greater physicochemical stability and enhanced biodistribution, enabling them to bind inaccessible epitopes more effectively. As a result, Nbs have found numerous applications in various medical and veterinary fields, particularly in diagnostics and therapeutics. Advances in biotechnology have made the production of recombinant antibodies feasible and compatible with large-scale manufacturing. Through the construction of immune phage libraries that display VHHs and subsequent selection through biopanning, it has become possible to isolate specific Nbs targeting pharmaceutical targets of interest, such as viruses. This review describes the processes involved in nanobody production, from hyperimmunization to purification, with the aim of their application in the pharmaceutical industry.

[end paper 99]

[begin paper 100]

PMCID = 9392344

Title = K205R specific nanobody-horseradish peroxidase fusions as reagents of competitive ELISA to detect African swine fever virus serum antibodies

African swine fever virus (ASFV) is a highly contagious hemorrhagic disease and often lethal, which has significant economic consequences for the swine industry. Due to lacking of commercial vaccine, the prevention and control of ASF largely depend on early large-scale detection and screening. So far, the commercial ELISA kits have a long operation time and are expensive, making it difficult to achieve large-scale clinical applications. Nanobodies are single-domain antibodies produced by camelid animals, and have unique advantages such as smaller molecular weight, easy genetic engineering modification and low-costing of mass production, thus exhibiting good application prospects.

The present study developed a new method for detection of ASFV specific antibodies using nanobody-horseradish peroxidase (Nb-HRP) fusion proteins as probe. By using camel immunization, phage library construction and phage display technology, five nanobodies against K205R protein were screened. Then, Nb-HRP fusion proteins were produced using genetic modification technology. Based on the Nb-HRP fusion protein as specific antibodies against K205R protein, a new type of cELISA was established to detect ASFV antibodies in pig serum. The cut-off value of the cELISA was 34.8%, and its sensitivity, specificity, and reproducibility were good. Furthermore, the developed cELISA exhibited 99.3% agreement rate with the commercial available ELISA kit (kappa value = 0.98).

The developed cELISA method has the advantages of simple operation, rapid and low-costing, and can be used for monitoring of ASFV infection in pigs, thus providing a new method for the prevention and control of ASF.

The online version contains supplementary material available at 10.1186/s12917-022-03423-0.

[end paper 100]



