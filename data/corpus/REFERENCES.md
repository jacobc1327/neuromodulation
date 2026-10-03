# rTMS literature corpus

42 studies (41 peer-reviewed; 1 preprint, labeled). Each record in [`studies.jsonl`](studies.jsonl) has verified citation metadata (title, authors, journal, year, DOI and PMID where confirmed), structured fields (design, n, target, protocol) and a paraphrased summary. The `verification_sources` field lists the pages each record was checked against. Run `python scripts/fetch_pubmed.py --resolve-doi` on a networked machine to add the official PubMed abstracts.

## PTSD / veterans (12)

- **Philip et al., 2025**: [Pragmatic accelerated transcranial magnetic stimulation for posttraumatic stress disorder](https://doi.org/10.1016/j.brs.2025.05.007). *Brain Stimulation*. observational, n=123, left DLPFC (scalp-based, modified Beam/F3).
- **Brown et al., 2024**: [Repetitive transcranial magnetic stimulation for post-traumatic stress disorder in adults](https://doi.org/10.1002/14651858.CD015040.pub2). *Cochrane Database of Systematic Reviews*. systematic review, n=577, various.
- **Young et al., 2024**: [Multimodal smoking cessation treatment combining repetitive transcranial magnetic stimulation, cognitive behavioral therapy, and nicotine replacement in veterans with posttraumatic stress disorder: A feasibility randomized controlled trial protocol](https://doi.org/10.1371/journal.pone.0291562). *PLoS ONE*. trial protocol, n=50, right post-central gyrus region most functionally connected to the right posterior insula (individualized, MRI-guided).
- **Madore et al., 2022**: [Prefrontal transcranial magnetic stimulation for depression in US military veterans - A naturalistic cohort study in the veterans health administration](https://doi.org/10.1016/j.jad.2021.10.025). *Journal of Affective Disorders*. observational, n=770, left DLPFC.
- **Kan et al., 2020**: [Non-invasive brain stimulation for posttraumatic stress disorder: a systematic review and meta-analysis](https://doi.org/10.1038/s41398-020-0851-5). *Translational Psychiatry*. meta-analysis, various (subgroup analyses of right DLPFC).
- **Kozel et al., 2019**: [One hertz versus ten hertz repetitive TMS treatment of PTSD: A randomized clinical trial](https://doi.org/10.1016/j.psychres.2019.01.004). *Psychiatry Research*. RCT, n=44, right prefrontal cortex.
- **Philip et al., 2019**: [Theta-Burst Transcranial Magnetic Stimulation for Posttraumatic Stress Disorder](https://doi.org/10.1176/appi.ajp.2019.18101160). *American Journal of Psychiatry*. RCT, n=50, right DLPFC.
- **Kozel et al., 2018**: [Repetitive TMS to augment cognitive processing therapy in combat veterans of recent conflicts with PTSD: A randomized clinical trial](https://doi.org/10.1016/j.jad.2017.12.046). *Journal of Affective Disorders*. RCT, n=103, right DLPFC.
- **Philip et al., 2016**: [5-Hz Transcranial Magnetic Stimulation for Comorbid Posttraumatic Stress Disorder and Major Depression](https://doi.org/10.1002/jts.22065). *Journal of Traumatic Stress*. observational, n=10.
- **Isserles et al., 2013**: [Effectiveness of deep transcranial magnetic stimulation combined with a brief exposure procedure in post-traumatic stress disorder - a pilot study](https://doi.org/10.1016/j.brs.2012.07.008). *Brain Stimulation*. pilot RCT, n=30, medial prefrontal cortex (deep TMS, H-coil).
- **Boggio et al., 2010**: [Noninvasive brain stimulation with high-frequency and low-intensity repetitive transcranial magnetic stimulation treatment for posttraumatic stress disorder](https://doi.org/10.4088/JCP.08m04638blu). *Journal of Clinical Psychiatry*. RCT, n=30, right DLPFC vs left DLPFC.
- **Cohen et al., 2004**: [Repetitive Transcranial Magnetic Stimulation of the Right Dorsolateral Prefrontal Cortex in Posttraumatic Stress Disorder: A Double-Blind, Placebo-Controlled Study](https://doi.org/10.1176/appi.ajp.161.3.515). *American Journal of Psychiatry*. RCT, n=24, right DLPFC.

## Alcohol (8)

- **Kim et al., 2025**: [Efficacy of non-invasive brain stimulation in reducing craving in patients with alcohol use disorder: systematic review and meta-analysis](https://doi.org/10.1186/s12888-025-06883-4). *BMC Psychiatry*. meta-analysis, various; DLPFC subgroup.
- **Hoven et al., 2023**: [Effects of 10 add-on HF-rTMS treatment sessions on alcohol use and craving among detoxified inpatients with alcohol use disorder: a randomized sham-controlled clinical trial](https://doi.org/10.1111/add.16025). *Addiction*. RCT, n=80, right DLPFC.
- **McCalley et al., 2023**: [Medial Prefrontal Cortex Theta Burst Stimulation Improves Treatment Outcomes in Alcohol Use Disorder: A Double-Blind, Sham-Controlled Neuroimaging Study](https://doi.org/10.1016/j.bpsgos.2022.03.002). *Biological Psychiatry: Global Open Science*. RCT, n=50, left frontal pole / medial prefrontal cortex.
- **Harel et al., 2022**: [Repetitive transcranial magnetic stimulation in alcohol dependence: a randomized, double-blind, sham-controlled proof-of-concept trial targeting the medial prefrontal and anterior cingulate cortices](https://doi.org/10.1016/j.biopsych.2021.11.020). *Biological Psychiatry*. RCT, n=51, medial prefrontal cortex and anterior cingulate cortex (midline frontocortical).
- **Belgers et al., 2022**: [rTMS Reduces Craving and Alcohol Use in Patients with Alcohol Use Disorder: Results of a Randomized, Sham-Controlled Clinical Trial](https://doi.org/10.3390/jcm11040951). *Journal of Clinical Medicine*. RCT, n=30, right DLPFC (F4).
- **Addolorato et al., 2017**: [Deep transcranial magnetic stimulation of the dorsolateral prefrontal cortex in alcohol use disorder patients: effects on dopamine transporter availability and alcohol intake](https://doi.org/10.1016/j.euroneuro.2017.03.008). *European Neuropsychopharmacology*. pilot RCT, n=14, DLPFC.
- **Ceccanti et al., 2015**: [Deep TMS on alcoholics: effects on cortisolemia and dopamine pathway modulation. A pilot study](https://doi.org/10.1139/cjpp-2014-0330). *Canadian Journal of Physiology and Pharmacology*. pilot RCT, n=18, medial prefrontal cortex.
- **Mishra et al., 2010**: [Efficacy of repetitive transcranial magnetic stimulation in alcohol dependence: a sham-controlled study](https://doi.org/10.1111/j.1360-0443.2009.02777.x). *Addiction*. RCT, n=45, right DLPFC.

## Nicotine (3)

- **Zangen et al., 2021**: [Repetitive transcranial magnetic stimulation for smoking cessation: a pivotal multicenter double-blind randomized controlled trial](https://doi.org/10.1002/wps.20905). *World Psychiatry*. RCT, n=262, bilateral lateral prefrontal and insular cortices.
- **Dinur-Klein et al., 2014**: [Smoking cessation induced by deep repetitive transcranial magnetic stimulation of the prefrontal and insular cortices: a prospective, randomized controlled trial](https://doi.org/10.1016/j.biopsych.2014.05.020). *Biological Psychiatry*. RCT, n=115, bilateral lateral prefrontal cortex and insula.
- **Li et al., 2013**: [Repetitive transcranial magnetic stimulation of the dorsolateral prefrontal cortex reduces nicotine cue craving](https://doi.org/10.1016/j.biopsych.2013.01.003). *Biological Psychiatry*. crossover, n=16, left DLPFC.

## Stimulants (9)

- **Zhao et al., 2023**: [Deep magnetic stimulation targeting the medial prefrontal and anterior cingulate cortices for methamphetamine use disorder: a randomised, double-blind, sham-controlled study](https://doi.org/10.1136/gpsych-2023-101149). *General Psychiatry*. RCT, n=23, medial PFC and anterior cingulate cortex (deep TMS, H7-coil).
- **Lolli et al., 2021**: [A randomised, double-blind, sham-controlled study of left prefrontal cortex 15 Hz repetitive transcranial magnetic stimulation in cocaine consumption and craving](https://doi.org/10.1371/journal.pone.0259860). *PLoS ONE*. RCT, n=62, left DLPFC (5 cm method).
- **Chen et al., 2020**: [The exploration of optimized protocol for repetitive transcranial magnetic stimulation in the treatment of methamphetamine use disorder: A randomized sham-controlled study](https://doi.org/10.1016/j.ebiom.2020.103027). *EBioMedicine*. RCT, n=74, left DLPFC (iTBS) and/or left vmPFC (cTBS).
- **Ma et al., 2019**: [Effects of Non-invasive Brain Stimulation on Stimulant Craving in Users of Cocaine, Amphetamine, or Methamphetamine: A Systematic Review and Meta-Analysis](https://doi.org/10.3389/fnins.2019.01095). *Frontiers in Neuroscience*. meta-analysis, n=321, mainly left DLPFC (rTMS and tDCS).
- **Kearney-Ramos et al., 2019**: [State-Dependent Effects of Ventromedial Prefrontal Cortex Continuous Thetaburst Stimulation on Cocaine Cue Reactivity in Chronic Cocaine Users](https://doi.org/10.3389/fpsyt.2019.00317). *Frontiers in Psychiatry*. crossover, n=19, medial/ventromedial prefrontal cortex.
- **Liang et al., 2018**: [Targeting Withdrawal Symptoms in Men Addicted to Methamphetamine With Transcranial Magnetic Stimulation: A Randomized Clinical Trial](https://doi.org/10.1001/jamapsychiatry.2018.2383). *JAMA Psychiatry*. RCT, n=48, left DLPFC.
- **Su et al., 2017**: [High frequency repetitive transcranial magnetic stimulation of the left dorsolateral prefrontal cortex for methamphetamine use disorders: A randomised clinical trial](https://doi.org/10.1016/j.drugalcdep.2017.01.037). *Drug and Alcohol Dependence*. RCT, n=30, left DLPFC.
- **Terraneo et al., 2016**: [Transcranial magnetic stimulation of dorsolateral prefrontal cortex reduces cocaine use: A pilot study](https://doi.org/10.1016/j.euroneuro.2015.11.011). *European Neuropsychopharmacology*. pilot RCT, n=32, left DLPFC.
- **Bolloni et al., 2016**: [Bilateral Transcranial Magnetic Stimulation of the Prefrontal Cortex Reduces Cocaine Intake: A Pilot Study](https://doi.org/10.3389/fpsyt.2016.00133). *Frontiers in Psychiatry*. pilot RCT, n=10, bilateral prefrontal cortex (deep TMS, H1-coil).

## Opioids (1)

- **Shen et al., 2016**: [10-Hz Repetitive Transcranial Magnetic Stimulation of the Left Dorsolateral Prefrontal Cortex Reduces Heroin Cue Craving in Long-Term Addicts](https://doi.org/10.1016/j.biopsych.2016.02.006). *Biological Psychiatry*. RCT, n=20, left DLPFC.

## SUD (multiple) (5)

- **Soleimani et al., 2025**: [Effectiveness of Noninvasive Brain Stimulation Protocols on Drug Craving and Consumption/Relapse in Substance Use Disorders: A Systematic Review and Meta-analysis of 208 Clinical Trials and 36 Protocols](https://doi.org/10.1101/2025.09.21.25335559). *medRxiv (preprint, not peer-reviewed)*. preprint meta-analysis, various (36 protocols); strongest for H4-coil deep TMS and HF rTMS of left DLPFC.
- **Mehta et al., 2024**: [A systematic review and meta-analysis of neuromodulation therapies for substance use disorders](https://doi.org/10.1038/s41386-023-01776-0). *Neuropsychopharmacology*. meta-analysis, various; strongest rTMS effects with left DLPFC.
- **Antonelli et al., 2021**: [Transcranial Magnetic Stimulation: A review about its efficacy in the treatment of alcohol, tobacco and cocaine addiction](https://doi.org/10.1016/j.addbeh.2020.106760). *Addictive Behaviors*. systematic review, various (mainly prefrontal).
- **Ekhtiari et al., 2019**: [Transcranial electrical and magnetic stimulation (tES and TMS) for addiction medicine: A consensus paper on the present state of the science and the road ahead](https://doi.org/10.1016/j.neubiorev.2019.06.007). *Neuroscience and Biobehavioral Reviews*. consensus/guideline, various (tES and TMS targets in SUD research).
- **Jansen et al., 2013**: [Effects of non-invasive neurostimulation on craving: a meta-analysis](https://doi.org/10.1016/j.neubiorev.2013.07.009). *Neuroscience and Biobehavioral Reviews*. meta-analysis, DLPFC.

## Depression (trial design) (3)

- **Cole et al., 2022**: [Stanford Neuromodulation Therapy (SNT): A Double-Blind Randomized Controlled Trial](https://doi.org/10.1176/appi.ajp.2021.20101429). *American Journal of Psychiatry*. RCT, n=29, left DLPFC (individualized with functional connectivity MRI).
- **Yesavage et al., 2018**: [Effect of Repetitive Transcranial Magnetic Stimulation on Treatment-Resistant Major Depression in US Veterans: A Randomized Clinical Trial](https://doi.org/10.1001/jamapsychiatry.2018.1483). *JAMA Psychiatry*. RCT, n=164, left prefrontal cortex.
- **Blumberger et al., 2018**: [Effectiveness of theta burst versus high-frequency repetitive transcranial magnetic stimulation in patients with depression (THREE-D): a randomised non-inferiority trial](https://doi.org/10.1016/S0140-6736(18)30295-2). *The Lancet*. RCT, n=414, left DLPFC (inferred; not stated in sources read).

## Guidelines (1)

- **Lefaucheur et al., 2020**: [Evidence-based guidelines on the therapeutic use of repetitive transcranial magnetic stimulation (rTMS): An update (2014-2018)](https://doi.org/10.1016/j.clinph.2019.11.002). *Clinical Neurophysiology*. consensus/guideline, multiple (M1, left DLPFC, others).
