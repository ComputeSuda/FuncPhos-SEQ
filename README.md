# FuncPhos-SEQ
FuncPhos-SEQ: a deep learning method for phosphosite functional prioritization was proposed based on protein sequence and PPI information.The method consisted of two feature encoding sub-networks (SeqNet and SPNet) and a feature combination sub-network (CoNet).where the SeqNet module encodes protein sequences and extract the phosphosite sequence features, the SPNet module is used to integrate the protein feature extracted from the PPI network network topologies, and the CoNet module serves to integrate and calculate the features obtained from both SeqNet and SPNet modules to identify functional phosphosites.
![image](https://github.com/ComputeSuda/FuncPhos-SEQ/blob/main/IMG/model.png)

# System requirement
FuncPhos-SEQ is develpoed under Linux environment with:
* Python (3.10.4):
    - keras==2.8.0
    - networkx==2.6.3
    - scipy==1.7.3
    - scikit-learn==0.24.2
    - numpy==1.23.3
    - tensorflow==2.8.0
    - biopython==1.78
    - prody==2.0
