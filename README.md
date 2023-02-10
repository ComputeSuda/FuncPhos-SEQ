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
* You can install the dependent packages by the following commands:
    - pip install python==3.10.4
    - pip install numpy==1.23.3
    - pip install keras==2.8.0
    - pip install tensorflow==2.8.0
# Dataset
We provide phosphosite data, collected from five databases - PSP, EPSD, PLMD, IPTMNet and PTMD - detailing information on phosphosites and their regulation of molecular functions, biological processes and intermolecular interactions.

# predict test data
If you want to use the model to predict phosphorylation site function,run the following command(The test data required to run can be found in the Datasets/example folder; you can also provide your own data, which needs to be in the same format as the test data))：
python ./src/model/FuncPhos_SEQ.py
