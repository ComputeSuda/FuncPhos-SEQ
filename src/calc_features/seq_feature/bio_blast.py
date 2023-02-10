from Bio.Blast import NCBIWWW
from Bio.Blast import NCBIXML
from Bio import SeqIO
from prody import *
from pylab import *
from Bio import AlignIO
from Bio.PDB.PDBParser import PDBParser
import os
import time


# Searching sequence by blast

# ion()
input_path = '../Datasets/fasta/uniprot_fastas'  # Original protein sequence file
output_path = './uniprot_blast/'  # Protein sequence file after blast
uniprot_files = os.listdir(input_path)

print(uniprot_files[-1::-1])
print(len(uniprot_files))

for uniprot_file in uniprot_files:
    print(uniprot_file)
    time.sleep(60)
    print('time over!')
    record = SeqIO.read(input_path + '/' + uniprot_file, format="fasta")
    result_handle = NCBIWWW.qblast("blastp", "nr", record.seq, hitlist_size=500)
    blast_record = NCBIXML.read(result_handle)
    for alignment in blast_record.alignments:
        for hsp in alignment.hsps:
            txt_file = open(output_path + uniprot_file[:6] + '_blast.fasta', 'a', encoding='utf-8')
            txt_file.write('>' + alignment.title)
            txt_file.write('\n')
            txt_file.write(hsp.sbjct)
            txt_file.write('\n')
            txt_file.close()







