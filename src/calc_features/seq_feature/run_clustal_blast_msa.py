# coding=UTF-8
import os

'''
Multiple sequence alignment of the blasted fasta file of input_path 
Results Save to blast/uniprot_pdb_aln folder, file name is xxx_blast_omega_aln.fasta 
'''

input_path = '../blast/uniprot_blast'
output_path = './uniprot_aln'
input_file_names = os.listdir(input_path)
get_items = [item.split('.')[0][:6]+'_blast.'+item.split('.')[1] for item in os.listdir(output_path)]
# file_name = 'P60174_blast.fasta'
print(len(get_items), len(input_file_names))
i = 0
for file_name in input_file_names:
    if file_name not in get_items:
        # # for i in [0]:
        print(file_name)
        input_file = input_path + '/' + file_name
        # input_file = 'O00429_blast.fasta'
        output_file = output_path + '/' + file_name[:6] + '_blast_omega_aln.fasta'
        # output_file = 'O00429_aln.fasta'
        # cmdr = os.system('clustalo -i %s --seqtype=Proteins_pairs -o %s'% (input_file, output_file))  # linux
        cmdr = os.system(
            'clustalo.exe -i %s --seqtype=Proteins_pairs -o %s --force --dealign' % (input_file, output_file))  # windows
        i += 1
        print(i)
        # print(file_name)
        if cmdr != 0:  # If there is error printing information
            print(file_name)
            print(i)

