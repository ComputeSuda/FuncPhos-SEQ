# -*- coding: utf-8 -*-


from prody import *
from pylab import *
from numpy import *

uniprot = {'P24941':[1,298], 'O14733':[101,400], 'P54646':[11,279], 'Q07912':[117,391], 'P28482':[12,357], 'P35557':[12,458], 'P43403':[1,257], 'O14965':[127,391],
           'P49674':[1,294], 'P06493':[1,297], 'Q8IVH8':[110,210], 'P14618':[31,431], 'O43781':[138,532], 'P31749':[144,477], 'P31751':[146,477], 'Q9NWZ3':[164,458],
           'Q13882':[185,443], 'P27707':[19,260], 'P51617':[194,526], 'Q13131':[20,505], 'O96017':[210,510], 'P22392':[2,152], 'Q00535':[2,292], 'P00519':[232,500],
           'Q96KB5':[23,321], 'Q14680':[2,335], 'Q13153':[249,543], 'P41743':[249,587], 'P27361':[25,374], 'O96013':[299,591], 'Q9H8S9':[31,213], 'O43318':[31,332],
           'O60825':[31,455], 'P15531':[3,151], 'Q9UQB9':[32,303], 'P78356':[32,416], 'P51955':[3,273], 'P60891':[3,313], 'Q99558':[332,672], 'Q13976':[337,671],
           'P00558':[3,417], 'Q16566':[34,321], 'Q08881':[357,615], 'P49841':[36,382], 'P43405':[364,633], 'P53350':[370,593], 'Q9BUB5':[37,334], 'Q04759':[374,706],
           'Q9NQU5':[385,671], 'Q06187':[389,659], 'Q15118':[41,423], 'Q05397':[414,686], 'Q16539':[4,353], 'Q16875':[33,446], 'P07332':[450,821], 'Q7KZI7':[46,363],
           'P51812':[49,346], 'Q9UK32':[51,349], 'P33981':[515,795], 'O43353':[5,310], 'Q15759':[5,348], 'O75460':[562,964], 'P36888':[569,896], 'Q16512':[605,940],
           'Q92918':[6,302], 'P07949':[705,1013], 'Q96GD4':[71,339], 'Q9NYV4':[718,1048], 'Q13546':[7,294], 'O95747':[7,295], 'P41279':[73,389], 'O43683':[735,1083],
           'O15530':[76,359], 'P35790':[80,457], 'P52333':[815,1098], 'O00141':[82,374], 'Q15208':[82,414], 'O75385':[8,281], 'Q9NYL2':[8,304], 'P50750':[8,326],
           'O43293':[9,284], 'O14920':[94,294], 'P00533':[696,1020], 'P12931':[254,532]}
           
#calculate sequence features
for i in uniprot:
    msa = parseMSA(i + "_omega_aln.fasta")
    with open(i + "_blast.txt", "r") as f:
        data = f.readlines()
        
    msa_refine = refineMSA(msa, label=data[0].split('|')[1], rowocc=0.7, seqid=0.996)
    msa_refine_1 = msa_refine[ : , (uniprot[i][0] - 1):(uniprot[i][1])]
    print(i)
    occupancy = calcMSAOccupancy(msa_refine_1, occ = 'res')
    name_1 = '/home/lzj/jxhe/evol/output_label/' + i + '_occupancy.txt'
    np.savetxt(name_1, occupancy)
    
    entropy = calcShannonEntropy(msa_refine_1)
    name_2 ='/home/lzj/jxhe/evol/output_label/' + i + '_entropy.txt'
    np.savetxt(name_2, entropy)
    
    mutinfo = buildMutinfoMatrix(msa_refine_1)
    name_3 ='/home/lzj/jxhe/evol/output_label/' + i + '_mutinfo.txt'
    writeArray(name_3, mutinfo)
    
    omes = buildOMESMatrix(msa_refine_1)
    name_4 ='/home/lzj/jxhe/evol/output_label/' + i + '_omes.txt'
    writeArray(name_4, omes)
    
    sca = buildSCAMatrix(msa_refine_1)
    name_5 ='/home/lzj/jxhe/evol/output_label/' + i + '_sca.txt'
    writeArray(name_5, sca)
    
    di = buildDirectInfoMatrix(msa_refine_1)
    name_6 ='/home/lzj/jxhe/evol/output_label/' + i + '_di.txt'
    writeArray(name_6, di)
    
    seqid = buildSeqidMatrix(msa_refine_1)
    name_7 ='/home/lzj/jxhe/evol/output_label/' + i + '_seqid.txt'
    writeArray(name_7, seqid)
    
    mutinfo_norm = applyMutinfoNorm(mutinfo, entropy, norm='minent')
    mutinfo_corr = applyMutinfoCorr(mutinfo, corr='apc')
    name_8 ='/home/lzj/jxhe/evol/output_label/' + i + '_mutinfo_norm.txt'
    writeArray(name_8, mutinfo_norm)
    name_9 ='/home/lzj/jxhe/evol/output_label/' + i + '_mutinfo_corr.txt'
    writeArray(name_9, mutinfo_corr)
    
    
    
    
    
    
    
    
    
    
    