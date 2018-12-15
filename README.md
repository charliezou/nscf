#Environments

- python 3.6
- keras 2.2.2


# run simple scf

python SimpleSCF.py --dataset ml-1m --epochs 20 --batch_size 256 --num_factors 64 --num_neg 32 --seq_len 4 --gt 1 --lr 0.001 --learner adam --verbose 1 --out 0

# run conv scf

python ConvSCF.py --dataset ml-1m --epochs 20 --batch_size 256 --num_factors 64 --num_neg 32 --seq_len 12 --gt 1 --lr 0.001 --learner adam --verbose 1 --out 0


# run attention scf

python AttentionSCF.py --dataset ml-1m --epochs 20 --batch_size 256 --num_factors 64 --num_neg 32 --seq_len 12 --gt 1 --lr 0.001 --learner adam --verbose 1 --out 0


# run union scf

python UnionSCF.py --dataset ml-1m --epochs 10 --batch_size 256 --num_factors 64 --num_neg 4 --seq_len [12,12,4] --num_layer [2,2,1] --ascf_pretrain ml-1m_ascf_8_1_12_2_32_0.8293_0.6188.h5 --cscf_pretrain ml-1m_cscf_12_1_12_2_32_0.8228_0.6080.h5 --sscf_pretrain ml-1m_sscf_8_1_4_1_32_0.8174_0.5999.h5 --alpha [0.45,0.35,0.20] --lr 0.0001 --learner adam --verbose 1 --out 0

