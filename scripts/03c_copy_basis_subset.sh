base_n=10

for FLAG in TRAIN800 TEST5646 FS16
do
    basis=/kaggle/working/CF-Font/subset.txt
    in_folder=data/imgs/Seen240_S80F50_${FLAG}
    out_folder=data/imgs/BASIS_S80F50_${FLAG}

    python scripts/basis/copy_basis_imgs.py -b ${basis} -i ${in_folder} -o ${out_folder}
done