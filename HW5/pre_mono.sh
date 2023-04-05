binpath="/neodata/ML/hw5_dataset/data-bin/mono"
src_dict_file='/neodata/ML/hw5_dataset/data-bin/ted2020/dict.en.txt'
tgt_dict_file=$src_dict_file
monopref="/neodata/ML/hw5_dataset/rawdata/mono/mono.tok" # whatever filepath you get after applying subword tokenization

python -m fairseq_cli.preprocess\
	--source-lang 'zh'\
	--target-lang 'en'\
	--trainpref ${monopref}\
	--destdir ${binpath}\
	--srcdict ${src_dict_file}\
	--tgtdict ${tgt_dict_file}\
	--workers 2

cp /neodata/ML/hw5_dataset/data-bin/mono/train.zh-en.zh.bin /neodata/ML/hw5_dataset/data-bin/ted2020/mono.zh-en.zh.bin
cp /neodata/ML/hw5_dataset/data-bin/mono/train.zh-en.zh.idx /neodata/ML/hw5_dataset/data-bin/ted2020/mono.zh-en.zh.idx
cp /neodata/ML/hw5_dataset/data-bin/mono/train.zh-en.en.bin /neodata/ML/hw5_dataset/data-bin/ted2020/mono.zh-en.en.bin
cp /neodata/ML/hw5_dataset/data-bin/mono/train.zh-en.en.idx /neodata/ML/hw5_dataset/data-bin/ted2020/mono.zh-en.en.idx

