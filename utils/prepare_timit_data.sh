# timit dir
dir=/nfs/corpus0/data/corpora/database/speech/asr/timit/original
mkdir -p data_timit/{train,test}/audio
cd data_timit

for j in train test; do
  for i in $dir/$j/*/*/*.wav; do
    id=$(echo $i | rev | cut -d '/' -f 1-2 | rev | sed 's/\//_/g' | sed 's/.wav//g')

    # wav
    echo $id $i >> $j/sph.scp
    cmd="/freeneb/home/tangzy/freekaldi/egs/timit/s5/../../../tools/sph2pipe_v2.5/sph2pipe -f wav $i $j/audio/${id}.wav"
    $cmd
    echo $id data_timit/$j/audio/${id}.wav >> $j/wav.scp

    # txt
    txt=$(echo $i | sed 's/.wav/.txt/')
    awk '{print id,$0}' id=$id $txt >> $j/text

    # wrd
    wrd=$(echo $i | sed 's/.wav/.wrd/')
    awk '{print id,$0}' id=$id $wrd >> $j/word.seg

    # phn
    phn=$(echo $i | sed 's/.wav/.phn/')
    awk '{print id,wav,$0}' id=$id wav=data_timit/$j/audio/${id}.wav $phn >> $j/phone.seg

    # spk
    spk=$(echo $i | rev | cut -d '/' -f 2 | rev)
    echo $id $spk >> $j/utt2spk
  done

  # not use dialect (_sa) and silence, length between 1000 to 3000
  grep -v -E '_sa|h#' $j/phone.seg | awk '$4-$3>1000' | awk '$4-$3<3000' > $j/phone.seg.selected
  grep -v -E '_sa|h#' $j/utt2spk > $j/utt2spk.selected
done
