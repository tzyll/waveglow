# first get data dir with Kaldi wsj recipe, cp data data_wsj
dir=data_wsj

# train/test: wv1 to wav
for i in train_si284 test_dev93 test_eval92 test_eval93; do
    subdir=$dir/$i
    mkdir -p $subdir/audio
    echo "Process $subdir"
    cat $subdir/wav.scp | while read line; do
        id=`echo $line | cut -d ' ' -f 1`
        cmd=`echo $line | cut -d ' ' -f 2- | cut -d '|' -f 1`
        cmd="$cmd $subdir/audio/${id}.wav"
        $cmd
        echo $cmd
        echo "$id $subdir/audio/${id}.wav" >> $subdir/wav.scp.new
    done
    mv $subdir/wav.scp $subdir/wav.scp.org
    mv $subdir/wav.scp.new $subdir/wav.scp
done

echo "Done."
exit 0