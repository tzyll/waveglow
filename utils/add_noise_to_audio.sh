#!/bin/bash

if [ $# != 1 ]; then
  echo "Usage: $0 test.wav"
  exit 1
fi

input=$1
dir=test_noise
mkdir -p $dir


# Use sox to add noise, no SNR, for more usage, see
# http://linguistics.berkeley.edu/plab/guestwiki/index.php?title=Sox_in_phonetic_research
vol=0.005
output=$dir/${input%.wav*}_sox_whitenoise_vol${vol}.wav
id=${input%.wav*}_sox_whitenoise_vol${vol}
sox $input -p synth whitenoise vol $vol | sox -m $input - $output
echo $id $output > test_noise.scp
echo "====> $output"


# Use Kaldi/$tool, more info see
# https://gitlab.com/tzyll/freekaldi/blob/master/egs/freeneb/tools/speech_data_augment/run_b.sh
tool=~/freekaldi/src/featbin/wav-reverberate

# reverberate
output=$dir/${input%.wav*}_kaldi_reverb.wav
id=${input%.wav*}_kaldi_reverb
noise=/work102/tangzy/freekaldi/egs/freeneb/tools/20181101_speech_data_augment/RIRS_NOISES/simulated_rirs/smallroom/Room123/Room123-00036.wav
cat $input | $tool --shift-output=true --impulse-response="sox $noise -r 16000 -t wav - |" - $output
echo $id $output >> test_noise.scp
echo "====> $output"

# add noise
snr=0
output=$dir/${input%.wav*}_kaldi_noise_snr$snr.wav
id=${input%.wav*}_kaldi_noise_snr$snr
noise=/nfs/corpus0/data/corpora/database/acoustic/noise/musan/noise/free-sound/noise-free-sound-0089.wav
$tool --shift-output=true --additive-signals="sox -t wav $noise -r 16k -t wav - |" --start-times='0' --snrs="$snr" $input $output
echo $id $output >> test_noise.scp
echo "====> $output"

# add music
snr=5
output=$dir/${input%.wav*}_kaldi_music_snr$snr.wav
id=${input%.wav*}_kaldi_music_snr$snr
noise=/nfs/corpus0/data/corpora/database/acoustic/noise/musan/music/jamendo/music-jamendo-0129.wav
$tool --shift-output=true --additive-signals="sox -t wav $noise -r 16k -t wav - |" --start-times='0' --snrs="$snr" $input $output
echo $id $output >> test_noise.scp
echo "====> $output"

# add babble
snr=15
output=$dir/${input%.wav*}_kaldi_babble_snr$snr.wav
id=${input%.wav*}_kaldi_babble_snr$snr
noise=/nfs/corpus0/data/corpora/database/acoustic/noise/musan/speech/us-gov/speech-us-gov-0220.wav
$tool --shift-output=true --additive-signals="sox -t wav $noise -r 16k -t wav - |" --start-times='0' --snrs="$snr" $input $output
echo $id $output >> test_noise.scp
echo "====> $output"

echo "== See test_noise.scp =="