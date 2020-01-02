# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************
import argparse
import json
import os
import torch
import shutil

from glow import WaveGlow, WaveGlowLoss
from mel2samp import scp_files_to_list, MAX_WAV_VALUE, load_wav_to_torch
from scipy.io.wavfile import write


def train(output_directory, epochs, learning_rate,
          sigma, seed, checkpoint_path):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    criterion = WaveGlowLoss(sigma)

    # Load checkpoint
    model = torch.load(checkpoint_path)['model']
    print("Loaded checkpoint '{}'" .format(checkpoint_path))
    # Set false grad, save computation
    for param in model.parameters():
        param.requires_grad = False

    model.cuda().train()
    epoch_offset = 0
    # ================ MAIN TRAINNIG LOOP! ===================
    wav_scp = scp_files_to_list(data_config["training_files"])
    for _, file_scp in enumerate(wav_scp):
        file_name, file_path = file_scp
        audio, sampling_rate = load_wav_to_torch(file_path)
        audio = torch.unsqueeze(audio, 0)
        audio = audio / MAX_WAV_VALUE  # trained with norm

        for epoch in range(epoch_offset, epochs):
            # compute grad on input
            audio = torch.autograd.Variable(audio.cuda(), requires_grad=True)

            model.zero_grad()
            outputs = model(audio)

            loss = criterion(outputs)
            loss.backward()
            print("Epoch {}:\t{:.9f}".format(epoch, loss.item()))

            # optimizer.step() not needed actually, update input manually
            with torch.no_grad():
                # after this, audio.grad becomes None, otherwise audio.grad.zero_()
                audio = audio - learning_rate * audio.grad

            # save the file
            audio_new = audio * MAX_WAV_VALUE
            audio_new = audio_new.squeeze()
            audio_new = audio_new.cpu().detach().numpy()
            audio_new = audio_new.astype('int16')
            audio_path = os.path.join(
                output_directory, "{}_lr{}_ep{}.wav".format(file_name, learning_rate, epoch))
            write(audio_path, sampling_rate, audio_new)
            print(audio_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')
    args = parser.parse_args()

    # Parse configs.  Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    train_config = config["train_config"]
    global data_config
    data_config = config["data_config"]

    # Store config file.
    output_directory = train_config["output_directory"]
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    print("output directory", output_directory)
    shutil.copy2(args.config, output_directory)


    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    train(**train_config)
