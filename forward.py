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
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************
import os
from scipy.io.wavfile import write
import torch
from mel2samp import scp_files_to_list, MAX_WAV_VALUE, load_wav_to_torch
from denoiser import Denoiser
import matplotlib.pyplot as plt
from glow import WaveGlowLoss


def main(wav_scp, waveglow_path, sigma, output_dir, sampling_rate, is_fp16,
         denoiser_strength):
    wav_scp = scp_files_to_list(wav_scp)
    waveglow = torch.load(waveglow_path)['model']
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow.cuda().eval()
    if is_fp16:
        from apex import amp
        waveglow, _ = amp.initialize(waveglow, [], opt_level="O3")

    if denoiser_strength > 0:
        denoiser = Denoiser(waveglow).cuda()

    criterion = WaveGlowLoss(sigma)

    z_all = []
    for _, file_scp in enumerate(wav_scp):
        file_name, file_path = file_scp
        audio, sampling_rate = load_wav_to_torch(file_path)
        audio = torch.autograd.Variable(audio.cuda())
        audio = torch.unsqueeze(audio, 0)
        audio = audio / MAX_WAV_VALUE  # trained with norm
        with torch.no_grad():
            z = waveglow.forward(audio)
        
        loss = criterion(z, details=True)
        print("{} loss, log_p_z, log_det: \t{:.9f}, {:.9f}, {:.9f}".format(file_name, 
            loss[0].item(), loss[1].item(), loss[2].item()))

        z_path = os.path.join(output_dir, "{}_z".format(file_name))
        torch.save(z[0], z_path)
        print(z_path)
        z_all.append(z[0])

    z_all_a = torch.cat(z_all, 2)
    plt.plot(z_all_a[0, 0, :].cpu().numpy(), z_all_a[0, 1, :].cpu().numpy(), 'r.')
    z_fig_path = os.path.join(output_dir, "{}.pdf".format("z_dim-1-2"))
    plt.savefig(z_fig_path)
    print(z_fig_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--filelist_path", required=True)
    parser.add_argument('-w', '--waveglow_path',
                        help='Path to waveglow decoder checkpoint with model')
    parser.add_argument('-o', "--output_dir", default="output_z")
    parser.add_argument("-s", "--sigma", default=1.0, type=float)
    parser.add_argument("--sampling_rate", default=16000, type=int)
    parser.add_argument("--is_fp16", action="store_true")
    parser.add_argument("-d", "--denoiser_strength", default=0.0, type=float,
                        help='Removes model bias. Start with 0.1 and adjust')

    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    main(args.filelist_path, args.waveglow_path, args.sigma, args.output_dir,
         args.sampling_rate, args.is_fp16, args.denoiser_strength)
