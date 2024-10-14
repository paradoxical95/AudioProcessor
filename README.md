# AudioProcessor
**Apply EQ and Reverb to an audio.wav file**

On Linux Terminal or x64 Windows VS Terminal or any other equivalent, write
To compile from Source (if you have CUDA Toolkit/nvcc installed) ->
**_nvcc -o audio_processor audio_processor.cu_**

To run ->
(Linux) ->  _**./audio_processor input.wav**_

(For Windows) -> _**audio_processor.exe input.wav**_

Where 'input.wav' is your input file, ideally to be placed next to the .cu/.exe file. Output would be generated in the same directory named "output.wav"

Still a work in progress and may not work properly for all WAV files. In my case, it fails on converted WAVs and at times makes an empty file on natively exported WAV files.
