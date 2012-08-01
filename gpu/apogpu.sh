#export COMPUTE_PROFILE=1

./apogpu_chained_512 500000
./apogpu_chained_512 750000
./apogpu_chained_512 1000000
./apogpu_chained_512 2500000
./apogpu_chained_512 5000000
./apogpu_chained_512 7500000
./apogpu_chained_512 10000000
./apogpu_chained_512 12500000
./apogpu_chained_512 15000000
./apogpu_chained_512 17500000
./apogpu_chained_512 20000000

#./apogpu_delay_512 500000
#./apogpu_delay_512 750000
#./apogpu_delay_512 1000000
#./apogpu_delay_512 2500000
#./apogpu_delay_512 5000000
#./apogpu_delay_512 7500000
#./apogpu_delay_512 10000000
#./apogpu_delay_512 12500000
#./apogpu_delay_512 15000000
#./apogpu_delay_512 17500000
#./apogpu_delay_512 20000000

#./apogpu_gain_512 500000
#./apogpu_gain_512 750000
#./apogpu_gain_512 1000000
#./apogpu_gain_512 2500000
#./apogpu_gain_512 5000000
#./apogpu_gain_512 7500000
#./apogpu_gain_512 10000000
#./apogpu_gain_512 12500000
#./apogpu_gain_512 15000000
#./apogpu_gain_512 17500000
#./apogpu_gain_512 20000000

#./apogpu_lowpass_512 500000
#./apogpu_lowpass_512 750000
#./apogpu_lowpass_512 1000000
#./apogpu_lowpass_512 2500000
#./apogpu_lowpass_512 5000000
#./apogpu_lowpass_512 7500000
#./apogpu_lowpass_512 10000000
#./apogpu_lowpass_512 12500000
#./apogpu_lowpass_512 15000000
#./apogpu_lowpass_512 17500000
#./apogpu_lowpass_512 20000000

#./apogpu_lowpass_512_noshared 500000
#./apogpu_lowpass_512_noshared 750000
#./apogpu_lowpass_512_noshared 1000000
#./apogpu_lowpass_512_noshared 2500000
#./apogpu_lowpass_512_noshared 5000000
#./apogpu_lowpass_512_noshared 7500000
#./apogpu_lowpass_512_noshared 10000000
#./apogpu_lowpass_512_noshared 12500000
#./apogpu_lowpass_512_noshared 15000000
#./apogpu_lowpass_512_noshared 17500000
#./apogpu_lowpass_512_noshared 20000000

#./apogpu_delay_16
#mv cuda_profile_0.log 16.log
#./apogpu_delay_32
#mv cuda_profile_0.log 32.log
#./apogpu_delay_64
#mv cuda_profile_0.log 64.log
#./apogpu_delay_128
#mv cuda_profile_0.log 128.log
#./apogpu_delay_256
#mv cuda_profile_0.log 256.log
#./apogpu_delay_512
#mv cuda_profile_0.log 512.log

#./apogpu_gain_16
#mv cuda_profile_0.log 16.log
#./apogpu_gain_32
#mv cuda_profile_0.log 32.log
#./apogpu_gain_64
#mv cuda_profile_0.log 64.log
#./apogpu_gain_128
#mv cuda_profile_0.log 128.log
#./apogpu_gain_256
#mv cuda_profile_0.log 256.log
#./apogpu_gain_512
#mv cuda_profile_0.log 512.log

#export COMPUTE_PROFILE=1
#./apogpu_lowpass_16
#mv cuda_profile_0.log 16.log
#./apogpu_lowpass_32
#mv cuda_profile_0.log 32.log
#./apogpu_lowpass_64
#mv cuda_profile_0.log 64.log
#./apogpu_lowpass_128
#mv cuda_profile_0.log 128.log
#./apogpu_lowpass_256
#mv cuda_profile_0.log 256.log
#./apogpu_lowpass_512
#mv cuda_profile_0.log 512.log
