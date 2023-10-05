## Install

py-spy record -o utils/profiles/profile.svg -- python3 utils/run_profiling.py


snakeviz utils/profiles/create_groundtruths.cprofile

mprof run utils/run_profiling.py --multiprocess

mprof plot -s

mprof clean