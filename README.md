# DADS Experiment

### Install

``` shell
pip install -r requirements.txt
```

### Run

To run default experiment 1.1 on DADS

``` shell
python3 benchmark.py --model dads --setting 1-1
```

or

``` shell
./run.sh
```

To specify algorithm and dataset setting, please:

choose a certain "**algo**" in
"**dads**", "**ssad**", "**deepSAD**", "**unsupervised**", "**vime**", "**devnet**", "**dplan**",
"**static_stoc**", "**dynamic_stoc**", "**supervised**";

choose a certain "**exp**" in "**1-1**", "**1-2**", "**2-1**", "**2-1**".

Then execute:

``` shell
python benchmark.py --model algo --setting exp
```

To change model configuration of a certain algorithm "**algo**", please switch to directory **
/config/algo/model_config.toml**.