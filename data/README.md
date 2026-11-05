# data/

Put the dataset here (it is not part of the repository, see [docs/2_dataset.md](../docs/2_dataset.md)):

```
data/recordings/<run>/images/<device 1-4>/<left|right>/<timestamp_us>_<side>.bmp
data/annotations/job_<id>.xml
data/job_frames.json
```

Check with `bash scripts/check_data.sh`. Everything in this folder except this file is ignored by git.
