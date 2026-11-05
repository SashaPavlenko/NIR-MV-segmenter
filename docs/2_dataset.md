# 2. Dataset

The dataset is **not public** (it shows the participants of the study). It is available on
request from the author. You receive one archive. Unpack it so that `data/` looks like this:

```
data/
├── recordings/                      the frames
│   ├── run_1_drill_3/
│   │   └── images/
│   │       ├── 1/                   device 1 (of 4)
│   │       │   ├── left/            left eye
│   │       │   │   ├── 1778595887632023_left.bmp     <capture time in µs>_<eye>.bmp
│   │       │   │   └── ...
│   │       │   └── right/
│   │       ├── 2/  3/  4/
│   ├── run_1_scalpel_2/
│   └── ...                          15 recordings: 5 participants × (drill, scalpel, screwdriver)
├── annotations/                     CVAT for images 1.1 export, one file per annotated view
│   ├── job_390.xml
│   └── ...                          52 files
└── job_frames.json                  the view list (see below)
```

```bash
mkdir -p data && tar xf <archive> -C data      # or unzip; then:
bash scripts/check_data.sh
```

If the data lives somewhere else (a shared disk), do not copy it: set `DATA_DIR` (or
`RECORDINGS_DIR`, `ANNOTATIONS_DIR`, `JOB_FRAMES` separately) in `env.sh`.

## What is in it

* **Recordings.** Four Stereo IR 170 devices, each with two eyes, give eight views per
  recording. Frames are 384 × 384 px, 8-bit NIR, BMP. The original recordings run at 90 fps.
  The dataset holds only the frames that were put into CVAT: every 3rd frame (30 fps),
  125 to 145 frames per view. The file name starts with the capture time in microseconds. The
  four devices are synchronised only in software, so the pipeline matches frames across
  devices by time (within 20 ms), not by index.
* **Annotations.** 52 of the 120 views are annotated with masks of 8 classes: `hand_left`,
  `hand_right`, `haptic_left`, `haptic_right`, `scalpel`, `screwdriver`, `drill`, `object`.
  Masks are stored as RLE inside the XML (`<image id=...>` = CVAT frame number).
* **`job_frames.json`.** One entry per view (= one CVAT job):

  ```json
  {"job_id": 390, "task_name": "run_1_drill_3__cam2_left",
   "frame_ids": [0, 3, 6, ...], "frame_names": ["1778595887632023_left.bmp", ...], ...}
  ```

  `task_name` = `<recording>__cam<device>_<eye>`. `frame_ids` are the CVAT frame numbers,
  `frame_names` the files, in the same order. The whole pipeline uses this list to find the
  frames of a view. It is written by `cvat/export_job_frames.py`.
* **Device grids** (`calibration/leap_grids/<device>/distortion_<eye>.bin`) are in the
  repository, not in the dataset. They are the `LEAP_DISTORTION_MATRIX` of each device and eye,
  the same in every recording, and are used to rectify the stereo pairs.

## Folds

`splits/folds.json` defines the leave-one-participant-out evaluation of the thesis. Fold *k*
holds out participant *k* (three recordings, 24 views): `infer_jobs` are the 24 views,
`test_jobs` the annotated ones among them (9 + 9 + 10 + 15 + 9 = 52).

## Own recordings

New recordings in the same folder layout can be put into `data/recordings/` (or any folder).
Without CVAT, make a view list for them with

```bash
python src/make_views.py --recordings data/recordings --runs run_7_drill_1 --step 3 --out outputs/new/views.json
```

and pass it to `predict.sh --views outputs/new/views.json` (see [4_inference.md](4_inference.md)).
Each view needs all four devices with both eyes for the disparity channel and the consensus.

## For the author: building the archive

`src/pack_dataset.py` copies exactly what is described above out of the original folders
(and nothing else, no participant tables):

```bash
python src/pack_dataset.py --recordings <dataset_jobs or dataset_12-05> --annotations <cvat_xml_13_10> \
    --job-frames <job_frames.json> --out package/data
tar czf hoist_nir_dataset.tar.gz -C package/data .
```
