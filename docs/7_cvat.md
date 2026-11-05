# 7. CVAT

The annotation of the thesis was made in a self-hosted [CVAT](https://github.com/cvat-ai/cvat)
(version 2.66). The model output can be uploaded into CVAT as pre-annotation and corrected
there by hand. This page describes the whole loop:

```
recordings ──create_tasks.py──► CVAT tasks/jobs ──export_job_frames.py──► job_frames.json
                                     ▲                                        │
     upload_preannotations.py ◄── cvat_xml/ ◄── scripts/predict.sh ◄──────────┘
                                     │
     correction by hand in CVAT ──export_job_annotations.py──► data/annotations/job_<id>.xml
```

All scripts are in `cvat/`. They talk to CVAT through its API with the Python SDK.

## 1. Install CVAT

Needs Docker and the Docker Compose plugin **2.24 or newer** (for `!override`), on Linux.

```bash
git clone --branch v2.66.0 --depth 1 https://github.com/cvat-ai/cvat.git ~/cvat
cp <this repository>/cvat/docker-compose.override.yml ~/cvat/
cd ~/cvat
docker compose up -d                      # first start downloads the images (a few GB)
docker compose ps                         # all containers "Up" / "healthy" after 1-2 minutes
```

`docker-compose.override.yml` publishes CVAT on port **8181** instead of 8080
(`CVAT_PORT=... docker compose up -d` for another one). Compose reads the override file by itself.

Create the admin user once:

```bash
docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
```

Open http://localhost:8181 and log in.

Stop / start (the data stays in Docker volumes):

```bash
cd ~/cvat && docker compose down        # stop
cd ~/cvat && docker compose up -d       # start
```

### Optional: Segment Anything in CVAT (needs an NVIDIA GPU)

For the interactive "AI tools" in the annotation view, CVAT runs models with nuclio:

```bash
cd ~/cvat
docker compose -f docker-compose.yml -f docker-compose.override.yml \
    -f components/serverless/docker-compose.serverless.yml up -d
# nuctl (the nuclio command line tool) must have the version of the nuclio dashboard in
# components/serverless/docker-compose.serverless.yml: 1.15.9 for CVAT 2.66.0
wget https://github.com/nuclio/nuclio/releases/download/1.15.9/nuctl-1.15.9-linux-amd64
sudo install nuctl-1.15.9-linux-amd64 /usr/local/bin/nuctl
./serverless/deploy_gpu.sh serverless/pytorch/facebookresearch/sam/nuclio/
```

Write the three files into `~/cvat/.env` to use them with plain `docker compose up -d` later:

```
COMPOSE_FILE=docker-compose.yml:docker-compose.override.yml:components/serverless/docker-compose.serverless.yml
```

### Optional: access from outside (ngrok)

So that annotators can work from home, the thesis CVAT ran behind an ngrok tunnel with a fixed
domain. Django accepts a login only from known origins, so the public address must be added:

```bash
cp <this repository>/cvat/docker-compose.remote.yml ~/cvat/
bash <this repository>/cvat/make_production_patch.sh ~/cvat     # writes ~/cvat/production_patched.py
echo "CVAT_PUBLIC_HOST=<your-domain>.ngrok-free.dev" >> ~/cvat/.env
echo "COMPOSE_FILE=docker-compose.yml:docker-compose.override.yml:docker-compose.remote.yml" >> ~/cvat/.env
cd ~/cvat && docker compose up -d
ngrok http --url=https://<your-domain>.ngrok-free.dev 8181
```

`cvat/run_web.sh` switches all of it on and off in one call and keeps the machine from going to
sleep meanwhile (Ubuntu / GNOME):

```bash
CVAT_DIR=~/cvat NGROK_URL=https://<your-domain>.ngrok-free.dev bash cvat/run_web.sh   # on; again = off
```

## 2. Python environment for the scripts

On the machine that can reach CVAT (usually the CVAT machine itself; the GPU cluster cannot):

```bash
python3 -m venv ~/cvat_venv && source ~/cvat_venv/bin/activate
pip install -r cvat/requirements.txt          # cvat-sdk 2.66.0 = the server version
export CVAT_LOGIN=<user> CVAT_PASSWORD=<password>
cd cvat
```

The scripts find CVAT through `cvat_host.py`: `CVAT_HOST` / `CVAT_PORT` if set, otherwise a
running ngrok tunnel, otherwise http://localhost:8181. For uploads and exports use localhost,
not the tunnel.

## 3. Project and tasks

```bash
python create_project.py --name "NIR hands and tools"          # prints: project #<id>
cp annotation_runs.example.csv annotation_runs.csv             # list the recordings to annotate
python create_tasks.py --project-id <id> --recordings ../data/recordings --runs-csv annotation_runs.csv
```

* The project gets the eight mask labels (`hand_left` … `object`). Do not rename them.
* `create_tasks.py` makes one task per view, named `<recording>__cam<device>_<eye>`. The
  `fps` column sets the frame step: the recordings run at 90 fps, `fps=30` keeps every 3rd frame.
  The thesis used 30 fps.

Then write the view list of all jobs:

```bash
python export_job_frames.py --output ../data/job_frames.json
```

## 4. Pre-annotation with the model

On the GPU machine, with the `job_frames.json` from step 3 (the job ids must be those of CVAT):

```bash
bash scripts/predict.sh --views data/job_frames.json --runs "run_7_drill_1" --fold 1 --out outputs/run7
```

Copy `outputs/run7/cvat_xml/` to the CVAT machine and upload:

```bash
python upload_preannotations.py --xml-dir <path>/cvat_xml --all --dry-run    # checks only, writes nothing
python upload_preannotations.py --xml-dir <path>/cvat_xml --all
```

What the upload does:

* each XML is checked first (RLE sums, boxes inside the frame, frames inside the job, length of
  the job, RLE orientation). A job with any error is not touched;
* the labels are found **by name** in the project; a missing name stops that job;
* masks are written as per-frame shapes (CVAT cannot open mask tracks); one instance = one `group`;
* the job's current annotation is **replaced**. It is saved first to
  `preanno_backup/job_<id>_<time>.json`. Shapes drawn by hand (`source=manual`, no group) are kept
  unless `--drop-manual` is given;
* afterwards the annotation is read back and compared mask by mask (`--no-verify` to skip).

## 5. Correct by hand, then export

Correct the masks in the CVAT UI (Jobs → open a job). Afterwards:

```bash
python export_job_annotations.py 390 391 409 --output ../data/annotations
# or: python export_job_annotations.py --jobs-file jobs.txt --output ../data/annotations
```

writes `job_<id>.xml` (CVAT for images 1.1, masks as RLE) per job, the format that
`src/evaluate.py` and `scripts/train_lopo.sh` read.

## Problems

* **The job does not open after an upload ("unexpected type of track mask").** Mask tracks were
  imported through the UI. Upload with `upload_preannotations.py`, which writes shapes.
* **Login fails through the tunnel (CSRF).** `CVAT_PUBLIC_HOST` is missing or wrong, see
  "access from outside".
* **Only frame 0 has masks after an upload.** The read-back check reports this as missing
  frames; the backup of the old annotation is in `preanno_backup/`.
* **Port 8181 is taken.** Set another one: `CVAT_PORT=8282 docker compose up -d` and
  `export CVAT_PORT=8282` for the scripts.
