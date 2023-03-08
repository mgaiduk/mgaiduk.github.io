This is not quite ready for a full-scale training, but just as a teaser, let's try to launch the training on a tpu.  
First, run this code to create a tpu:
```
export TPU_ZONE=europe-west4-a
# WARNING! Billable and expensive
> yes | gcloud beta compute tpus create mgaiduk_recall1 --zone=$TPU_ZONE --version=2.11 --accelerator-type=v3-8 --project=my-first-project;
# Run this after the training to make sure you delete your TPU and stop receiving bills for it
> yes | gcloud compute tpus delete mgaiduk_recall1 --zone=$TPU_ZONE --project=my-first-project|| true
```
