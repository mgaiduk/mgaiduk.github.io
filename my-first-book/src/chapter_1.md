# Chapter 1. What is this book about
Who am I and why am I writing this  

------
This book is about Tensorflow and recsys.  

It has been a couple of years since leaving a big tech company (Yandex) specialized in search and recommendations. Back then, we had an entire ecosystem for everything that we could need: mapreduce, SQL, machine learning, research reproducibility and process automation (airflow/flyte). Right now, I tend to think that what we had there was actually pretty good, better then stuff that is available in the open source or as enterprise solutions (like Google Cloud).  

Back then, we had a c++ written tool to train neural networks. It was fast, well-integrated into our ecosystem - it was easy to set up distributed training while loading data from MapReduce table, for example. It was not flexible - but we did implement most important stuff into it: dense networks, embedding tables, transformers, convolution networks.  

But all that is in the past. Now I have to leave in the open world, use open technologies, learn their weaknessess and strengths. It is at least my third time trying to set up big scale training in Tensorflow. For me it was always a huge pain in the butt.  

Tensorflow tutorials usually deal with datasets that fit entirely into memory, sometimes even just preloaded with a provided library functions. When you deal with real-world problems, you have to first parse the data somehow, and Python is not really effective at that.  

It is really easy to make mistakes - and Python is not really good at pointing where exactly those mistakes were made.for example, I was getting this error: `tensorflow ValueError: 'outputs' must be defined before the loop.`. My code had no loops and it was not obvious what 'outputs' should mean. The fix was to set `steps_per_execution=1` in `model.compile()`. Thank god this error was popular enough and Google knew about it. Getting data from wherever it is stored into model training inputs was always troublesome - parsing it with my own code in Python was too costly, library functions were not exactly what I needed for my input, and so on. Sometimes stuff works but is just too slow - like EmbeddingLayer when you try to scale it up to millions of parameters. 

Another big problem was to set up training in a convenient, reproducible way. To make models accept neat configs and parameters so that different ML engineers can just reuse the same code - this seems to be a weak point in a lot of organizations, with people copy-pasting each other code, storing it locally or on some VMs without commiting it, rewriting the same thing over and over again. To save models in proper format, usable from another language during inference, with meta information about model parameters, input data and architecture. 

Now, after all that struggle, I got into a team where some of these problems are solved. We train models with around 100m vocab size, 50b training pool. We store data in google cloud (bigquery + gcs) and train models on TPUs. We focus on recsys models (think "Netflix Challenge"). So here is what the book is about:
- How to get data from bigquery to gcs, to tensorflow tensors in a simple and performant way
- Scaling the data: datasets that don't fit into memory
- How to set up model training, how to train on TPUs, what TPUs are, why can they be better and are they really better?
- Convenience features setup for model customization and saving
- Recsys architecture overview and experiments: collaborative filtering models, deep neural networks, transformers and other things to try