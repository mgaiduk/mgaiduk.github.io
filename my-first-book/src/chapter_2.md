# Chapter 2. The data
To make the code reproducible, we will be using public dataset. I've decided to go with Movielens (https://grouplens.org/datasets/movielens/), although it is not available in bigquery. We will upload it to bq by hand.
## Getting movielens data
At the time of writing, movielens data is available to download at https://grouplens.org/datasets/movielens/.   
Inside the zip archive, there is a file called ratings.csv with the following content:  
```
userId,movieId,rating,timestamp
1,296,5.0,1147880044
...
```

Which is just what we needed! Now, let's load it to bq.  
First, we need to load it to gcs:
```
gsutil cp ratings.csv "gs://mgaiduk/tmp/ratings"
```
Then, we upload it to bq:
```
bq load \
--autodetect \
--source_format=CSV \
mgaiduk.ratings \
"gs://mgaiduk/tmp/ratings"
```
Finally, do a quick select to validate that the data is there and to see how it looks like:
```
-- in bigquery console
SELECT * FROM `mgaiduk.ratings` LIMIT 1000
```
This is the result:
|Row|userId|movideId|rating|timestamp|
|---|------|--------|------|---------|
|1  |    70|    3948|2.0   |1255219128|
|2  |    188|    653|2.0   |1025333400|
|3  |    243|    103249|2.0   |1464280162|
Preparation is done, and now the data looks like what you'd expect in a big tech company residing in GCP: a bigquery table. Now let's talk about how to actually load it into Tensorflow

## BQ to tensorflow?
All operations in Tensorflow are performed on Tensors. A tensor has to reside in memory. There are 2 challenges for that.  

First of all, python code is very slow. Tensorflow intrinsics are written in something else (C?). For user pipelines, they provide a `@tf.function` decorator and other facilities that do approximately the following: they look into your python code, turn it into some intrinsic, optimized representation, and compile it to make it really fast. The only problem is - it works only for code working with tensors. But you have to get your data and turn it into tensors somehow!  
Second problem is scaling up. Entire dataset will not fit in memory or on disk on a single machine. We will need to get it from some scalable storage and parse on-the-fly. We will want our models training on many servers, either on several TPU pods or several GPU workers. Therefore, we need to write the code in such a way that each worker will load its own portion of the dataset.

In the past, I've tried different approaches of doing that: loading an entire file into memory, then turning into tensors in one go; parsing input line by line in python code and passing it to Tensorflow through a "dataset from generator" feature. Neither seemed to be effective or convenient - "dataset from generator" api is exceptionally unintuitive, and you have to call a lot of python code in a cycle before converting data to tensors, which makes it slow and ineffectve. One solution that I find both easy and effective is to use either [tf.data.TFRecordDataset](https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset) or [tf.data.TextLineDataset](https://www.tensorflow.org/api_docs/python/tf/data/TextLineDataset) that allow you to create a dataset (thus solving the initial "bootstrap" problem of getting your data into tensors somehow!) using filename or even a file pattern for local filesystem or GCS. GCS one is excatly what we need for future scalable solutions!  

Tensorflow Datasets API provides only 2 file formats for easy data loading: TextLineDataset and TFRecordDataset. Other solutions (like parsing Parquet files) seem to also be available, but they are less popular, sometimes have experimental APIs, and, at a first glance, are not made as convenient as the first two to use in Tensorflow. So we will stick to those two in this book. 
### Getting data from BQ to GCS - CSV
Let's see how we can get them from BQ to gcs.  

Native extraction API, available through "export" tab in BQ UI, or `bq extract` command line tool, has following formats: CSV, NEWLINE_DELIMITED_JSON, AVRO, PARQUET. Avro and Parquet create binary files that could not be parsed from Tensorflow with the dataset API functions that we agreed on. And there is no TFRecord here - we will have to resort to Dataflow Templates to do that ) But that is to be expected - TFRecord is a low-popularity, very specific format, and its not like Tensorflow and Bigquery were made by the same company.  

We are left with CSV and JSON. CSV is actually pretty good, I've seen it used at large scale in production. Its downside is that it is a text format, which will bloat the size of numbers. It is a "plain" format, which won't work out of the box for arrays or nested structures. Upside is that it doesn't store anything extra, like labels per each record (json and tfrecord both do that). It is also human-readable, so it will be possible to check the data out with your own eyes. 
JSON is a bit more bloated, with labels per each row; but it makes it possible to store nested structures and arrays.  

Here is how to export data from bq to gcs:
```
bq extract --noprint_header --destination_format CSV --compression GZIP mgaiduk.ratings "gs://mgaiduk-us-central1/ratings/csv_gzip/part*"
```
There are some interesting details here. First of all, here we are using "native" export utility, which has access to private datastore api's not available for general public. It is faster and cheaper then the alternatives (like dataflow), takes mere minutes for petabyte datasets. However, using this way limits us to format options described above - CSV or JSON.  

Exported files represent a sharded gcs file: 
```
part000000000000	12.1 MB	application/octet-stream	Mar 3, 2023, 1:19:15 PM	Standard	Mar 3, 2023, 1:19:15 PM	Not public	—	
Google-managed key
—			
part000000000001	12.1 MB	application/octet-stream	Mar 3, 2023, 1:19:14 PM	Standard	Mar 3, 2023, 1:19:14 PM	Not public	—	
Google-managed key
—			
part000000000002	12.1 MB	application/octet-stream	Mar 3, 2023, 1:19:13 PM	Standard	Mar 3, 2023, 1:19:13 PM	Not public	—	
Google-managed key
—	
```
Which makes it quite convenient to work with, especially in ditributed training setups.  

We do not extract headers. Having to skip 1 line from each file is, again, not quite convenient in distributed training setups.  

We use GZIP compression, which is optional, of course. It is supported natively by Tensorflow, and helps partially mitigate the bloat we discussed above. It should be a trade-off between network and storage cost and CPU spent on decompression; whether or not it actually helps speed up the training will be clear from later chapters, when we will run some experiments.  

Finally, as can be seen from the bucket name, it is located in a specific region - `mgaiduk-us-central1`. It is important to track data locality, and to launch tranining in the same region where the data is located. It helps both with input pipeline throughput and network costs.  

### BQ to GCS TFRecords using dataflow
Another alternative to native BQ export is to use Dataflow. The idea is simple: launch a bunch of dataflow workers that will grab data from Bigquery, convert to a desired format, and write it to GCS.  
Unlike native export, this doesn't allow workers to use private storage API, so the export will be much slower. There is also a HUGE overhead in time and costs for launching the dataflow job: you have to wait for all the workers to be created and start doing their job; you will not receive any feedback in case something goes wrong, and will have to wait for all the workers to start up, do some retries, and then fail. Typically, on a petabyte of data, the job takes a few hours, and if there are errors in the configuration, the job will fail within half an hour. You also have to pay for worker cpu and other excess resources, which makes it much more costly then native BQ export.  
The actual export format here can be anything we want, but Google provides a set of templates with some of the most popular formats already suported, including TFRecords: https://cloud.google.com/dataflow/docs/guides/templates/provided-batch.   
Another big problem with this approach is that it is error-prone. If one of the fields is nullable in the table, and your input data pipeline doesn't expect that - you will spend an entire day (and quite a few hundred bucks!) collecting the data only to learn that you have to fix the problem and do it all over again. If your table format is not supported by the template (say, you have nested structures), your job will run for half an hour before failing. So my advice here is - first test everything, including model training in the final setup - on a sample of data, say, first 100k rows, then collect the entire thing.  

TFRecordDataset and TFRecords were designed specifically for Tensorflow. It is a protobuf-based, binary format, which means that there is no bloat from textual representation, and parsing is faster. However, TFRecords save labels for every row, as well as some protobuf metadata. For our example - userId,movieId,rating,timestamp - this makes TFRecord dataset be actually bigger in size then CSV. We shall see if easier parsing makes it worthwile.

## Look at the data
The data is now in GCS, where it will be stored for all our training needs. Now let's have a first glance at the data and on how to actually load it into tensorflow:
```
import tensorflow as tf
tf.__version__
# csv version
dataset = tf.data.TextLineDataset(["gs://mgaiduk-us-central1/ratings/csv_gzip/part000000000000"], compression_type="GZIP")
for line in dataset:
    break
print(line)
```
`tf.Tensor(b'70,3948,2,1255219128', shape=(), dtype=string)`
As promiseed, parsing the data is very easy. Output is the dataset with 1 row yielding one csv string tensor. We will need to parse it further to actually use it in our model; but it is already a tensor, so every parsing done on it will be optimized with Tensorflow utilities.
```
# tfrecord version
dataset2 = tf.data.TFRecordDataset(["gs://mgaiduk-us-central1/ratings/tfrecord/train/output-00000-of-00012.tfrecord"])
for line in dataset2:
    break
print(line)
```
`tf.Tensor(b'\n:\n\x11\n\x06userId\x12\x07\x1a\x05\n\x03\xf3\x9d\x03\n\x11\n\x07movieId\x12\x06\x1a\x04\n\x02\xca\x15\n\x12\n\x06rating\x12\x08\x12\x06\n\x04\x00\x00\x80@', shape=(), dtype=string)`  

Code examples are also available as [jupyter notebook](https://github.com/mgaiduk/mgaiduk.github.io/blob/main/my-first-book/src/code/chapter2/data_first_glance.ipynb).