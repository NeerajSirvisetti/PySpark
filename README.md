# Apache Spark
it’s a unified analytics engine for big data processing with built in modules for streaming, SQL, machine learning and graph processing. 
- Spark cluster consists of one driver, who’s doing the management, and then many workers who are actually doing the compute.
- when Spark was created, RDDs were the original form of data representation. RDD itself stands for a resilient distributed dataset, and even the terminology is really important, because it describes how Spark data is represented at a very high level.
# Spark Jobs
a Spark job, there can be one or more Stages. And those are further units of work. The way that we define where a Stage starts and ends is when any sort of data needs to be exchanged.
Anytime there’s a shuffle, we’re going to have a Stage boundary.
In Spark, we also further breakdown Stages into what’s called tasks. And these are just pretty much the smallest unit of work that Spark can do, that’s an individual compute core or compute power operating on a subset of the data.

# PySpark
Apache Spark is written in Scala programming language. PySpark has been released in order to support the collaboration of Apache Spark and Python, it actually is a Python API for Spark. In addition, PySpark, helps you interface with Resilient Distributed Datasets (RDDs) in Apache Spark and Python programming language. This has been achieved by taking advantage of the Py4j library. PySpark LogoPy4J is a popular library which is integrated within PySpark and allows python to dynamically interface with JVM objects. PySpark features quite a few libraries for writing efficient programs.

# PySpark SQL
A PySpark library to apply SQL-like analysis on a huge amount of structured or semi-structured data. We can also use SQL queries with PySparkSQL. It can also be connected to Apache Hive. HiveQL can be also be applied. PySparkSQL is a wrapper over the PySpark core. PySparkSQL introduced the DataFrame, a tabular representation of structured data that is similar to that of a table from a relational database management system.





This Repo is for getting familiarize with the Pyspark with Python for BigData through Jupyter Notebook
