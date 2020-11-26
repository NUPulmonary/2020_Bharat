#!/bin/bash

mkdir -p data/01samples

wget "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE158127&format=file" -O data/01samples/GSE158127_RAW.tar
cd data/01samples
tar xf GSE158127_RAW.tar
cd -
