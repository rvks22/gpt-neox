#!/bin/bash
python deepy.py train.py pythia-160m-deduped.yml  2>&1 | tee output.txt