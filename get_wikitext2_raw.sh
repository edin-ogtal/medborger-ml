#!/bin/bash
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip
unzip wikitext-2-raw-v1.zip
cd wikitext-2-raw
mv wiki.test.raw test && mv wiki.train.raw train && mv wiki.valid.raw valid
rm wikitext-2-raw-v1.zip