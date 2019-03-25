#!/bin/bash

wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
mkdir -p speech_commands
tar xvzf speech_commands_v0.02.tar.gz -C speech_commands
