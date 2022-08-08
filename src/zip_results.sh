#!/bin/bash

read -p "Please enter the directory: " -r res_path
echo "$res_path"
cd "$res_path"

shopt -s extglob
zip every500agents.zip *000.agent *500.agent
zip final_agents.zip trial!(*+(epd)*).agent
zip stats_data.zip *.npy *.txt *.cfg
