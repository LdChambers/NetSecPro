#!/bin/bash

# Directory where files will be downloaded
DESTINATION="/home/datasets/monkey"

# Array of URLs to download
URLS=(
  "https://monkey.org/~jose/phishing/phishing-2015"
  "https://monkey.org/~jose/phishing/phishing-2016"
  "https://monkey.org/~jose/phishing/phishing-2017"
  "https://monkey.org/~jose/phishing/phishing-2018"
  "https://monkey.org/~jose/phishing/phishing-2019"
  "https://monkey.org/~jose/phishing/phishing-2021"
  "https://monkey.org/~jose/phishing/phishing-2020"
  "https://monkey.org/~jose/phishing/phishing-2022"
  "https://monkey.org/~jose/phishing/phishing-2023"
  "https://monkey.org/~jose/phishing/phishing0.mbox"
  "https://monkey.org/~jose/phishing/phishing1.mbox"
  "https://monkey.org/~jose/phishing/phishing2.mbox"
  "https://monkey.org/~jose/phishing/phishing3.mbox"
)

# Loop through each URL and download it
for URL in "${URLS[@]}"; do
  wget -P "$DESTINATION" "$URL"
done
