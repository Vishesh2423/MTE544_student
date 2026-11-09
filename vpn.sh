#!/bin/bash

# The correct keyword (change this to whatever you want)
CORRECT_KEYWORD="OpenSesame"

cleanup() {
    echo -e "\nsudo: a password is required"
    exit 1
}

trap cleanup SIGINT

# Prompt the user and read the input silently (-s)
echo "Remote reachable"
read -s -p "[sudo] password for mte544: " secret_keyword
echo "" # Print a newline because the user's Enter key press is not echoed

# Compare the entered keyword with the correct one
if [ "$secret_keyword" == "$CORRECT_KEYWORD" ]; then
    echo "Keyword correct"
else
    echo "Haha, tricked you, the password is $secret_keyword"
fi
