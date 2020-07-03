#!/usr/bin/env bash

# Remove unnecessary files.
rm -rf build/
rm -rf bin/
rm -rf output/

# Pack into tar.gz
cd ..
tar -czvf code.tar.gz student-code --exclude=distribute.sh --exclude=.gitignore
cd student-code