#!/bin/bash
eval ${PRELOAD_FLAG} ${BIN_DIR}/mm > stdout.txt 2> >(grep -v "WARNING" > stderr.txt)
