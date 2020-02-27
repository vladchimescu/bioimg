#!/bin/bash

# download all the files

# AML data
wget -O AML_screen.zip https://www.dropbox.com/sh/a8hg0if0nfps03o/AADpbSnzzAlFGzgycKSNPf-ma?dl=1
unzip AML_screen.zip -d AML_screen
rm AML_screen.zip

# CLL data
wget -O CLL-coculture.zip https://www.dropbox.com/sh/5jv64grgrcwqr8p/AAC_8-d2RYgAv4DaPZkS3EOEa?dl=1
unzip CLL-coculture.zip -d CLL-coculture
rm CLL-coculture.zip

# BiTE data
wget -O BiTE.zip https://www.dropbox.com/sh/dqkshzjj4y2bmtz/AACaK_zJWZOatIVfSZtDb4OFa?dl=1
unzip BiTE.zip -d BiTE
rm BiTE.zip


