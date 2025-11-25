#!/bin/bash


git add .gitignore
git rm --cached -r tutorials/checkpoints/*.pth 2>/dev/null || true
git add .
git status --short
git commit -m "Update README and tutorials"
git push
