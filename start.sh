#!/bin/bash
rm -rf ~/.cache/huggingface/*
while true; do python fashion_chat.py && break; done
