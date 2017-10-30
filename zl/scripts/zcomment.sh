#!/usr/bin/env bash

# comment out specific lines

# for example, in release(non-debugging) mode
sed -i -r "s/^([\t ]+)utils.zcheck/\1#utils.zcheck/g"
