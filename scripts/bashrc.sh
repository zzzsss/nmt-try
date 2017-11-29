#!/usr/bin/env bash

# useful functions & alias (in ~/.bashrc)
# -- put here for possible future usage

# basic
export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4
export LD_PRELOAD=""
if uname -m | grep x86; then echo "x86 platform"; fi
if hostname | grep 5055; then echo "special platform"; fi

# PS1
export TERM=xterm-color
export LANG=en_US.UTF-8
export COLOR_NC='\e[0m' # No Color
export COLOR_WHITE='\e[1;37m'
export COLOR_BLACK='\e[0;30m'
export COLOR_BLUE='\e[0;34m'
export COLOR_LIGHT_BLUE='\e[1;34m'
export COLOR_GREEN='\e[0;32m'
export COLOR_LIGHT_GREEN='\e[1;32m'
export COLOR_CYAN='\e[0;36m'
export COLOR_LIGHT_CYAN='\e[1;36m'
export COLOR_RED='\e[0;31m'
export COLOR_LIGHT_RED='\e[1;31m'
export COLOR_PURPLE='\e[0;35m'
export COLOR_LIGHT_PURPLE='\e[1;35m'
export COLOR_BROWN='\e[0;33m'
export COLOR_YELLOW='\e[1;33m'
export COLOR_GRAY='\e[0;30m'
export COLOR_LIGHT_GRAY='\e[0;37m'
export PS1="$TITLEBAR\n\[${UC}\]\u@\h \[${COLOR_LIGHT_BLUE}\]\${PWD} \[${COLOR_BLACK}\]\n\[${COLOR_LIGHT_GREEN}\]->\[${COLOR_NC}\] "

# showings
function zpgrep0 { ps -elf | grep "`whoami`.*$1"; }
function zpgrep { ps -elf | grep `whoami` | grep $1; }
function zpkill { ps -elf | grep `whoami` | grep $1 | sed -r 's/[ ]+/ /g' | tee /proc/self/fd/2  |  cut -f 4 -d ' ' | xargs kill -9; }
function zcgpu
{
nvidia-smi | grep "^|[ \t]*[0-9].*MiB" | sed -r 's/[ ]+/ /g' | while read line; do
G_ID=`echo $line | cut -d ' ' -f 2`
G_PID=`echo $line | cut -d ' ' -f 3`
echo "`hostname`+${G_ID}===========$line"
ps -lf -p $G_PID | tail -1
done
}

# checkings
ZGPUS=( 44 45 46 47 48 )
function zcg0
{
for n in ${ZGPUS[*]}; do
echo; echo GPUs@50$n; ssh -i key mm$n bash -ic zcgpu;
done
}

# tools
alias seg-jp="$ZZ/mt/tools/kytea-0.4.7/src/bin/kytea -model $ZZ/mt/tools/kytea-0.4.7/data/model.bin -notags"
function seg-ch
{
bash $ZZ/mt/tools/stanford-seg/segment.sh ctb $1 UTF-8 0 2>/dev/null;
}

alias vim2='vim -O +"windo set scb"'
