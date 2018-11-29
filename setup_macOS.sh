#!/bin/bash

which -s brew
if [[ $? != 0 ]] ; then
    echo "Installing Hombrew"
    ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
else
    echo "Updating Homebrew"
    brew update
fi

echo "Homebrew : Done."

which -s python3.7
if [[ $? != 0 ]] ; then
    echo "Installing python3.7"
    brew install python3
else
    echo "Updating python3.7"
    brew upgrade python3.7
fi

echo "python3.7 : Done."

echo "Updating pip3"
curl https://bootstrap.pypa.io/get-pip.py | python3
pip3 install --upgrade pip
echo "pip3 : Done."

echo "Installing package sc2"
python3.7 -m pip install sc2
echo "Package sc2 : Done."
