#!/bin/sh
if [ -d "./nclt/train/rgb" ]
then
    echo "There are exisitng files in nclt. Are you sure you want to delete them? (Y/n)"
    read -p "" confirm
    case "$confirm" in [yY])
        echo "Removing"
        rm -rf ./nclt/*
        ;;
    *)
        exit 1
        ;;
    esac
fi

echo "Making train and test directories"
mkdir ./nclt/train
mkdir ./nclt/train/rgb
mkdir ./nclt/train/calibration
mkdir ./nclt/train/poses
mkdir ./nclt/test
mkdir ./nclt/test/rgb
mkdir ./nclt/test/poses
mkdir ./nclt/test/calibration
echo "Done"