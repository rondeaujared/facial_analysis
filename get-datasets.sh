#!/bin/bash

if [ -z "$1" ]
    then
        echo "Error: Please enter directory to store all datasets"
    else
        if [ ! -d $1 ]
        then
            mkdir -p $1
            echo "Creating symbolic link for ${1}"
            ln -s $1 datasets
        fi

        cd datasets

        if [ ! -d "appa-real" ]
        then
            echo "Downloading appa-real..."
            mkdir appa-real
            cd appa-real
            wget http://158.109.8.102/AppaRealAge/appa-real-release.zip
            unzip -q appa-real-release.zip
            cd ..
        fi

        if [ ! -d "adience" ]
        then
            echo "Downloading adience..."
            mkdir adience
            cd adience

            wget -r -l 2 --user adiencedb --password adience http://www.cslab.openu.ac.il/download/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/
            mv www.cslab.openu.ac.il/download/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/aligned.tar.gz .
            mv www.cslab.openu.ac.il/download/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/faces.tar.gz .
            rm -r www.cslab.openu.ac.il
            tar -xzf aligned.tar.gz
            tar -xzf faces.tar.gz
            cd ..
        fi

        if [ ! -d "imdb-wiki" ]
        then
            echo "Downloading imdb-wiki; this will take awhile!"
            mkdir imdb
            cd imdb
            for i in {0..9}
            do
                wget https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_${i}.tar
                tar -xf imdb_${i}.tar
            done
            wget https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_meta.tar
            tar -xf imdb_meta.tar

            wget https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar
            tar -xf imdb_crop.tar
            cd ..
        fi

        if [ ! -d "YouTubeFaces" ]
        then
            echo "Downloading YouTubeFaces..."
            mkdir YouTubeFaces
            cd YouTubeFaces

            wget --user wolftau --password wtal997 http://www.cslab.openu.ac.il/download/wolftau/YouTubeFaces.tar.gz
            tar -xzf YouTubeFaces.tar.gz
            cd ..
        fi
    fi

