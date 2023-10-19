## Some useful commands

* Remove a folder with progress: `rm -rv FOLDER | pv -l -s $( du -a FOLDER | wc -l ) > /dev/null`
* Tar a progress with progress: `tar -c SOURCE_DIR | pv -s $(du -sb SOURCE_DIR | awk '{print $1}') > OUTPUT.tar`
* Copy a file with progress: `pv SOURCE > DESTINATION`
