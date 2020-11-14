NAME=imsitu
CODEDIR=$HOME/research/imSitu_GAT
DATADIR=$HOME/data
WORKDIR=/home/$NAME

docker run -idt -w $WORKDIR -e "TERM=$TERM" \
	--name $NAME --pid host \
	-v=$CODEDIR:$WORKDIR/code \
	--gpus all --shm-size 32g \
	-v=$DATADIR:$WORKDIR/data:ro \
	$NAME:devel /bin/bash -c "service ssh restart && /bin/bash"