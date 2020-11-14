NAME=imsitu

IMAGE=yonyeoseok/$NAME:devel
docker pull $IMAGE

# add local user
docker run --rm -idt --name $NAME $IMAGE
docker exec $NAME apt-get update --fix-missing
docker exec $NAME apt-get install -y sudo git vim screen ssh
docker exec $NAME apt-get clean
docker exec $NAME useradd -m -u $UID -G sudo,conda $NAME
docker exec $NAME /bin/bash -c "echo $NAME:passwd | chpasswd"
docker commit $NAME $NAME:devel
docker stop $NAME