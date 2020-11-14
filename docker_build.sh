NAME=imsitu

IMAGE=yonyeoseok/$NAME:devel
docker build -t $IMAGE - < Dockerfile
docker push $IMAGE