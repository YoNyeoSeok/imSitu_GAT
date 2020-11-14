NAME=imsitu

IMAGE=yonyeoseok/$NAME:cuda10.0-devel
docker build -t $IMAGE - < Dockerfile
docker push $IMAGE
