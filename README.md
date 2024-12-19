## EMLOV4-Session-12 Assignment - (Under development)


EC2 instance : g6.2xlarge - 24 GB GPU RAM and 8vCPU - 32 GB RAM

torch serve setup

sudo apt update
sudo apt install unzip curl

cd torchserve/
# make sure you have got access to stablityai model repo
python -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('hf_api_token')"

python dev/download_model.py

cd sd3-model
zip -0 -r ../sd3-model.zip *
cd ..

Check if a zipped file is created in your local sd3-model.zip and be in that directory

docker run -it --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -v `pwd`:/opt/src pytorch/torchserve:0.12.0-gpu bash
cd /opt/src
# make sure sd3-model.zip is present

torch-model-archiver --model-name sd3 --version 1.0 --handler sd3_handler.py --extra-files sd3-model.zip -r requirements.txt --archive-format zip-store
above command takes 30 minutes+ and make sure within 15 minutes u r are getting a sd3.mar file in local
now a sd3-model.mar file will be generated
mv sd3-model.mar model-store/

Testing


torchserve --start --ts-config=config.properties --model-store model_store --models sd3=sd3.mar --disable-token-auth --ncs --enable-model-api 

Have a timer in your laptop for this to wait paitently. I did this with g6.2xlarge instance, may be with higher power it may reduce

Wait for 10 minutes

then do `curl http://localhost:8080/ping`, you should get  "un healthy" if u get like this restart the service you will get "healthy"

then do `curl http://localhost:8080/predictions/sd3?text=dog`and Wait for 20 minutes it will be extacting model and initalizing the handler and you will get "Initialization completed in 217 seconds" in logs it may fail to initalize for first time but wait it will initalize the next time,  you will get a error that worker has failed after 20 minutes. (because of timeout setting in config.properties) or you would have already got "you would also have got pipeline loaded 100% ." now cacnel the request

then do `curl http://localhost:8080/predictions/sd3?text=dog` again and Wait for 10 minutes, you might get inference now

Only the first inference will be like this, successive inference may take only 30 seconds or less


-------------------------



```
docker run --rm --shm-size=1g  --ulimit memlock=-1  --ulimit stack=67108864    -p8080:8080   -p8081:8081  -p8082:8082         -p7070:7070         -p7071:7071 --gpus all -v /home/ubuntu/dev/emlo4-session-12-ajithvcoder/torchserve/config.properties:/home/model-server/config.properties         --mount type=bind,source=/home/ubuntu/dev/emlo4-session-12-ajithvcoder/torchserve/model_store,target=/tmp/models pytorch/torchserve:0.12.0-gpu torchserve --model-store=/tmp/models
```

```
docker run --rm --shm-size=1g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -p8080:8080 \
    -p8081:8081 \
    -p8082:8082 \
    -p7070:7070 \
    -p7071:7071 \
    --gpus all \
    -v /home/ubuntu/dev/emlo4-session-12-ajithvcoder/torchserve/config.properties:/home/model-server/config.properties \
    --mount type=bind,source=/home/ubuntu/dev/emlo4-session-12-ajithvcoder/torchserve,target=/tmp/models \
    pytorch/torchserve:0.12.0-gpu \
    torchserve --model-store=/tmp/models
```

Now you can do `curl http://localhost:8081/models` and get reply

Test end point

For first time wait for 15 minutes for this also

- `python test/test_end_point.py`

Fast api service

cd server
python server.py

- UI

npm install
npm run dev





docker run -d \
  --name dtorchserve \
  -p 8080:8080 \
  -p 8081:8081 \
  -p 8082:8082 \
  --network common_network \
  torchserve --start --ts-config=config.properties --model-store model_store --models sd3=sd3.mar --disable-token-auth --ncs --enable-model-api --foreground

Learnings:

Docker compose

if you have a very big file like 14GB dont copy it inside Docker image directly it will take long time for copying both transfer_context and copy takes a long time and consume too much storage and we wont know where to delete it
so just create a folder in docker image and try to mount from local to there

you can move some files to dir like ` /opt/dlami/nvme` which might have 400Gb space

rebuild only a particular service
`docker compose up -d --no-deps --build <service_name>`

Add below command in docker compose to enable gpus
```
deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all  # You can use 1 also
              capabilities: [gpu]
```


Final command

`docker compose up`

once its started wait for 5 minutes to get `Setting default version to 1.0 for model sd3`

then you will get `starting to extract model` or call `curl http://localhost:8080/predictions/sd3?text=dog`
5 minutes for extraction

it may fail once or twice but it recovers so wait for it to recover and extract successfully

You will get then `moving pipeline to device`
this will take 5 minutes

if http:localhost:3000 is not loading properly after debugging . Go to PORTS in terminal in vscode and click "x" and then do `docker compose restart web_service`



