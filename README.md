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

then do `curl http://localhost:8080/ping`, you should get healthy

then do `curl http://localhost:8080/predictions/sd3?text=dog`and Wait for 20 minutes, you will get a error that worker has failed after 20 minutes. (because of timeout setting in config.properties)

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
