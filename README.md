```
docker run --name searxng -d \                                              
    -p 8888:8080 \
    -v "./config/:/etc/searxng/" \
    -v "./data/:/var/cache/searxng/" \
    docker.io/searxng/searxng:latest
```

```
docker run -d --name searxng -p 8888:8080 \
  -v "$(pwd)/config:/etc/searxng" \
  -v "$(pwd)/data:/var/cache/searxng" \
  searxng/searxng:latest
```

```
docker run --name searxng -d -p 8888:8080 -v "%cd%/config/:/etc/searxng/" -v "%cd%/data/:/var/cache/searxng/" docker.io/searxng/searxng:latest 

docker run -d --name mullvad-browser -p 3000:3000 -p 3001:3001 -e TZ=Asia/Ho_Chi_Minh --shm-size=1gb lscr.io/linuxserver/mullvad-browser:latest 

docker run -d --name mullvad-browser -p 3000:3000 -p 3001:3001 -e TZ=Asia/Ho_Chi_Minh -v D:/PiconT/Stockmask/mullvadData:/config/Downloads lscr.io/linuxserver/mullvad-browser:latest
```

```
docker run -d \
  --name mullvad-browser \
  --platform=linux/amd64 \
  -p 3000:3000 \
  -p 3001:3001 \
  -e TZ=Asia/Ho_Chi_Minh \
  -v "$(pwd)/mullvadData:/config/Downloads" \
  --shm-size=1gb \
  lscr.io/linuxserver/mullvad-browser:latest
```