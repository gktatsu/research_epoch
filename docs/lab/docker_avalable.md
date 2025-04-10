# 各マシンのDocker, Docker-composeの利用可否まとめ

## Docker-compose, dockerが利用可能なマシン
- neumann(49, 49)
    - docker: 20.10.5
    - docker-compose: 1.17.1
- bombieri(24)
    - docker: 20.10.12
    - docker-compose: 1.25.0


## dockerのみ利用可能なマシン
- serre
    - dokcer: 20.10.17
- hormander
    - docker: 20.10.21
- smale
    - docker: 20.10.21

## 利用不可なマシン(7GB以上のマシン)
- douglas
- carleson
- rheticus
- reinhold
- godel
- callippus
- aryabhata
- artin
- guldin
- descartes
- whitehead
- tate
- shwartz
- neirenberg
- hypatia
- banach


## チェック用コマンド

@Ron9bee9sslab

docker -v
docker-compose -v
