CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=12LF2WJgSzqtn8t2D8-r3V8uunT84t1v-" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=12LF2WJgSzqtn8t2D8-r3V8uunT84t1v-" -O weights_64.h5

rm -rf /tmp/cookies.txt
