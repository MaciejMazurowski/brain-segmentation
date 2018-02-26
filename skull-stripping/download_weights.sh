CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=1yfXFngcJvBO3EKuC6-ASboW_TPyLSLgd" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=1yfXFngcJvBO3EKuC6-ASboW_TPyLSLgd" -O weights_128.h5

rm -rf /tmp/cookies.txt
