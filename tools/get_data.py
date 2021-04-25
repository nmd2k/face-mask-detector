import wget

print('Downloading')

try:
    url = 'https://doc-04-00-docs.googleusercontent.com/docs/securesc/ltq9iqjft92b73tndgo7e4sjfb8tdcmb/99lrdg3objqsrrclj3qkk17nho5cjppe/1619337300000/07276395443399843357/03772830615363167239/1p-INf8ixUohBGQgDaPAJTGrnVDvHi2Sf?e=download&authuser=0&nonce=1as4rlvjpclps&user=03772830615363167239&hash=57b5c3b4n02j0n2q6fkmcaaqva4n5tsb'
    wget.download(url, './')
except:
    print('Unable to download, please check the download url again')

print('Done')