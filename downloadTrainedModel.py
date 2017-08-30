from __future__ import absolute_import
from __future__ import print_function

import hashlib
import urllib.request
import zipfile

url = 'https://www.dropbox.com/s/anh93ggeh9xtsnr/nat_trained.zip'

# fetch adv_trained model
#url = 'https://www.dropbox.com/s/9z7tnleh2hrf158/adv_trained.zip'

fname = url.split('/')[-1]  # get the name of the file

url = url + "?dl=1"

# model download
print('URL : ' + url)
print('filename : ' + fname)

urllib.request.urlretrieve(url, fname)
sha256 = hashlib.sha256()

with open(fname, 'rb') as f:
    data = f.read()
    sha256.update(data)

print('SHA256 hash: {}'.format(sha256.hexdigest()))
print('Extracting model')

with zipfile.ZipFile(fname, 'r') as model_zip:
    model_zip.extractall()
    print('Extracted model in {}'.format(model_zip.namelist()[0]))
