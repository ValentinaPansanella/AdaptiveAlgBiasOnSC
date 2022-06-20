import os

for subdir, dirs, files in os.walk("."):
    dirname = subdir[2:]
    for filename in files:
        newname = dirname+'_'+filename
        os.rename(os.path.join(subdir, filename),os.path.join(subdir,newname))