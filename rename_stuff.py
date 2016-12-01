import os

mypath = os.getcwd()
reldirlist = ['test_train_partitions/ptosis_partition/ptosis_all/', 'test_train_partitions/str_partition/str_all/', 'test_train_partitions/osd_partition/osd_all/']
for reldir in reldirlist:
    flist = os.listdir(os.path.join(mypath, reldir))
    #print(flist)
    # os.rename(f,'IMG'+f.replace(' ','_'))
    [os.rename(os.path.join(reldir,f),os.path.join(reldir,'IMG'+f.replace(' ','_'))) for f in flist if not f.startswith('.')]
