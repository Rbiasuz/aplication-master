import shutil
import random

def leitura_paste(path):
    cont_nodulo = 0
    cont_nonolo = 0
    url_base = path
    

    for file_name in os.listdir(url_base):
        
        
        for file2 in os.listdir(url_base+'/'+file_name):
            for file3 in os.listdir(url_base+'/'+file_name+'/'+file2):
                for file4 in os.listdir(url_base+'/'+file_name+'/'+file2+'/'+file3):
                    if file4.endswith('.dcm'):
                        img = im.imread(os.path.join(url_base, file_name, file2, file3, file4)).astype(np.float64)
                        img_dic = img.meta
                        scan = str(img_dic['SeriesNumber'])
                        slice_n = str(img_dic['InstanceNumber'])
                        referencia = scan+slice_n
                        if True: #random.choices(mylist) == [7]:
                            if to_add[to_add['referencia'] == referencia]['has_tumor'].any():
                                cont_nodulo +=1
                                if cont_nodulo < 20:
                                    shutil.copy2(os.path.join(url_base, file_name, file2, file3, file4), 'nodulo')
                            else:
                                pass
                                #cont_nonolo +=1
                                #if cont_nonolo < 20:
                                    #shutil.copy2(os.path.join(url_base, file_name, file2, file3, file4), 'sem nodulo')


import shutil
import random

def leitura_paste(path):
    cont_nodulo = 0
    cont_nonolo = 0
    url_base = path
    mylist = np.arange(50)

    for file_name in os.listdir(url_base):
        
        
        for file2 in os.listdir(url_base+'/'+file_name):
            for file3 in os.listdir(url_base+'/'+file_name+'/'+file2):
                for file4 in os.listdir(url_base+'/'+file_name+'/'+file2+'/'+file3):
                    if file4.endswith('.dcm'):
                        if random.choices(mylist) == [7]:
                            img = im.imread(os.path.join(url_base, file_name, file2, file3, file4)).astype(np.float64)
                            img_dic = img.meta
                            scan = str(img_dic['SeriesNumber'])
                            slice_n = str(img_dic['InstanceNumber'])
                            referencia = scan+slice_n
                            if to_add[to_add['referencia'] == referencia]['has_tumor'].any():
                                pass
                                #cont_nodulo +=1
                                #if cont_nodulo < 20:
                                    #shutil.copy2(os.path.join(url_base, file_name, file2, file3, file4), 'nodulo')
                            else:
                                cont_nonolo +=1
                                if cont_nonolo < 25:
                                    shutil.copy2(os.path.join(url_base, file_name, file2, file3, file4), 'sem nodulo')
