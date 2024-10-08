import lib_misc         as misc
from lib_matlab         import PWD
from datetime           import date


def merge_if_fail():
    server                          = 0
    Date                            = date.today().strftime("%Y-%m-%d")
    tx,fibre,rx,saving              = misc.init_dict(server)
    
    path = PWD(show=False)+f'/data-{Date[2:]}'
    misc.remove_n_characters_from_filenames(path, 20)
    
    misc.organise_files(path)
    
    
merge_if_fail()