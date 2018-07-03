import PIL
import lmdb
import sys


def _get_image_transaction(db):
    try:
        trans=db.begin(write=True)
    except:
        print('!!!!!!!!!!!!!\n\nTransaction begin error',sys.exc_info()[0])
        raise

    return trans

def _db_just_put(txn,key,value):
    try:
        txn.put(key,value)
    # except lmdb.MapFullError:
    #     txn.abort()
    #     curr_limit = db.info()['map_size']
    #     new_limit = curr_limit*2
    #     print '>>> Doubling LMDB map size to %sMB ...' % (new_limit/1024**2,)
    #     db.set_mapsize(new_limit) # double it
    except:
        txn.abort()
        print('###########\n\nTransaction Error', sys.exc_info()[0],type(key),type(value))
        raise

def _db_commit_sync(db,txn):
    try:
        txn.commit()
        db_status=db.sync(True)

    # except lmdb.MapFullError:
    #     txn.abort()
    #     # double the map_size
    #     curr_limit = db.info()['map_size']
    #     new_limit = curr_limit*2
    #     print '>>> Doubling LMDB map size to %sMB ...' % (new_limit/1024**2,)
    #     db.set_mapsize(new_limit) # double it
    except:
        txn.abort()
        print('###########\n\nTransaction Error', sys.exc_info()[0])
        raise

    return db_status

def _write_to_lmdb(db, db_dikt):
    """
    Write (key,value) to db
    """
    success = False
    while not success:
        txn = db.begin(write=True)
        try:
            for key,value in db_dikt:
                txn.put(key, value)
            # txn.put(str(int(key)+67467474574), value)

            txn.commit()
            success = True
        except lmdb.MapFullError:
            txn.abort()

            # double the map_size
            curr_limit = db.info()['map_size']
            new_limit = curr_limit*2
            print( '>>> Doubling LMDB map size to %sMB ...' % (new_limit/1024**2,))
            db.set_mapsize(new_limit) # double it
    return db.sync(True)

def _save_mean(mean, filename):
    """
    Saves mean to file

    Arguments:
    mean -- the mean as an np.ndarray
    filename -- the location to save the image
    """
    if filename.endswith('.binaryproto'):
        raise Exception('Can\' do binary...')

    elif filename.endswith(('.jpg', '.jpeg', '.png')):
        image = PIL.Image.fromarray(mean)
        image.save(filename)
    else:
        raise ValueError('unrecognized file extension')


def _save_mean(mean, filename):
    """
    Saves mean to file

    Arguments:
    mean -- the mean as an np.ndarray
    filename -- the location to save the image
    """
    if filename.endswith('.binaryproto'):
        raise Exception('Can\' do binary...')

    elif filename.endswith(('.jpg', '.jpeg', '.png')):
        image = PIL.Image.fromarray(mean)
        image.save(filename)
    else:
        raise ValueError('unrecognized file extension')



def _reduce_lmdbsize(lmdb_loc):
    with lmdb.open(lmdb_loc) as lm_env:
        # print(lm_env.stat())
        lm_stat = lm_env.stat()
        needed_size = (lm_stat['psize'] * (lm_stat['overflow_pages']
                                           + lm_stat['branch_pages'] + lm_stat['leaf_pages'])) / 1024 ** 3

        needed_size *= 1.05

        lm_env.set_mapsize(int(  needed_size*1024**3))

    return needed_size