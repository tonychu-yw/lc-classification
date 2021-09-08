# import packages if not exist
def import_or_install(package):
    try:
        __import__(package)
    except ImportError:
        pip.main(['install', package])