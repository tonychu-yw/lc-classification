import os
import shutil
import zipfile
import json
import xml.etree.ElementTree as ET
import time
import numpy as np
import pandas as pd

#-----------------------------------------------------------------

def unzip_books(source_folder, dest_folder):
    """
    Arguments:
        source_folder:  folder path of downloaded gutenberg text files
        dest_folder:    folder path to store the selected books (do not set this same as source_folder)
    Return:
        None            (move files directly to the destination folder)
    """

    os.chdir(source_folder)  # set directory
    onlyfiles = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]  # read file names

    # keep only one version per file
    # may lose some files that are not number-labeled while filtering
    for i in range(len(onlyfiles[:-1])):

        fileName  = str(i+1) + ".zip"
        fileName0 = str(i+1) + "-0.zip"
        fileName8 = str(i+1) + "-8.zip"

        if fileName in onlyfiles[:-1]:
            shutil.copy2(fileName, dest_folder)
        elif fileName0 in onlyfiles[:-1]:
            shutil.copy2(fileName0, dest_folder)
        elif fileName8 in onlyfiles[:-1]:
            shutil.copy2(fileName8, dest_folder)

    # set dest as dir folder
    os.chdir(dest_folder)
    folder = os.listdir(dest_folder)

    # unzip and delete zip files
    for item in folder:
        if item.endswith(".zip"):
            try:
                with zipfile.ZipFile(item, "r") as zip_ref:
                    zip_ref.extractall()
            except:
                pass
            os.remove(os.path.join(dest_folder, item))

    # note that 17421, 17422, 17423, 17424 are musics

#-----------------------------------------------------------------

def clean_books(books_folder, output_dir, n_tokens=12000):
    """
    Arguments:
        books_folder:   folder of gutenberg text files
        output_dir:     file path/name to store the aggregated books (.json)
        n_tokens:       number of tokens to grab (first n tokens)
    Return:
        None            (save output file directly)
    """

    #import nltk
    #nltk.download('punkt')
    #from nltk.tokenize import word_tokenize
    #from nltk.corpus import stopwords

    books = []
    books_idx = []
    print("Locating files ...")
    list_docs = os.listdir(books_folder)

    print("Reading files ...")
    # import and clean books of all classes
    iter = 0
    start = time.time()
    for doc in list_docs:
        idx = doc.replace('-0', '').replace('-8', '')[:-4]
        books_idx.append(idx)
        try:
            with open(os.path.join(books_folder, doc), 'r') as f:
                books.append(f.read())
        except UnicodeDecodeError:
            with open(os.path.join(books_folder, doc), 'rb') as f:
                books.append(f.read())
        #if iter%1000 == 0:
        #    end = time.time()
        #    print(iter, "- time:", round((end-start)//60), "min",  round((end-start)%60), "sec")
        iter += 1
    end = time.time()
    print("Read time:", round((end-start)//60), "min",  round((end-start)%60), "sec")

    # slicing books - not using anymore to remain text consistancy
    """
    # simple slicer to remove heads and tails of books
    books_sliced = []
    print("Start slicing files ...")
    start = time.time()
    for i in range(len(books)):
        # set encoding
        if type(books[i]) == bytes:
            text = books[i].decode("latin-1")
        else:
            text = books[i]
        # begin slicing
        begin = text.find("Title:")
        ends = ["END OF THIS PROJECT GUTENBERG", "END OF THE PROJECT GUTENBERG",
               "End of Project Gutenberg", "End of the Project Gutenberg"]
        success = 0
        for e in ends:
            end = text.find(e)
            if (begin !=-1) and (end !=-1):
                books_sliced.append(text[begin:end])
                success += 1
                break
        if success == 0:
            print('Not sliced: ' + books_idx[i])
            books_sliced.append(text)
        #if i%1000 == 0:
        #    end = time.time()
        #    print(i, "- time:", round((end-start)//60), "min",  round((end-start)%60), "sec")
    end = time.time()
    print("Slice time:", round((end-start)//60), "min",  round((end-start)%60), "sec")
    """

    # clean text
    print("Cleaning files ...")
    books_clean = []
    #stop_words = set(stopwords.words('english'))
    iter = 0
    start = time.time()
    for book in books:
        if type(book) == bytes:
            text = book.decode("latin-1")
        else:
            text = book
        tokens = text.split()
        tokens = [w.lower() for w in tokens]
        #words = [w for w in tokens if not w in stop_words]
        books_clean.append(' '.join(tokens[:min(n_tokens,len(tokens))]))
        #if iter%1000 == 0:
        #    end = time.time()
        #    print(iter, "- time:", round((end-start)//60), "min",  round((end-start)%60), "sec")
        iter += 1
    end = time.time()
    print("Clean time:", round((end-start)//60), "min",  round((end-start)%60), "sec")
    #print("--- Cleaning files complete! ---")

    # save cleaned books
    books_json = {int(k): v for k, v in zip(books_idx, books_clean)}
    print("Saving files ...")
    with open(output_dir, "w") as f:
        json.dump(books_json, f)

#-----------------------------------------------------------------

def get_metadata(source_folder, dest_dir):
    """
    Arguments:
        source_folder:  file path of the metadata files (.json)
        dest_dir:       file path/name to store the cleaned metadata (.json)
    Return:
        None            (save output file directly)
    """

    # get file
    books = {}
    for file in os.listdir(source_folder):
        tree = ET.parse(os.path.join(source_folder, file))
        root = tree.getroot()
        ebook = root.find('{http://www.gutenberg.org/2009/pgterms/}ebook')
        record = {}

        # find title
        try:
            title = ebook.find('{http://purl.org/dc/terms/}title')
            record['title'] = title.text
        except AttributeError:
            pass

        # find author
        try:
            creator = []
            for c in ebook.findall('{http://purl.org/dc/terms/}creator'):
                a = c.find('{http://www.gutenberg.org/2009/pgterms/}agent')
                author = a.find('{http://www.gutenberg.org/2009/pgterms/}name')
                creator.append(author.text)
            record['creator'] = creator
        except AttributeError:
            pass

        # find subjects
        try:
            subjects = []
            for s in ebook.findall('{http://purl.org/dc/terms/}subject'):
                subject = s.find('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description')
                d = subject.find('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}value')
                subjects.append(d.text)
            record['subjects'] = subjects
        except AttributeError:
            pass

        # aggregate to final record
        books[file] = record

    # save metadata
    with open(dest_dir, 'w') as file:
        json.dump(books, file)

#-----------------------------------------------------------------

def clean_metadata(metadata_dir, books_folder, output_dir):
    """
    Arguments:
        metadata_dir:   file path of the metadata file (.json)
        books_folder:   folder path of the gutenberg books
        output_dir:     file path/name to store the cleaned metadata (.json)
    Return:
        None            (save output file directly)
    """

    # import data
    df = pd.read_json(metadata_dir)  # import data
    df = df.T

    # clean index and document names
    df.reset_index(level=0, inplace=True)
    df.rename(columns={'index':'document'}, inplace=True)
    df.document = df.document.apply(lambda x: x[2:-4])

    # remove metadata not in the downloaded books
    list_docs = os.listdir(books_folder)
    docs_no = [x.replace('-0', '').replace('-8', '')[:-4] for x in list_docs]
    df_final = df[df.document.isin(docs_no)]

    # further adjust order and column name for consistancy
    df_final['document'] = [int(x) for x in df_final['document']]
    df_final = df_final.sort_values(by=['document']).reset_index(drop=True)  # sort metadata
    df_final.columns = ['id', 'title', 'creator', 'subjects']

    # save cleaned metadata
    df_final.to_json(output_dir)

#-----------------------------------------------------------------

def remove_suffix(metadata):
    """
    Arguments:
        metadata:   pandas dataframe
    Return:
        metadata:   cleaned pandas dataframe
    """

    new_subjects = []
    for row in metadata.subjects:
        clean_subject = []
        for subject in row:
            suf = subject.find(" --")
            if suf == -1:
                sub = subject
            else:
                sub = subject[:suf]
            clean_subject.append(sub)
        if clean_subject == []:
            clean_subject.append('')
        new_subjects.append(clean_subject)

    # save parsed subjects
    metadata['subjects_new'] = new_subjects

    return metadata
