# -*- coding: utf-8 -*-
"""
    Download historical FOMC statements and save the raw text. The filename
    is the date of the statement.
"""

import os, datetime

# Move to the folder that contains all of the project code
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

import FOMCStatements as FOMC

# Move to the folder where you want to save the raw text ouput
os.chdir('../FOMCStatements')

all_links = FOMC.get_links()
for l in all_links:
    doc_soup = FOMC.get_soup(l)
    doc_date = FOMC.get_statement_date(doc_soup)
    doc_text = FOMC.get_statement_text(doc_soup)
    doc_name = datetime.datetime.strftime(doc_date,'%Y%m%d') + '.txt'
    with open(doc_name,"w+",encoding='utf-8') as file:
        file.write(doc_text)