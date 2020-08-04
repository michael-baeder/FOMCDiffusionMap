# -*- coding: utf-8 -*-
"""
    Collection of functions for downloading FOMC statements.
"""

from bs4 import BeautifulSoup
import requests
import datetime
import re
from utils import throttle, flatten_list

# Base URLs
page_root = "https://www.federalreserve.gov"
to_url = lambda url: ''.join([page_root,url])

start_page_current =  "/monetarypolicy/fomccalendars.htm"
start_page_historical = "/monetarypolicy/fomc_historical_year.htm"

# Wrapper for get to space out queries and apply BeautifulSoup
@throttle(milliseconds=500)
def get_soup(url): 
    return BeautifulSoup(requests.get(to_url(url)).content, 'html.parser')

def filter_field(soup, find_all_args, regex):
    """
        Filter a field from HTML based on whether the text inside matches
        a regex.
    """
    return [l for l in soup.find_all(*find_all_args)
            if regex.match(l.text)]
    
def filter_links(soup, regex):
    """
        Use-case of filter_field specifically for pulling links from a page.
    """
    links = filter_field(soup, 'a', regex)
    return [l.get('href') for l in links]

def get_links_historical(start = start_page_historical):
    """
        Get list of links to statements from the FOMC archive.
    """
    start_soup = get_soup(start)
    # Only links that are a year go to historical pages
    hist_pages = filter_links(start_soup, re.compile('([1-3][0-9]{3})'))
    # Only follow links that literally say "Statement"
    links = [filter_links(get_soup(page), re.compile('Statement'))
             for page in hist_pages]
    links = flatten_list(links)
    return links
    
def get_links_current(start = start_page_current):
    """
        Get list of links from the current FOMC calendar.
    """
    # Get links from the table
    start_soup = get_soup(start)
    tbl_div = start_soup.find_all('div',{'class':'col-xs-12 col-md-4 col-lg-2'})
    tbl_links = flatten_list(
            [filter_links(d,re.compile('HTML'))
             for d in tbl_div
             if filter_field(d,['strong'],re.compile('Statement:'))])
        
    # There are also some direct links labeled "Statement"; get them too
    direct_links = filter_links(start_soup, re.compile('^Statement$'))
    return tbl_links + direct_links

def get_links(start_current = start_page_current, start_historical=start_page_historical):
    return get_links_historical(start_page_historical) + get_links_current(start_page_current)

def get_statement_date(doc_soup):
    """
        Determine the date associated with a given statement. Current files
        have a special tag, while old documents do not.
    """
    statement_date = [datetime.datetime.strptime(p.text, '%B %d, %Y')
        for p in doc_soup.findAll("p", {"class": "article__time"})]
    if not statement_date:
        statement_date = [datetime.datetime.strptime(i.text, 'Release Date: %B %d, %Y') for i in doc_soup.find_all('i')]
    return statement_date[0]
        
def get_statement_text(doc_soup):
    return '\n'.join([p.text for p in doc_soup.find_all('p')])