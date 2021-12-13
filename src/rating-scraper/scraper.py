"""File that defines the scraper to parse books' rating from 'lirtes.ru'"""

import requests
from bs4 import BeautifulSoup
import os
import csv


def get_files():
    """get files from corpus"""
    # Changing directory to corpus location
    if '/rating-scraper' in os.path.abspath(os.curdir):
        os.chdir('..')
    list_of_files = os.listdir('corpus txt')
    return list_of_files


def get_mark(book_name):
    """get marks of a book from 'litres'"""
    search_url = 'https://www.litres.ru/pages/rmd_search_arts/?q=' + book_name
    search_response = requests.get(search_url)
    soup = BeautifulSoup(search_response.text, 'lxml')
    # Checking if the book has been found
    if soup.find('div', class_="ab-container b_interested__book") is not None:
        return "Not found"
    # Playing safe with the parser
    try:
        book_link = soup.find('a', class_="art-item__name__href").get('href')
    except AttributeError:
        return "Not found"
    book_url = 'https://www.litres.ru' + book_link
    book_response = requests.get(book_url)
    soup = BeautifulSoup(book_response.text, 'lxml')
    marks = soup.find_all("div", class_='rating-number bottomline-rating')
    all_votes = soup.find_all('div', class_='votes-count bottomline-rating-count')
    # Comparing names of a book
    if book_name in soup.find('div', class_="biblio_book_name biblio-book__title-block").text:
        pass
    else:
        return "COULD BE AN ERROR"
    # Checking for livelib marks 
    if len(marks) > 1:
        litres_mark, livelib_mark = marks[0].text, marks[1].text
        litres_votes, livelib_votes = all_votes[0].text, all_votes[1].text
    else:
        litres_mark, livelib_mark = marks[0].text, 'None'
        litres_votes, livelib_votes = all_votes[0].text, 'None'
    return litres_mark + '-' + litres_votes + '-' + livelib_mark + '-' + livelib_votes


def scrape_marks():
    """write parsed data to csv-file"""
    with open("marks.csv", "w", newline="") as mark_file:
        writer = csv.writer(mark_file, quotechar='"')
        for book_name in get_files():
            # Tracing the book to be parsed
            print(book_name)
            # Example: |War and Peace - Leo Tolstoy.txt| -> War and Peace, Leo Tolstoy, txt
            if not len(book_name.split('-')) > 1:
                writer.writerow('not found')
                continue
            title_and_author = [book_name.split('-')[0], book_name.split('-')[1].split('.')[0]]
            writer.writerow(title_and_author +  get_mark(book_name.split(".")[0]).split('-'))


def filter_marks():
    """cut unknown books"""
    # Changing directory to scraper location
    if not '/rating-scraper' in os.path.abspath(os.curdir):
        os.chdir(os.path.abspath('rating-scraper'))
        with open("evaluated_texts.csv", "w", newline="") as eval_file:
            with open("marks.csv", "r", newline="") as mark_file:
                reader = csv.reader(mark_file, quotechar='"')
                writer = csv.writer(eval_file, quotechar='"')
                for row in reader:
                    if not 'Not found' in row:
                        print(','.join(row))
                        writer.writerow(row)


# def check_marked():
#     if not '/rating-scraper' in os.path.abspath(os.curdir):
#         os.chdir(os.path.abspath('rating-scraper'))
#         with open("evaluated_texts.csv", "w", newline="") as eval_file:
#             reader = csv.reader(eval_file, quotechar='"')

# print(get_mark("Человек для особых поручений"))
